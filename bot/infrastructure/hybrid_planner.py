"""
Hybrid bot = heuristic + CNN.
  1. The heuristic (heuristic.py, fast) ranks meteor spots and decides meteor vs volcano.
  2. If it wants a meteor and time is left, the CNN (cnn_model.py) simulates its top spots and picks
     the one with the most predicted kills.
  3. If the CNN runs out of time, the heuristic's choice is sent instead of a blind guess.

Use it:  cp hybrid_planner.py bot.py
Needs in the folder/zip: cnn_model.py, heuristic.py, state_encoder.py, models/cnn_lookahead_1/model.pt
Scores:  logs/scores_hybrid.csv
"""
from game_message import *
import time
from collections import deque

import cnn_model as C
import heuristic as H

LOG_GAMES = False
HEURISTIC_BUDGET_MS = 35       # time the heuristic may spend ranking spots; the CNN gets the rest of the tick

C.BOT_NAME = "hybrid"
C.LOG_GAMES = LOG_GAMES
H.LOG_GAMES = False            # the hybrid writes its own log through cnn_model
H.TIME_BUDGET_MS = HEURISTIC_BUDGET_MS


class Bot(C.Bot):
    def __init__(self):
        super().__init__()                     # loads the CNN, warms it up, opens the game log
        self.heur = H.Bot()                       # heuristic state: movement history, recent impacts, last volcano
        self.h_ranked = []
        self.used = {"cnn": 0, "heuristic": 0, "volcano": 0, "nothing": 0}

    # ---------------------------------------------------------------- heuristic state (same as heuristic)
    def update_heuristic_state(self, s):
        for (x, y, turns) in self.heur.prev_meteors:
            if turns == 1:
                self.heur.recent_impacts.append((x, y, s["currentTick"] - 1))
        self.heur.prev_meteors = [(m["target"]["x"], m["target"]["y"], m["turnsUntilImpact"]) for m in s["meteors"]]
        alive = set()
        for d in s["dinosaurs"]:
            alive.add(d["id"])
            self.heur.hist.setdefault(d["id"], deque(maxlen=6)).append((d["position"]["x"], d["position"]["y"]))
        for k in [k for k in self.heur.hist if k not in alive]:
            del self.heur.hist[k]

    def heuristic_plan(self, ctx, t0):
        """heuristic.choose_action, but also returns the ranked meteor spots"""
        if not ctx.dinos:
            return [], []
        base = {d["id"]: H.predict_dino_heuristic(ctx, d, None, keep_dists=True) for d in ctx.dinos}
        scored = []
        if len(ctx.meteors) < ctx.maxM:
            for cand in H.generate_meteor_candidates(ctx, base):
                if (time.perf_counter() - t0) * 1000 > HEURISTIC_BUDGET_MS:
                    break
                scored.append((H.score_meteor(ctx, cand, base)[0], cand))
        scored.sort(key=lambda z: -z[0])
        best_m, best_ms = (scored[0][1], scored[0][0]) if scored else (None, -1e9)

        best_v, best_vs = None, -1e9
        if len(ctx.volcanoes) < ctx.maxV and ctx.mountains and (time.perf_counter() - t0) * 1000 < HEURISTIC_BUDGET_MS * 1.3:
            used = set(ctx.volcanoes)
            mtns = [m for m in ctx.mountains if m not in used and ctx.in_window(*m)]
            mtns.sort(key=lambda m: -sum(1.0 / (1 + H.manh(m, d["pos"])) for d in ctx.dinos))
            for m in mtns[:H.MAX_VOLCANO_CANDIDATES]:
                if (time.perf_counter() - t0) * 1000 > HEURISTIC_BUDGET_MS * 1.3:
                    break
                sc = H.score_volcano(ctx, m, base, self.heur.last_volcano)
                if sc > best_vs:
                    best_v, best_vs = m, sc

        straggler = len(ctx.dinos) <= H.FEW_DINOS
        floor = 0.0 if straggler else H.METEOR_MIN_SCORE
        if best_m is not None and best_ms >= floor and best_ms >= best_vs:
            action = [LaunchMeteorAction(target=WorldPosition(*best_m))]
        elif best_v is not None and best_vs >= (H.VOLCANO_MIN_SCORE if best_m is not None else H.VOLCANO_MIN_SCORE_WHEN_FULL):
            action = [TriggerVolcanoAction(target=WorldPosition(*best_v))]
        elif best_m is not None and best_ms > 0:
            action = [LaunchMeteorAction(target=WorldPosition(*best_m))]
        else:
            action = []
        return action, [p for _, p in scored]

    # ---------------------------------------------------------------- CNN uses the heuristic's ranking first
    def candidates(self, s):
        own = super().candidates(s)
        seen = set(self.h_ranked)
        return list(self.h_ranked) + [p for p in own if p not in seen]

    # ---------------------------------------------------------------- decision (called by cnn_model.get_next_move)
    def choose_actions(self, s):
        self.update_heuristic_state(s)
        ctx = H.Ctx(s, self.heur.hist, self.heur.recent_impacts)
        h_action, self.h_ranked = self.heuristic_plan(ctx, self.tick_t0)
        h_action = H.Bot.validate(h_action, ctx)

        if h_action and h_action[0].type == "LAUNCH_METEOR":
            fb_before = self.fallbacks
            target = self.plan_meteor(s)       # CNN: tries heuristic's top spots first, within the time budget
            if target is not None and self.fallbacks == fb_before:
                self.used["cnn"] += 1
                action = [LaunchMeteorAction(target=WorldPosition(*target))]
            else:
                self.used["heuristic"] += 1     # CNN had no time: trust the heuristic's pick
                action = h_action
            return H.Bot.validate(action, ctx)

        if h_action and h_action[0].type == "TRIGGER_VOLCANO":
            self.used["volcano"] += 1
            self.heur.last_volcano = (h_action[0].target.x, h_action[0].target.y)
        else:
            self.used["nothing"] += 1
        return h_action

    def finish(self, s):
        super().finish(s)
        print("decisions:", self.used)
