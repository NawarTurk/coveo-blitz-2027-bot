"""
Fast-clear bot = the hybrid (heuristic ranks spots -> small CNN picks the final spot), with a new REWARD:

  points of a kill  = 160 / (1 + age/30)          (measured from your logs; was 1/(1+age/20))
  x multi-kill      = 1 + 0.5 (k-1)                (confirmed exactly)
  + CLEAR_BONUS     if the meteor kills the last dinos on the board (next wave comes 2 ticks later)
  strategy-paragraph claims OFF by default (logs showed they're false) -> USE_STRATEGY_CLAIMS = True to test

Plus 2 rules:
  - never leave a meteor slot empty while a dino is alive and a shot has any value
  - meteors first: volcanoes only on ticks when all 3 meteor slots are busy
  - but if every living dino is already under a falling meteor, hold the slot for the next wave

Same score units in the heuristic and the CNN, so both agree on what a good shot is.

Use it:  cp fast_clear.py bot.py
Needs:   hybrid_planner.py, cnn_model.py, heuristic.py, state_encoder.py, models/cnn_lookahead_1/model.pt
Scores:  logs/scores_fastclear.csv
"""
from game_message import *
import gc, time

import hybrid_planner as HY

C, H = HY.C, HY.H

# ---- knobs ---------------------------------------------------------------
LOG_GAMES = False
AGE_HALF = 30.0                 # real points: 160 / (1 + age/30)
CLEAR_BONUS = 4.0               # in "newborn kills": worth ~4 fresh kills to finish the board
HOLD_WHEN_ALL_DOOMED = True     # don't waste a slot on dinos that are already under a meteor
USE_STRATEGY_CLAIMS = False     # NE bonus, Triceratops CCW, raptor bait, double bounce, funnels, repeat impacts
TICK_BUDGET_MS = 75             # server is ~2x slower than a Mac: v6 missed 4 ticks with 90/115
HARD_STOP_MS = 100
# --------------------------------------------------------------------------

C.BOT_NAME = "fastclear"
C.TICK_BUDGET_MS, C.HARD_STOP_MS = TICK_BUDGET_MS, HARD_STOP_MS
C.LOG_GAMES = LOG_GAMES

# ---- same reward in both scorers ----
C.AGE_HALF = AGE_HALF            # CNN scorer + its spot ranking
H.AGE_HALF = AGE_HALF            # heuristic scorer
H.BOARD_CLEAR_W = CLEAR_BONUS    # heuristic: bonus * P(every dino dead)
H.MEDIUM_DINOS = 99              # ...considered at any dino count
H.METEOR_MIN_SCORE = 0.0         # never idle: fire any shot with value >= 0
H.VOLCANO_MIN_SCORE = float("inf")   # meteors first: volcano only when all meteor slots are busy
if not USE_STRATEGY_CLAIMS:
    H.NE_METEOR_BONUS = H.NE_VOLCANO_BONUS = 0.0
    H.TRI_CCW_PROB = 0.5
    H.TRI_OFFSET_BONUS = 0.0
    H.RAPTOR_BAIT_BONUS = H.FUTURE_RAPTOR_W = 0.0
    H.DOUBLE_BOUNCE_WEIGHT = 0.0
    H.FUNNEL_BONUS = 0.0
    H.REPEAT_IMPACT_BONUS = 0.0


class Bot(HY.Bot):
    def __init__(self):
        super().__init__()
        self.used["held"] = 0
        # Python's garbage collector caused rare ~200 ms pauses mid-tick. Memory use is the same without it
        # (tested over full games), so freeze what's loaded and switch it off.
        gc.collect(); gc.freeze(); gc.disable()

    # ---------------------------------------------------------------- hold rule
    def all_doomed(self, s):
        """every living dino is standing in the blast of a meteor that is already falling"""
        zone = {(p["x"], p["y"]) for m in s["meteors"] for p in m["impactedTiles"]}
        return bool(s["dinosaurs"]) and all((d["position"]["x"], d["position"]["y"]) in zone for d in s["dinosaurs"])

    def choose_actions(self, s):
        if HOLD_WHEN_ALL_DOOMED and self.all_doomed(s):
            self.update_heuristic_state(s)
            self.used["held"] += 1
            return []
        return super().choose_actions(s)

    def heuristic_plan(self, ctx, t0):
        if len(ctx.meteors) < ctx.maxM:      # a meteor slot is free -> volcano can't win, skip scoring it (saves time)
            ctx.maxV = 0
        return super().heuristic_plan(ctx, t0)

    # ---------------------------------------------------------------- CNN planner with the new reward
    def plan_meteor(self, s):
        c = s["constants"]
        m = s["map"]
        self.ox, self.oy, self.h, self.w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        self.delay, self.shift = c["meteorDelay"], c["mapShiftInterval"]
        self.foot = self.footprint(c["meteorRadius"])
        ranked = self.candidates(s)
        if not ranked:
            return None
        steps = self.delay
        left = C.TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 5
        n = int(left / (steps * self.unit_ms)) if self.unit_ms > 0 else C.MAX_CANDIDATES
        if n < C.MIN_CANDIDATES:
            self.n_cand = 0
            self.fallbacks += 1
            return ranked[0]
        n = min(C.MAX_CANDIDATES, n)
        self.n_cand = n
        cands = ranked[:n]
        t_plan = time.perf_counter()
        self.static, _ = C.encode_state(s, None, {}, self.foot, self.H, self.W)

        mets0 = [{"tiles": [(p["x"], p["y"]) for p in me["impactedTiles"]], "turnsUntilImpact": me["turnsUntilImpact"],
                  "mine": False} for me in s["meteors"]]
        sims = []
        for cx, cy in cands:
            dinos = {d["id"]: dict(d, position=dict(d["position"])) for d in s["dinosaurs"]}
            sims.append({"target": (cx, cy), "dinos": dinos, "meteors": [dict(mm) for mm in mets0],
                         "prev": dict(self.prev_pos), "kills": [], "conf": {i: 1.0 for i in dinos},
                         "landed": False, "survivors": None})

        tick = s["currentTick"]
        for step in range(self.delay + 2):
            n_active = sum(1 for sm in sims if not sm["landed"])
            if 1000 * (time.perf_counter() - self.tick_t0) + self.unit_ms * n_active > C.HARD_STOP_MS:
                self.fallbacks += 1
                self.unit_ms *= 1.05
                return ranked[0]
            active = [sm for sm in sims if not sm["landed"]]
            if not active:
                break
            for sm in active:
                self.land_meteors(sm)
                if sm["landed"] and sm["survivors"] is None:
                    sm["survivors"] = [sm["conf"].get(i, 1.0) for i in sm["dinos"]]
            active = [sm for sm in active if not sm["landed"]]
            if not active:
                break
            Xs, scs = [], []
            for sm in active:
                X, sc = self.encode_sim(sm, tick + step, sm["target"] if step == 0 else None)
                Xs.append(X); scs.append(sc)
            probs = self.predict(Xs, scs)
            for k, sm in enumerate(active):
                moves = {}
                for i_d, d in sm["dinos"].items():
                    i, j = d["position"]["y"] - self.oy, d["position"]["x"] - self.ox
                    if 0 <= i < self.h and 0 <= j < self.w:
                        p = probs[k, :, i, j]
                        b = int(p.argmax())
                        moves[i_d] = self.moves[b]
                        if C.USE_CONFIDENCE:
                            sm["conf"][i_d] *= float(p[b])
                self.apply_moves(sm, moves)
                self.tick_down(sm, step)

        unit = 1000 * (time.perf_counter() - t_plan) / max(1, len(cands) * steps)
        self.unit_ms = 0.7 * self.unit_ms + 0.3 * unit
        best_t, best_v = None, -1
        for sm in sims:
            v = self.reward(sm)
            if v > best_v:
                best_t, best_v = sm["target"], v
        return best_t if best_v > 0 else cands[0]

    def reward(self, sm):
        """points (in newborn-kill units) x multi-kill + bonus if nothing is left alive after impact"""
        kills = sm["kills"]
        if not kills:
            return 0.0
        exp_kills = sum(cf for _, cf in kills)
        v = sum(C.age_value(a) * cf for a, cf in kills) * (1 + 0.5 * max(exp_kills - 1, 0))
        if sm["landed"] and not sm["survivors"]:              # board empty after our impact
            p_all = 1.0
            for _, cf in kills:
                p_all *= cf
            v += CLEAR_BONUS * p_all
        return v

    def finish(self, s):
        super().finish(s)
