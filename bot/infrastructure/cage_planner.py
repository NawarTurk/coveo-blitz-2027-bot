"""
CAGE bot, version A (small 1-step CNN only). Built on fast_clear.py: same new reward
(points 160/(1+age/30), multi-kill bonus, wave-clear bonus, hold rule, no strategy-paragraph claims).

What your flee analysis showed:  dinos see meteors and run (80% step away at once, mostly straight),
91% of meteors kill nothing, BUT a dino with few free escape tiles is caught far more often:
    free escape tiles   0     1-3    4-7    8-14   15+
    caught             79%   47%    26%    10%    4%

So each tick:
  METEOR (a slot is free)
    1. cage spots: for every dino, all 13 blast centres that cover it -> count its free escape tiles
       outside that blast (walls, mountains, lava, edges and other falling meteors don't count).
       A centre on the dino's OPEN side leaves few exits -> it runs into the wall.
    2. chain spots: dinos already under a falling meteor will run 3-5 tiles away from its centre;
       a meteor landing 1-2 ticks later on those escape tiles catches them.
    3. prior = expected points from the table above (+ multi-kill), used to rank the spots.
    4. the CNN simulates the best spots (as many as fit in the time budget) -> final score =
       average of CNN reward and prior.  No time -> best prior spot.
  VOLCANO (all meteor slots busy, no volcano active; version C also: no meteor worth firing / slots held)
    5. simulate lava from the best 3 mountains, put it on the map and roll the CNN 5 ticks:
       score = dinos the lava kills + escape tiles it cuts for dinos under falling meteors.

Use it:  cp cage_planner.py bot.py
Needs:   fast_clear.py, hybrid_planner.py, cnn_model.py, heuristic.py, state_encoder.py, models/cnn_lookahead_1/model.pt
Scores:  logs/scores_cage.csv
"""
from game_message import *
import time

import numpy as np

import fast_clear as FC

C, H = FC.C, FC.H

# ---- knobs ---------------------------------------------------------------
LOG_GAMES = False
PRIOR_W = 0.5            # final score = (1-PRIOR_W) * CNN reward + PRIOR_W * prior
CHAIN_W = 0.6            # how much we trust the "they run straight away" escape guess
MIN_SHOT = 0.01          # fire if the best spot is worth at least this
VOLC_CANDS = 3           # mountains simulated with the CNN
LAVA_STEPS = 5           # ticks of CNN rollout with lava on the map
VOLC_MIN = 0.1           # trigger the volcano only if worth this (when meteor slots are full)
VOLC_MIN_IDLE = 0.0      # ...and this when the tick would otherwise be wasted (no good shot / held)
IDLE_VOLCANO = False     # version C sets True: try the volcano on held ticks and when no meteor is worth firing
CUT_W = 0.5              # value of cutting escape tiles of dinos under a falling meteor
# caught rate vs free escape tiles (from flee_behavior.py on 550 games)
TRAP = [(0, 0.79), (3, 0.47), (7, 0.26), (14, 0.10), (999, 0.036)]
# --------------------------------------------------------------------------

C.BOT_NAME = "cage"
C.LOG_GAMES = LOG_GAMES


def p_trap(exits):
    for hi, p in TRAP:
        if exits <= hi:
            return p
    return TRAP[-1][1]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Bot(FC.Bot):
    def __init__(self):
        super().__init__()
        self.used = {"meteor": 0, "fallback": 0, "volcano": 0, "nothing": 0, "held": 0}
        self.prior = {}

    # ================================================================ per-tick geometry
    def setup(self, s):
        c, m = s["constants"], s["map"]
        self.t = s["currentTick"]
        self.D, self.R = c["meteorDelay"], c["meteorRadius"]
        self.ox, self.oy, self.h, self.w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        self.delay, self.shift = c["meteorDelay"], c["mapShiftInterval"]
        self.foot = self.footprint(self.R)
        self.free = set()
        for i, row in enumerate(m["tiles"]):
            for j, tl in enumerate(row):
                if not (tl["isImpassable"] or tl["isMountain"] or tl["hasLava"]):
                    self.free.add((j + self.ox, i + self.oy))
        # falling meteors: (landing tick, centre, tiles)
        self.falling = [(self.t + me["turnsUntilImpact"] - 1, (me["target"]["x"], me["target"]["y"]),
                         frozenset((p["x"], p["y"]) for p in me["impactedTiles"])) for me in s["meteors"]]
        self.dinos = [{"id": d["id"], "age": d["age"], "pos": (d["position"]["x"], d["position"]["y"])}
                      for d in s["dinosaurs"] if self.in_window(d["position"]["x"], d["position"]["y"])]
        moves = self.D - 1                                   # moves a dino gets after it sees the meteor
        for d in self.dinos:
            x, y = d["pos"]
            d["reach"] = [q for q in ((x + a, y + b) for a in range(-moves, moves + 1)
                                      for b in range(-moves, moves + 1) if abs(a) + abs(b) <= moves)
                          if q in self.free]

    # ================================================================ meteor: cage prior
    def cage_prior(self, c):
        """expected points of a meteor at c (new reward units), from the flee-analysis trap table"""
        land = self.t + self.D
        deadly = set()                                       # tiles another meteor hits around our landing
        for L, _, tiles in self.falling:
            if land - 1 <= L <= land + 1:
                deadly |= tiles
        val, E = 0.0, 0.0
        for d in self.dinos:
            if manh(d["pos"], c) > self.R:
                continue
            exits = sum(1 for q in d["reach"] if manh(q, c) > self.R and q not in deadly)
            p = p_trap(exits)
            val += p * C.age_value(d["age"] + self.D)
            E += p
        # chain: dinos under a meteor landing 1-2 ticks before ours, caught where they flee to
        for L, cen, tiles in self.falling:
            if not (1 <= land - L <= 2):
                continue
            for d in self.dinos:
                if d["pos"] not in tiles or manh(d["pos"], c) > self.R + self.D + 1:
                    continue
                esc = [q for q in d["reach"] if q not in tiles]
                if not esc:
                    continue
                base = manh(d["pos"], cen)
                w = {q: max(manh(q, cen) - base, 0) + 1 for q in esc}          # they run straight away
                tot = sum(w.values())
                hit = sum(v for q, v in w.items() if manh(q, c) <= self.R) / tot
                exits_m = len(esc)
                p = CHAIN_W * hit * (1 - p_trap(exits_m))
                val += p * C.age_value(d["age"] + self.D)
                E += p
        val *= 1 + 0.5 * max(E - 1, 0)
        return val

    def candidates(self, s):
        """cage + chain spots, ranked by prior (the CNN simulates them in this order)"""
        self.setup(s)
        spots = set()
        for d in self.dinos:
            x, y = d["pos"]
            for a, b in self.foot:
                if self.in_window(x + a, y + b):
                    spots.add((x + a, y + b))
        land = self.t + self.D
        for L, cen, tiles in self.falling:
            if 1 <= land - L <= 2:
                for d in self.dinos:
                    if d["pos"] in tiles:
                        for q in d["reach"]:
                            if q not in tiles and manh(q, cen) >= self.R + 1 and self.in_window(*q):
                                spots.add(q)
        self.prior = {p: self.cage_prior(p) for p in spots}
        return sorted(spots, key=lambda p: -self.prior[p])

    def reward(self, sm):
        return (1 - PRIOR_W) * super().reward(sm) + PRIOR_W * self.prior.get(sm["target"], 0.0)

    # ================================================================ decision
    def choose_actions(self, s):
        self.update_heuristic_state(s)
        c = s["constants"]
        if FC.HOLD_WHEN_ALL_DOOMED and self.all_doomed(s):
            self.used["held"] += 1
            if not IDLE_VOLCANO:
                return []
            return self.try_volcano(s, VOLC_MIN_IDLE)        # meteor slot kept free, but the volcano is free to use
        if s["dinosaurs"] and len(s["meteors"]) < c["maxMeteors"]:
            fb = self.fallbacks
            target = self.plan_meteor(s)
            if target is not None and self.prior.get(target, 0.0) >= MIN_SHOT and self.in_window(*target):
                self.used["fallback" if self.fallbacks > fb else "meteor"] += 1
                return [LaunchMeteorAction(target=WorldPosition(*target))]
            if not IDLE_VOLCANO:
                self.used["nothing"] += 1
                return []
            return self.try_volcano(s, VOLC_MIN_IDLE)        # no meteor worth firing -> volcano instead of nothing
        return self.try_volcano(s, VOLC_MIN)                 # all meteor slots busy

    def try_volcano(self, s, threshold):
        c = s["constants"]
        why = None
        if not s["dinosaurs"]:
            why = "v_no_dinos"
        elif len(s["volcanoes"]) >= c["maxVolcanoes"]:
            why = "v_already_active"
        elif not s["mountains"]:
            why = "v_no_mountain"
        elif 1000 * (time.perf_counter() - self.tick_t0) > C.TICK_BUDGET_MS:
            why = "v_no_time"
        else:
            v = self.plan_volcano(s, threshold)
            if v is not None:
                self.used["volcano"] += 1
                self.heur.last_volcano = v
                return [TriggerVolcanoAction(target=WorldPosition(*v))]
            why = "v_low_score"
        self.used["nothing"] += 1
        self.used[why] = self.used.get(why, 0) + 1
        return []

    # ================================================================ volcano: lava + CNN
    def plan_volcano(self, s, threshold=VOLC_MIN):
        self.setup(s)
        ctx = H.Ctx(s, self.heur.hist, self.heur.recent_impacts)
        used = set(ctx.volcanoes)
        mtns = [m for m in ctx.mountains if m not in used and ctx.in_window(*m)
                and any(manh(m, d["pos"]) <= 8 for d in self.dinos)]
        if not mtns:
            self.used["v_no_mountain_near"] = self.used.get("v_no_mountain_near", 0) + 1
            return None
        scored = []
        for m in mtns:
            lava = H.simulate_lava(ctx, m, LAVA_STEPS)
            if lava:
                scored.append((self.volcano_prior(lava), m, lava))
        if not scored:
            return None
        scored.sort(key=lambda z: -z[0])
        top = scored[:VOLC_CANDS]
        # CNN rollout with lava on the map (if there is time)
        left = C.HARD_STOP_MS - 1000 * (time.perf_counter() - self.tick_t0)
        if left < self.unit_ms * len(top) * LAVA_STEPS * 1.3:
            best = top[0]
            return best[1] if (best[0] > 0 if threshold <= 0 else best[0] >= threshold) else None
        cnn = self.lava_rollout(s, top)
        best_m, best_v = None, -1.0
        for (pr, m, _), kv in zip(top, cnn):
            v = (1 - PRIOR_W) * kv + PRIOR_W * pr
            if v > best_v:
                best_m, best_v = m, v
        if threshold <= 0:
            return best_m if best_v > 0 else None           # idle tick: any lava that kills or blocks something
        return best_m if best_v >= threshold else None

    def volcano_prior(self, lava):
        """lava reaching a dino fast + escape tiles cut for dinos under a falling meteor"""
        v = 0.0
        for d in self.dinos:
            a = lava.get(d["pos"])
            if a is not None and a <= 2:
                v += 0.5 * C.age_value(d["age"] + a)
            for L, _, tiles in self.falling:
                if d["pos"] in tiles:
                    esc = [q for q in d["reach"] if q not in tiles]
                    if esc:
                        before = p_trap(len(esc))
                        after = p_trap(sum(1 for q in esc if lava.get(q, 99) > L - self.t))
                        v += CUT_W * (after - before) * C.age_value(d["age"] + max(L - self.t, 0))
                    break
        return v

    def lava_rollout(self, s, top):
        """CNN rollout: lava arrives tile by tile, dinos see it and react; count lava kills"""
        self.static, _ = C.encode_state(s, None, {}, self.foot, self.H, self.W)
        mets0 = [{"tiles": [(p["x"], p["y"]) for p in me["impactedTiles"]], "turnsUntilImpact": me["turnsUntilImpact"],
                  "mine": False} for me in s["meteors"]]
        sims = []
        for _, m, lava in top:
            dinos = {d["id"]: dict(d, position=dict(d["position"])) for d in s["dinosaurs"]}
            sims.append({"dinos": dinos, "meteors": [dict(mm) for mm in mets0], "prev": dict(self.prev_pos),
                         "kills": [], "conf": {i: 1.0 for i in dinos}, "landed": False, "lava": lava, "lk": []})
        tick = s["currentTick"]
        lava_ch = C.CH["lava"]
        for k in range(LAVA_STEPS):
            Xs, scs = [], []
            for sm in sims:
                self.land_meteors(sm)
                now = {q for q, a in sm["lava"].items() if a <= k}
                for i in [i for i, d in sm["dinos"].items() if (d["position"]["x"], d["position"]["y"]) in now]:
                    sm["lk"].append((sm["dinos"][i]["age"], sm["conf"].get(i, 1.0)))
                    del sm["dinos"][i]
                X, sc = self.encode_sim(sm, tick + k, None)
                for (qx, qy) in now:
                    i, j = qy - self.oy, qx - self.ox
                    if 0 <= i < self.h and 0 <= j < self.w:
                        X[lava_ch, i, j] = 1
                Xs.append(X); scs.append(sc)
            probs = self.predict(Xs, scs)
            for n, sm in enumerate(sims):
                moves = {}
                for i_d, d in sm["dinos"].items():
                    i, j = d["position"]["y"] - self.oy, d["position"]["x"] - self.ox
                    if 0 <= i < self.h and 0 <= j < self.w:
                        p = probs[n, :, i, j]
                        b = int(p.argmax())
                        moves[i_d] = self.moves[b]
                        sm["conf"][i_d] *= float(p[b])
                self.apply_moves(sm, moves)
                self.tick_down(sm, -1)                       # -1: no new meteor of ours in this rollout
        out = []
        for sm in sims:
            E = sum(cf for _, cf in sm["lk"])
            out.append(sum(C.age_value(a) * cf for a, cf in sm["lk"]) * (1 + 0.5 * max(E - 1, 0)))
        return out

    def finish(self, s):
        C.Bot.finish(self, s)
        print("decisions:", self.used)
