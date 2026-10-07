"""
HEURISTIC Z1 = cage B (CNN4 for meteors, CNN1 for the volcano, old volcano rule) with MANY more meteor spots:

  1. spots: the cage + chain spots of version B, PLUS every tile within R+2 (= 4) steps of any dino
     -> catches dinos walking or fleeing into a blast, pairs/groups of dinos, chain catches further out
  2. cheap pre-filter: trap-table prior + a small "dinos that can reach this blast" score ranks them all
  3. CNN4 scores as many of the best as fit in the time budget: batch size adapts to the time left
     (bigger batches = cheaper per spot), up to MAX_Z spots per tick
  4. pick the best: same reward as B (70% CNN4, 30% trap table)

Use it:  cp bot_heuristic_z1.py bot.py
Needs:   bot.py, cage_planner.py, fast_clear.py, hybrid_planner.py, cnn_model.py, heuristic.py,
         state_encoder.py, models/cnn_lookahead_1/model.pt, models/cnn_lookahead_4/model.pt
Scores:  logs/scores_z1.csv
"""
from game_message import *
import time

import numpy as np

import cage_cnn4 as CB

CA, C, FC = CB.CA, CB.C, CB.FC

# ---- knobs ---------------------------------------------------------------
LOG_GAMES = True
RADIUS_EXTRA = 2         # spots up to meteorRadius + this from any dino
MAX_Z = 150              # max spots scored by CNN4 per tick
MIN_CHUNK, MAX_CHUNK = 8, 64
REACH_W = 0.05           # weight of the "dinos that can reach it" score in the pre-filter
# --------------------------------------------------------------------------

CA.IDLE_VOLCANO = False  # old volcano rule (the one from the 16k game)
C.BOT_NAME = "wide_search_cnn4"
C.LOG_GAMES = LOG_GAMES


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Bot(CB.Bot):
    def candidates(self, s):
        ranked = super().candidates(s)                      # B's cage + chain spots (sets geometry + prior)
        spots = set(ranked)
        rr = self.R + RADIUS_EXTRA
        for d in self.dinos:
            x, y = d["pos"]
            for a in range(-rr, rr + 1):
                k = rr - abs(a)
                for b in range(-k, k + 1):
                    if self.in_window(x + a, y + b):
                        spots.add((x + a, y + b))
        reach = self.D + self.R
        vals = [(d["pos"], C.age_value(d["age"] + self.D)) for d in self.dinos]
        self.rank = {}
        for p in spots:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
            r = 0.0
            for q, v in vals:
                m = manh(q, p)
                if m <= reach:
                    r += v / (1 + m)
            self.rank[p] = self.prior[p] + REACH_W * r
        return sorted(spots, key=lambda p: -self.rank[p])

    def plan_meteor(self, s):
        ranked = self.candidates(s)
        if not ranked:
            return None
        cands = ranked[:MAX_Z]
        X0, sc = self.base_input(s)
        vg = self.value_grid()
        land = self.t + self.D
        n_left = sum(1 for d in self.dinos
                     if not any(d["pos"] in tiles and L < land for L, _, tiles in self.falling))
        best_t, best_v, done, a = None, -1.0, 0, 0
        t_start = time.perf_counter()
        while a < len(cands):
            el = 1000 * (time.perf_counter() - self.tick_t0)
            k = min(MAX_CHUNK, int((C.TICK_BUDGET_MS - el) / max(self.unit4, 1e-3)))
            if k < MIN_CHUNK:
                if done == 0 and el + self.unit4 * MIN_CHUNK < C.HARD_STOP_MS:
                    k = MIN_CHUNK                           # always score at least a few spots if it's safe
                else:
                    break
            chunk = cands[a:a + k]
            Xs, masks = [], []
            for cx, cy in chunk:
                X = X0.copy()
                mk = np.zeros((self.H, self.W), bool)
                for dx, dy in self.foot:
                    i, j = cy + dy - self.oy, cx + dx - self.ox
                    if 0 <= i < self.h and 0 <= j < self.w:
                        X[self.i_new, i, j] = 1
                        mk[i, j] = True
                Xs.append(X); masks.append(mk)
            t0 = time.perf_counter()
            pd = self.predict4(Xs, sc)
            self.unit4 = 0.7 * self.unit4 + 0.3 * 1000 * (time.perf_counter() - t0) / len(chunk)
            for n, c in enumerate(chunk):
                P = pd[n][masks[n]]
                E = float(P.sum())
                v = float((P * vg[masks[n]]).sum()) * (1 + 0.5 * max(E - 1, 0))
                if n_left > 0:
                    v += FC.CLEAR_BONUS * min(1.0, E / n_left) ** n_left
                v = (1 - CB.PRIOR_W) * v + CB.PRIOR_W * self.prior.get(c, 0.0)
                if v > best_v:
                    best_t, best_v = c, v
            done += len(chunk)
            a += len(chunk)
        self.n4 = self.n_cand = done
        if best_t is None:
            self.fallbacks += 1
            return ranked[0]
        self.prior[best_t] = max(self.prior.get(best_t, 0.0), best_v)
        return best_t
