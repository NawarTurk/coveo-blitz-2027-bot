"""
N14c rule blend = N14 (N7 candidates + XGBoost v1 scores every spot) + ONE change (Option B):
each spot's final score mixes in the RULE MODEL's opinion (behavior_model.py: 16 confirmed spec rules x
learned weights, forecast 3 moves ahead):

    final = (1 - BLEND_W) * XGBoost expected points  +  BLEND_W * rule-model expected points in the blast

Same candidates as N14, no retraining. BLEND_W = 0 plays exactly like N14.
Needs: bot/models/behavior/weights.json (training/fit_behavior.py) + N14's bot/models/scorer/.
Logs ([N14c]): how often the blend changed the pick vs XGBoost alone, and what those shots killed.
Use: ./scripts/ship.sh n14c_rule_blend
"""
import os, time
from collections import Counter

import numpy as np

import n14_xgb_scorer as N14
import scorer_features as SF

N12, N4, CB, C, FC = N14.N12, N14.N4, N14.CB, N14.C, N14.FC
print = N14.print

# ---- knobs ---------------------------------------------------------------
BLEND_W = 0.3            # share of the rule model in the final score
RULE_MAX_MS = 45         # skip the forecast on ticks already slower than this (pure XGBoost that tick)
RULE_PATH = os.path.join(C.HERE, "models", "behavior", "weights.json")
# --------------------------------------------------------------------------

C.BOT_NAME = "n14c_rule_blend"
try:
    import behavior_model as BMOD
    RULE = BMOD.BehaviorModel.load(RULE_PATH)
    RULE_MSG = "rule model loaded"
except Exception as e:
    RULE, RULE_MSG = None, f"RULE MODEL NOT LOADED ({e!r}) -> plays as N14"
I_CNN, I_PRIOR = SF.FEATURES.index("cnn_score"), SF.FEATURES.index("prior")


def plan_meteor_blend(self, s):
    """N14's plan; each spot's XGBoost points are blended with the rule model's expected points"""
    ranked = self.candidates(s)
    if not ranked:
        return None
    self.static = None
    if self.unit4 > CB.UNIT_CAP:
        self.unit4 = CB.UNIT_CAP
        self.capped += 1
    left = C.TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 3
    n = min(CB.MAX4, len(ranked), int(left / self.unit4) if self.unit4 > 0 else CB.MAX4)
    if n < 1:
        self.fallbacks += 1
        return ranked[0]
    cands = ranked[:n]
    fc = None
    if RULE is not None and BLEND_W > 0 and 1000 * (time.perf_counter() - self.tick_t0) < RULE_MAX_MS:
        t0 = time.perf_counter()
        fc = SF.rule_forecast(RULE, s, self.mt.last)
        self.r14c["ms"] += 1000 * (time.perf_counter() - t0)
        self.r14c["ticks"] += 1
    elif RULE is not None:
        self.r14c["skipped"] += 1
    X0, sc = self.base_input(s)
    vg = self.value_grid()
    names = getattr(self, "names", {})
    follow = getattr(self, "follow", set())
    best = (-1e9, None, None)                 # (final, spot, row)
    best_x = (-1e9, None)                     # XGBoost alone
    done = 0
    t_start = time.perf_counter()
    for a in range(0, len(cands), CB.CHUNK):
        if 1000 * (time.perf_counter() - self.tick_t0) + self.unit4 * CB.CHUNK > C.HARD_STOP_MS:
            if done == 0:
                self.fallbacks += 1
                return ranked[0]
            break
        chunk = cands[a:a + CB.CHUNK]
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
        pd = self.predict4(Xs, sc)
        rows = [SF.features(self, c, pd[k], vg, masks[k], names, self.mode, self.flags, follow, FC.CLEAR_BONUS)
                for k, c in enumerate(chunk)]
        tx = time.perf_counter()
        pts = N14.PTS.predict(np.asarray(rows, np.float32))
        self.xgb_ms += 1000 * (time.perf_counter() - tx)
        for k, c in enumerate(chunk):
            x = float(pts[k])
            r = SF.rule_features(fc, {(c[0] + dx, c[1] + dy) for dx, dy in self.foot})[1] if fc else 0.0
            f = (1 - BLEND_W) * x + BLEND_W * r if fc else x
            if f > best[0]:
                best = (f, c, rows[k])
            if x > best_x[0]:
                best_x = (x, c)
        done += len(chunk)
    unit = 1000 * (time.perf_counter() - t_start) / max(done, 1)
    self.unit4 = 0.7 * self.unit4 + 0.3 * unit
    self.n4 = done
    self.n_cand = done
    f, best_t, row = best
    if best_t is None:
        return ranked[0]
    self.changed = best_t != best_x[1]
    self.r14c["picks"] += 1
    self.r14c["changed"] += self.changed
    self.prior[best_t] = max(self.prior.get(best_t, 0.0), 1.0)
    self.parts = (best_t, f, float(N14.K1.predict(np.asarray([row], np.float32))[0]),
                  0.7 * row[I_CNN] + 0.3 * row[I_PRIOR])
    return best_t


if N14.USE_XGB:
    CB.Bot.plan_meteor = plan_meteor_blend


class Bot(N14.Bot):
    def __init__(self):
        super().__init__()
        self.mt = SF.MoveTracker()
        self.r14c = Counter()
        self.changed = False
        self.p14c = []
        print(f"[N14c] loaded n14c_rule_blend | {RULE_MSG} | final = {1 - BLEND_W:.1f} x XGBoost + {BLEND_W:.1f} x rule model")

    def choose_actions(self, s):
        self.mt.update(s)
        self.changed = False
        return super().choose_actions(s)

    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, N12.LaunchMeteorAction) and self.changed:
                c = (a.target.x, a.target.y)
                self.p14c.append((s["currentTick"] + s["constants"]["meteorDelay"], N4.blast(c, s["constants"]["meteorRadius"])))
                self.r14c["changed_shots"] += 1
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles in self.p14c:
                if L == T - 1:
                    self.r14c["kills"] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                              and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.p14c = [x for x in self.p14c if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        r = self.r14c
        print(f"[N14c] blend | forecast on {r['ticks']} ticks (skipped {r['skipped']}, {r['ms'] / max(r['ticks'], 1):.1f} ms/tick) "
              f"| blend changed the pick on {r['changed']} of {r['picks']} shots | those shots killed {r['kills']} "
              f"({r['kills'] / max(r['changed_shots'], 1):.2f}/shot)")
