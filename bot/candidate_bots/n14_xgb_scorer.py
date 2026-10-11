"""
N14 xgb scorer = N7 targeting + N12 mode logs + ONE change: each candidate tile is scored by the learned
XGBoost model instead of N7's hand formula (0.7 * CNN4 + 0.3 * trap table).

Same candidates as N7 (cage spots + follow-ups), same CNN4, same timing safeguards. Per candidate:
  CNN4 forecast for that blast -> 33 features (scorer_features.py: CNN4 parts, trap prior, species, ages,
  escape tiles, falling meteors, terrain, mode) -> XGBoost "pts" model = expected points of the shot
  -> fire at the highest.
Models: bot/models/scorer/pts.json (ranking) + k1.json (P(kill), logged only), made by training/train_scorer.py.
Run with plain numpy (xgb_numpy.py): no xgboost library needed on the server.
If the models are missing the bot says so and plays exactly like N7.

Test (100 unseen local games): top 10% shots kill 70% (N7 formula 55%), AUC 0.92 vs 0.84.

Logs ([N14]): one line per shot when its result is known (mode, source, XGB points + P(kill), N7 score, kills);
per mode: shots, kills/shot, predicted vs real kills (is the model honest in real games?); xgb time per tick.
Use: ./scripts/ship.sh n14_xgb_scorer
"""
import os, time
from collections import Counter, defaultdict

import numpy as np

import n12_modes as N12
import scorer_features as SF
import xgb_numpy as XN

for k in N12.ON:
    N12.ON[k] = False                     # no N12 strategies: N7 candidates
N4, CB, C = N12.N4, N12.CB, N12.C
FC = CB.FC
print = N12.print

# ---- knobs ---------------------------------------------------------------
USE_XGB = True           # False = N7 formula (A/B switch)
MODEL_DIR = os.path.join(C.HERE, "models", "scorer")
SHOT_LOG = True
# --------------------------------------------------------------------------

C.BOT_NAME = "n14_xgb_scorer"
PTS = K1 = None
LOAD_MSG = ""
try:
    PTS = XN.Forest(os.path.join(MODEL_DIR, "pts.json"))
    K1 = XN.Forest(os.path.join(MODEL_DIR, "k1.json"))
    LOAD_MSG = f"models loaded: pts {PTS.T} trees, k1 {K1.T} trees"
except Exception as e:
    LOAD_MSG = f"MODELS NOT LOADED ({e!r}) -> playing as N7"
    USE_XGB = False


def plan_meteor_xgb(self, s):
    """cage_cnn4.plan_meteor, but every candidate is scored by the XGBoost points model"""
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
    X0, sc = self.base_input(s)
    vg = self.value_grid()
    names = getattr(self, "names", {})
    follow = getattr(self, "follow", set())
    best_t, best_v, best_info, done = None, -1e9, None, 0
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
        pts = PTS.predict(np.asarray(rows, np.float32))
        self.xgb_ms += 1000 * (time.perf_counter() - tx)
        for k, c in enumerate(chunk):
            v = float(pts[k])
            if v > best_v:
                n7 = 0.7 * rows[k][SF.FEATURES.index("cnn_score")] + 0.3 * rows[k][SF.FEATURES.index("prior")]
                best_t, best_v, best_info = c, v, (rows[k], v, n7)
        done += len(chunk)
    unit = 1000 * (time.perf_counter() - t_start) / max(done, 1)
    self.unit4 = 0.7 * self.unit4 + 0.3 * unit
    self.n4 = done
    self.n_cand = done
    if best_t is None:
        return ranked[0]
    self.prior[best_t] = max(self.prior.get(best_t, 0.0), 1.0)    # always fire the best spot (same as N7 in practice)
    row, v, n7 = best_info
    self.parts = (best_t, v, float(K1.predict(np.asarray([row], np.float32))[0]), n7)
    return best_t


ORIG_PLAN = CB.Bot.plan_meteor
if USE_XGB:
    CB.Bot.plan_meteor = plan_meteor_xgb       # the N7 / N6 / N4 wrappers call this through super()


class Bot(N12.Bot):
    def __init__(self):
        super().__init__()
        self.parts, self.xgb_ms, self.xgb_ticks = None, 0.0, 0
        self.m14 = defaultdict(Counter)
        self.p14 = []
        print(f"[N14] loaded n14_xgb_scorer | USE_XGB {USE_XGB} | {LOAD_MSG}")

    def choose_actions(self, s):
        self.parts = None
        return super().choose_actions(s)

    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, N12.LaunchMeteorAction):
                c = (a.target.x, a.target.y)
                st = self.m14[self.mode]
                st["shots"] += 1
                src = "follow" if c in getattr(self, "follow", ()) else "normal"
                info = self.parts[1:] if self.parts and self.parts[0] == c else None
                if info:
                    st["scored"] += 1
                    st["pred_pts"] += info[0]
                    st["pred_k"] += info[1]
                    self.xgb_ticks += 1
                self.p14.append(dict(t=s["currentTick"], L=s["currentTick"] + s["constants"]["meteorDelay"],
                                     tiles=N4.blast(c, s["constants"]["meteorRadius"]), c=c, mode=self.mode,
                                     src=src, info=info))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            keep = []
            for x in self.p14:
                if x["L"] == T - 1:
                    dead = [d for d in p["dinosaurs"] if d["id"] not in now
                            and (d["position"]["x"], d["position"]["y"]) in x["tiles"]]
                    pts = sum(160 / (1 + (d["age"] + 1) / 30) for d in dead)
                    st = self.m14[x["mode"]]
                    st["kills"] += len(dead)
                    st["hit"] += bool(dead)
                    st["pts"] += pts
                    if x["info"]:
                        st["hit_scored"] += bool(dead)
                        st["pts_scored"] += pts
                    if SHOT_LOG:
                        i = x["info"]
                        txt = (f"xgb pts {i[0]:.1f} P(kill) {i[1]:.2f} | N7 score {i[2]:.3f}" if i
                               else "fallback (no CNN4 / XGB this tick)")
                        print(f"[N14] shot t={x['t']} {x['mode']} {x['src']} at {x['c']} | {txt} | "
                              f"kills {len(dead)} pts {pts:.0f}")
                elif x["L"] >= T:
                    keep.append(x)
            self.p14 = keep
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        for mode in N12.MODES:
            m = self.m14.get(mode)
            if not m or not m["shots"]:
                continue
            sc = max(m["scored"], 1)
            print(f"[N14] mode {mode:9} | {m['shots']} shots | {m['kills']} kills {m['kills'] / m['shots']:.2f}/shot "
                  f"| XGB-scored shots {m['scored']}: predicted P(kill) {m['pred_k'] / sc:.2f} vs real "
                  f"{m['hit_scored'] / sc:.2f} | predicted pts/shot {m['pred_pts'] / sc:.1f} vs real {m['pts_scored'] / sc:.1f}")
        print(f"[N14] xgb time {self.xgb_ms / max(self.xgb_ticks, 1):.1f} ms per scored tick")
