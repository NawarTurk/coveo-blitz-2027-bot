"""
N15 xgb v2 = N14c (N7 candidates + XGBoost scores every spot + 30% rule-model blend) with the VERSION 2 model:
  final score = (1 - BLEND_W) * XGBoost v2 points + BLEND_W * rule-model points      (BLEND_W 0.3 like N14c)
  + 3 rule-model inputs (behavior_model.py: 16 spec rules x learned weights, forecast 3 moves ahead):
    expected dinos in the blast, expected points, best single-dino chance
  + 7 wave inputs (wave type 1-10, ticks since the wave started, ticks until the next forced wave, dinos left
    in the wave, wave size, open waves, dinos left in the oldest open wave)
  + trained on local + exploration + server games
Model: bot/models/scorer_v2/ (pts.json, k1.json, features.json) from training/train_scorer.py --features v2.
The bot builds each row from features.json, so it also runs a v1 model if pointed at bot/models/scorer.

EXPLORATION (for making training games only, never on the server):
  environment variable BLITZ_EXPLORE=0.2 -> 20% of shots go to a random spot among the top EXPLORE_K by XGBoost.
  Default 0 = always the best spot. play_local.sh passes it through:
     BLITZ_EXPLORE=0.2 ./scripts/play_local.sh n15_xgb_v2 300 explore
  (works with the v1 model too, so exploration games can be made BEFORE the v2 model exists:
   BLITZ_MODEL_DIR=models/scorer BLITZ_EXPLORE=0.2 ./scripts/play_local.sh n15_xgb_v2 300 explore)

Logs: all N14 logs ([N14] shot / mode lines) + [N15] loaded line and explored-shot count.
Use: ./scripts/ship.sh n15_xgb_v2
"""
import json, os, random, time

import numpy as np

import n14_xgb_scorer as N14
import scorer_features as SF
import xgb_numpy as XN

N12, N4, CB, C, FC = N14.N12, N14.N4, N14.CB, N14.C, N14.FC
print = N14.print

# ---- knobs ---------------------------------------------------------------
MODEL_DIR = os.path.join(C.HERE, os.environ.get("BLITZ_MODEL_DIR", os.path.join("models", "scorer_v2")))
EXPLORE = float(os.environ.get("BLITZ_EXPLORE", "0"))     # share of shots sent to a random top-K spot
EXPLORE_K = 10
BLEND_W = float(os.environ.get("BLITZ_BLEND", "0.3"))     # rule-model share of the final score (N14c = 0.3, 0 = off)
RULE_PATH = os.path.join(C.HERE, "models", "behavior", "weights.json")
RULE_MAX_MS = 45         # skip the rule-model forecast on ticks already slower than this (inputs become "missing")
# --------------------------------------------------------------------------

C.BOT_NAME = "n15_xgb_v2"
try:
    PTS = XN.Forest(os.path.join(MODEL_DIR, "pts.json"))
    K1 = XN.Forest(os.path.join(MODEL_DIR, "k1.json"))
    NAMES = json.load(open(os.path.join(MODEL_DIR, "features.json")))["features"]
    IDX = [SF.ALL_FEATURES.index(n) for n in NAMES]
    LOAD_MSG = f"models loaded from {os.path.relpath(MODEL_DIR, C.HERE)}: pts {PTS.T} trees, {len(NAMES)} inputs"
    OK = True
except Exception as e:
    LOAD_MSG, OK = f"MODELS NOT LOADED from {MODEL_DIR} ({e!r}) -> playing as N14 (v1 model) if bot/models/scorer exists, else N7", False
I_CNN, I_PRIOR = SF.ALL_FEATURES.index("cnn_score"), SF.ALL_FEATURES.index("prior")
try:
    import behavior_model as BMOD
    RULE = BMOD.BehaviorModel.load(RULE_PATH)
    RULE_MSG = "rule model loaded"
except Exception as e:
    RULE, RULE_MSG = None, f"rule model NOT loaded ({e!r}) -> rule inputs missing"
USES_RULE = OK and any(n in SF.RULE_FEATURES for n in NAMES)
NEED_RULE = USES_RULE or BLEND_W > 0
I_RULE_V = SF.ALL_FEATURES.index("rule_v")


def plan_meteor_v2(self, s):
    """N14's plan, with the full v2 feature row (incl. waves) and optional exploration"""
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
    wf = self.wt.features(s)
    fc = None
    if NEED_RULE and RULE is not None and 1000 * (time.perf_counter() - self.tick_t0) < RULE_MAX_MS:
        tr = time.perf_counter()
        fc = SF.rule_forecast(RULE, s, self.mt.last)
        self.rule_ms += 1000 * (time.perf_counter() - tr)
        self.rule_ticks += 1
    elif NEED_RULE:
        self.rule_skipped += 1
    scored, done = [], 0                                   # (pts, spot, full row)
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
        rows = [SF.features(self, c, pd[k], vg, masks[k], names, self.mode, self.flags, follow, FC.CLEAR_BONUS) + wf
                + SF.rule_features(fc, {(c[0] + dx, c[1] + dy) for dx, dy in self.foot})
                for k, c in enumerate(chunk)]
        tx = time.perf_counter()
        M = np.asarray(rows, np.float32)[:, IDX]
        pts = PTS.predict(M)
        self.xgb_ms += 1000 * (time.perf_counter() - tx)
        if fc is not None and BLEND_W > 0:
            scored += [((1 - BLEND_W) * float(pts[k]) + BLEND_W * rows[k][I_RULE_V], c, rows[k])
                       for k, c in enumerate(chunk)]
        else:
            scored += [(float(pts[k]), c, rows[k]) for k, c in enumerate(chunk)]
        done += len(chunk)
    unit = 1000 * (time.perf_counter() - t_start) / max(done, 1)
    self.unit4 = 0.7 * self.unit4 + 0.3 * unit
    self.n4 = done
    self.n_cand = done
    if not scored:
        return ranked[0]
    scored.sort(key=lambda x: -x[0])
    pick = scored[0]
    self.explored = False
    if EXPLORE > 0 and len(scored) > 1 and random.random() < EXPLORE:
        pick = random.choice(scored[1:EXPLORE_K])
        self.explored = True
        self.n_explore += 1
    v, best_t, row = pick
    self.prior[best_t] = max(self.prior.get(best_t, 0.0), 1.0)      # always fire it
    p1 = float(K1.predict(np.asarray([row], np.float32)[:, IDX])[0])
    self.parts = (best_t, v, p1, 0.7 * row[I_CNN] + 0.3 * row[I_PRIOR])
    return best_t


if OK:
    CB.Bot.plan_meteor = plan_meteor_v2


class Bot(N14.Bot):
    def __init__(self):
        super().__init__()
        self.wt = SF.WaveTracker()
        self.mt = SF.MoveTracker()
        self.rule_ms, self.rule_ticks, self.rule_skipped = 0.0, 0, 0
        self.n_explore, self.explored = 0, False
        print(f"[N15] loaded n15_xgb_v2 | {LOAD_MSG} | {RULE_MSG} (model inputs: {USES_RULE}, blend {BLEND_W:.0%}) | EXPLORE {EXPLORE:.2f} (top {EXPLORE_K})")

    def choose_actions(self, s):
        try:
            self.wt.update(s)
            self.mt.update(s)
        except Exception as e:
            print("[N15] wave error:", repr(e))
        return super().choose_actions(s)

    def finish(self, s):
        super().finish(s)
        print(f"[N15] explored shots {self.n_explore} (EXPLORE {EXPLORE:.2f}) | waves seen {self.wt.n} | rule forecast "
              f"{self.rule_ms / max(self.rule_ticks, 1):.1f} ms/tick on {self.rule_ticks} ticks, skipped {self.rule_skipped}")
