"""
N14b rule spots = N14 (N7 candidates + XGBoost v1 scores every spot) + ONE change (Option A):
the RULE MODEL (behavior_model.py: 16 confirmed spec rules x learned weights) suggests extra spots.

Each tick (when there is time) the rule model forecasts where every dino will be after 3 moves.
The blast centres that would cover the most forecast dinos are added to the candidate list (up to RULE_K,
right after N6's priority follow-ups so they always get scored). N14's XGBoost then scores ALL spots as
usual and fires at the best - nothing is forced, no retraining.

Needs: bot/models/behavior/weights.json (training/fit_behavior.py) + N14's bot/models/scorer/.
Logs ([N14b]): rule spots added per tick, how often XGBoost picked one, and what those shots killed.
Use: ./scripts/ship.sh n14b_rule_spots
"""
import os, time
from collections import Counter

import n14_xgb_scorer as N14
import scorer_features as SF

N12, N4, CB, C = N14.N12, N14.N4, N14.CB, N14.C
print = N14.print

# ---- knobs ---------------------------------------------------------------
RULE_K = 6               # extra spots per tick
RULE_MIN = 0.3           # only spots whose blast holds at least this many forecast dinos
RULE_MAX_MS = 45         # skip the forecast on ticks already slower than this
RULE_PATH = os.path.join(C.HERE, "models", "behavior", "weights.json")
# --------------------------------------------------------------------------

C.BOT_NAME = "n14b_rule_spots"
try:
    import behavior_model as BMOD
    RULE = BMOD.BehaviorModel.load(RULE_PATH)
    RULE_MSG = "rule model loaded"
except Exception as e:
    RULE, RULE_MSG = None, f"RULE MODEL NOT LOADED ({e!r}) -> plays as N14"


def rule_spots(bot, s, have, k):
    """top-k blast centres by forecast dinos inside (rule model, 3 moves ahead), excluding spots in `have`"""
    fc = SF.rule_forecast(RULE, s, bot.mt.last)
    mass = Counter()
    for dist, _ in fc:
        for t, p in dist.items():
            mass[t] += p
    cents = {(t[0] + a, t[1] + b) for t in mass if mass[t] > 0.01 for a, b in bot.foot}
    scored = []
    for c in cents:
        if c in have or not bot.in_window(*c):
            continue
        m = sum(mass.get((c[0] + a, c[1] + b), 0.0) for a, b in bot.foot)
        if m >= RULE_MIN:
            scored.append((m, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:k]]


class Bot(N14.Bot):
    def __init__(self):
        super().__init__()
        self.mt = SF.MoveTracker()
        self.rspots = set()
        self.r14b = Counter()
        self.p14b = []
        print(f"[N14b] loaded n14b_rule_spots | {RULE_MSG} | up to {RULE_K} rule spots/tick (min {RULE_MIN} dinos)")

    def choose_actions(self, s):
        self.mt.update(s)
        self.rspots = set()
        return super().choose_actions(s)

    def candidates(self, s):
        lst = super().candidates(s)                         # N7 list (sets geometry, follow-ups, prior)
        if RULE is None or not lst or self.cool > 0:
            return lst
        if 1000 * (time.perf_counter() - self.tick_t0) > RULE_MAX_MS:
            self.r14b["skipped"] += 1
            return lst
        t0 = time.perf_counter()
        extra = rule_spots(self, s, set(lst), RULE_K)
        self.r14b["ms"] += 1000 * (time.perf_counter() - t0)
        self.r14b["ticks"] += 1
        if not extra:
            return lst
        for p in extra:
            self.prior[p] = self.cage_prior(p)
        self.rspots = set(extra)
        self.r14b["added"] += len(extra)
        k = 0
        while k < len(lst) and lst[k] in self.follow:
            k += 1
        return lst[:k] + extra + lst[k:]

    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, N12.LaunchMeteorAction) and (a.target.x, a.target.y) in self.rspots:
                c = (a.target.x, a.target.y)
                self.r14b["chosen"] += 1
                self.p14b.append((s["currentTick"] + s["constants"]["meteorDelay"], N4.blast(c, s["constants"]["meteorRadius"])))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles in self.p14b:
                if L == T - 1:
                    self.r14b["kills"] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                              and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.p14b = [x for x in self.p14b if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        r = self.r14b
        print(f"[N14b] rule spots | ticks {r['ticks']} (skipped {r['skipped']}) | added {r['added']} | "
              f"chosen by XGBoost {r['chosen']} -> {r['kills']} kills ({r['kills'] / max(r['chosen'], 1):.2f}/shot) | "
              f"forecast {r['ms'] / max(r['ticks'], 1):.1f} ms/tick")
