"""
N14d rule by count = N14c (final = (1-w) x XGBoost + w x rule model) but the rule model's share w grows as
the board empties - stragglers are about predicting ONE dino's species movement, where the spec rules help most.

  dinos alive   XGBoost   rule model
  6+            90%       10%
  3-5           80%       20%
  2             70%       30%
  1             55%       45%

Same candidates as N14, no retraining. Needs bot/models/behavior/weights.json + bot/models/scorer/.
Logs: all N14c / N14 / N12 logs (per-mode kills/shot shows whether straggler improved) + [N14d] shots and
kills per weight level.
Use: ./scripts/ship.sh n14d_rule_by_count
"""
from collections import Counter

import n14c_rule_blend as N14C

N14, N12, N4, C = N14C.N14, N14C.N12, N14C.N4, N14C.C
print = N14C.print

# ---- knobs ---------------------------------------------------------------
W_BY_COUNT = [(1, 0.45), (2, 0.30), (5, 0.20), (999, 0.10)]     # (max dinos alive, rule-model share)
# --------------------------------------------------------------------------

C.BOT_NAME = "n14d_rule_by_count"


def rule_share(n):
    for hi, w in W_BY_COUNT:
        if n <= hi:
            return w
    return W_BY_COUNT[-1][1]


class Bot(N14C.Bot):
    def __init__(self):
        super().__init__()
        self.w_now = 0.0
        self.r14d = Counter()
        self.p14d = []
        print(f"[N14d] loaded n14d_rule_by_count | rule share by dinos alive: "
              + ", ".join(f"<={hi}: {w:.0%}" for hi, w in W_BY_COUNT))

    def choose_actions(self, s):
        self.w_now = rule_share(len(s["dinosaurs"]))
        N14C.BLEND_W = self.w_now                       # read by the blended plan this tick
        return super().choose_actions(s)

    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, N12.LaunchMeteorAction):
                k = f"{self.w_now:.2f}"
                self.r14d["shots_" + k] += 1
                c = (a.target.x, a.target.y)
                self.p14d.append((s["currentTick"] + s["constants"]["meteorDelay"],
                                  N4.blast(c, s["constants"]["meteorRadius"]), k))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles, k in self.p14d:
                if L == T - 1:
                    self.r14d["kills_" + k] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                                   and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.p14d = [x for x in self.p14d if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        r = self.r14d
        parts = []
        for hi, w in W_BY_COUNT:
            k = f"{w:.2f}"
            n = r["shots_" + k]
            parts.append(f"<={hi} dinos (rule {w:.0%}): {n} shots, {r['kills_' + k]} kills "
                         f"({r['kills_' + k] / max(n, 1):.2f}/shot)")
        print("[N14d] " + " | ".join(parts))
