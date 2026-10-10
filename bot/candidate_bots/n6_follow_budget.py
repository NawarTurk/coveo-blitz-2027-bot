"""
N6 follow budget = N4 survival follow (SURVIVE_MIN 0.5, everything else identical) + ONE change:
follow-up spots can no longer take over the CNN4 budget.

N4:  ALL follow-up spots first (in no particular order), then the normal cage spots.
     On a slow tick CNN4 may only score ~16 spots, so follow-up guesses can push strong cage spots out.
N6:  follow-up spots sorted by the existing trap-table prior (best first);
     only max(2, budget // 4) of them go first, the rest go AFTER the normal cage spots.
     budget = the number of spots CNN4 can score this tick (same estimate cage_cnn4 uses).

Logs: [N6] summary (follow-ups put first / pushed back, average budget) + all N4 logs + full trace.
Use: ./scripts/ship.sh n6_follow_budget
"""
import time

import n4_survival_follow as N4

CB, C = N4.CB, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
FOLLOW_SHARE = 4         # at most budget // FOLLOW_SHARE follow-ups go first (25%)
FOLLOW_MIN = 2           # ... but always at least this many (when there are any)
# --------------------------------------------------------------------------

C.BOT_NAME = "n6_follow_budget"


class Bot(N4.Bot):
    def __init__(self):
        super().__init__()
        self.n6 = dict(ticks=0, first=0, back=0, budget=0)
        print(f"[N6] loaded n6_follow_budget | SURVIVE_MIN {N4.SURVIVE_MIN} | follow-ups first: max({FOLLOW_MIN}, budget/{FOLLOW_SHARE})")

    def budget(self):
        """spots CNN4 can score this tick = same estimate as cage_cnn4.plan_meteor"""
        left = C.TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 3
        unit = min(self.unit4, CB.UNIT_CAP)
        return max(0, min(CB.MAX4, int(left / unit) if unit > 0 else CB.MAX4))

    def candidates(self, s):
        ranked = super(N4.Bot, self).candidates(s)        # plain cage_cnn4 order (skips N4's reordering)
        self.plain_first = ranked[0] if ranked else None
        fs = self.follow_spots()
        self.follow = set(fs)
        if not fs:
            return ranked
        for p in fs:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        pos = {p: i for i, p in enumerate(ranked)}
        self.st["spots"] += len(fs)
        self.st["cut"] += sum(1 for p in fs if pos.get(p, 10 ** 9) >= CB.MAX4)
        fs.sort(key=lambda p: -self.prior[p])                # best follow-ups first
        b = self.budget()
        k = min(len(fs), max(FOLLOW_MIN, b // FOLLOW_SHARE))
        first, back = fs[:k], fs[k:]
        n = self.n6
        n["ticks"] += 1; n["first"] += len(first); n["back"] += len(back); n["budget"] += b
        return first + [p for p in ranked if p not in self.follow] + back

    def finish(self, s):
        super().finish(s)
        n = self.n6
        t = max(n["ticks"], 1)
        print(f"[N6] summary | ticks with follow-ups {n['ticks']} | put first {n['first']} ({n['first'] / t:.1f}/tick) "
              f"| pushed back {n['back']} ({n['back'] / t:.1f}/tick) | avg CNN4 budget {n['budget'] / t:.1f} spots")
