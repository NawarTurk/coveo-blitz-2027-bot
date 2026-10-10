"""
N7 safe timing = N6 follow budget (= N4 + follow-up cap; targeting UNCHANGED, SURVIVE_MIN 0.5) + cautious CNN4 recovery.

Why: in a server game the bot took 130-217 ms for 6 ticks in a row and missed them all. The UNIT_CAP fix in
cage_cnn4 makes the bot retry a full CNN4 batch every tick, even while the server is really slow.

The change (per tick, by measured tick time):
  level 3 = normal (up to 60 spots, batch 16)
  a tick slower than SLOW_MS     -> COOLDOWN: no CNN4 for COOLDOWN ticks (best trap-table spot, like a fallback)
  after the cooldown             -> level 0 = PROBE: CNN4 on only 4 spots
  a tick where CNN4 ran and took less than SAFE_MS -> one level up: 4 -> 8 -> 16 -> normal
  a slow tick at any level       -> back to cooldown

Logs: [N7] line on every slow tick / level change, [N7] summary at the end, + all N4 logs and the full trace.
Use: ./scripts/ship.sh n7_safe_timing
"""
import time

import n4_survival_follow as N4
import n6_follow_budget as N6

CB, C = N4.CB, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
SLOW_MS = 110            # a tick slower than this -> cooldown (server misses a tick at ~150+)
SAFE_MS = 70             # a tick faster than this -> one level up
COOLDOWN = 3             # ticks without CNN4 after a slow tick
LEVELS = [(4, 4), (8, 8), (16, 16), (CB.MAX4, CB.CHUNK)]     # (max spots, batch size)
# --------------------------------------------------------------------------

C.BOT_NAME = "n7_safe_timing"


class Bot(N6.Bot):
    def __init__(self):
        super().__init__()
        self.level = len(LEVELS) - 1
        self.cool = 0
        self.n7 = dict(slow=0, cool_ticks=0, probes=0, ups=0, worst=0.0)
        self.at_level = [0] * len(LEVELS)
        print(f"[N7] loaded | SLOW_MS {SLOW_MS} SAFE_MS {SAFE_MS} COOLDOWN {COOLDOWN} levels {LEVELS}")

    def choose_actions(self, s):
        T = s["currentTick"]
        if self.cool > 0:
            self.cool -= 1
            self.n7["cool_ticks"] += 1
            CB.MAX4 = 0                                   # n = 0 -> best trap-table spot, no CNN4
        else:
            CB.MAX4, CB.CHUNK = LEVELS[self.level]
            self.at_level[self.level] += 1
            if self.level == 0:
                self.n7["probes"] += 1
        self.ran_cnn = False
        actions = super().choose_actions(s)
        ms = 1000 * (time.perf_counter() - self.tick_t0)
        self.n7["worst"] = max(self.n7["worst"], ms)
        cooling = CB.MAX4 == 0
        CB.MAX4, CB.CHUNK = LEVELS[-1]                    # restore defaults
        if ms > SLOW_MS:
            self.n7["slow"] += 1
            print(f"[N7] t={T} slow tick {ms:.0f} ms at level {self.level} -> cooldown {COOLDOWN} ticks, then probe 4 spots")
            self.cool, self.level = COOLDOWN, 0
        elif not cooling and self.ran_cnn and ms < SAFE_MS and self.level < len(LEVELS) - 1:
            self.level += 1
            self.n7["ups"] += 1
            print(f"[N7] t={T} tick {ms:.0f} ms -> level {self.level} (max spots {LEVELS[self.level][0]})")
        return actions

    def plan_meteor(self, s):
        fb = self.fallbacks
        t = super().plan_meteor(s)
        self.ran_cnn = self.fallbacks == fb and self.n4 > 0       # CNN4 really scored spots this tick
        return t

    def finish(self, s):
        super().finish(s)
        n = self.n7
        print(f"[N7] summary | slow ticks (>{SLOW_MS} ms) {n['slow']} | cooldown ticks {n['cool_ticks']} | probes {n['probes']} "
              f"| level-ups {n['ups']} | worst tick {n['worst']:.0f} ms | ticks per level (4/8/16/normal) {self.at_level}")
