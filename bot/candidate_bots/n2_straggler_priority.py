"""
N2 straggler priority = N1 straggler hunter + stragglers become (almost) infinitely valuable.

When STRAGGLER_N or fewer dinos are left alive (not already under a falling meteor), each of them is worth
STRAGGLER_VALUE (1000x a normal kill). The meteor score is
    sum over blast tiles of  P(dino on tile at impact) x value of the dinos that can reach it
so with a huge value the score becomes, in practice, "probability of killing a straggler": the bot fires
at the spot most likely to finish the wave, ignoring everything else (trap table, ages, other dinos).
Why not truly infinite: then every spot with any chance would tie and the bot would pick at random.

Everything else = N1 (clear bonus 10, running-line spots for Stegosaurus and stragglers, Cage B core).
Use: in bot/bot.py ->  from n2_straggler_priority import Bot
"""
from game_message import *

import numpy as np

import n1_straggler_hunter as N1

CB, CA, C, FC = N1.CB, N1.CA, N1.C, N1.FC

# ---- knobs ---------------------------------------------------------------
STRAGGLER_N = N1.STRAGGLER_N      # 3: this many or fewer live dinos -> straggler mode
STRAGGLER_VALUE = 1000.0          # value of one straggler kill (a newborn kill = 1.0)
# --------------------------------------------------------------------------

C.BOT_NAME = "n2_straggler_priority"


class Bot(N1.Bot):
    def __init__(self):
        super().__init__()
        self.n2 = dict(priority_ticks=0)

    def value_grid(self):
        live = self.live_ids()
        if not (0 < len(live) <= STRAGGLER_N):
            return super().value_grid()
        self.n2["priority_ticks"] += 1
        self.n1["straggler_ticks"] += 1
        pos = np.array([d["pos"] for d in self.dinos])
        val = np.array([STRAGGLER_VALUE if d["id"] in live else C.age_value(d["age"] + self.D) for d in self.dinos],
                       np.float32)
        ii, jj = np.mgrid[0:self.H, 0:self.W]
        X, Y = jj + self.ox, ii + self.oy
        dist = np.abs(X[..., None] - pos[:, 0]) + np.abs(Y[..., None] - pos[:, 1])
        near = dist <= self.D + 1
        # a tile is worth the most valuable dino that can reach it (a straggler must not be averaged away)
        g = np.where(near, val, 0).max(-1)
        return g.astype(np.float32)

    def finish(self, s):
        super().finish(s)
        print(f"[N2] ticks in straggler-priority mode: {self.n2['priority_ticks']}")
