"""
N1 straggler hunter = cage_cnn4 (Cage B, our best) + 3 changes aimed at clearing waves faster.

Why: in 1,054 local games, score follows wave speed (best 10% of games: 55 ticks per wave, 17.4k;
worst 10%: 240 ticks, 4k). The last 20% of a wave takes 35% of its time, and the last survivor is a
Stegosaurus 57% of the time. A new wave arrives 2 ticks after the board is empty, so the last dino is
worth far more than its own (often old, low) value.

Changes (everything else is exactly cage_cnn4):
  1. CLEAR_BONUS 4 -> 10: emptying the board is worth about a whole new wave (~10 young dinos).
  2. Stragglers: when STRAGGLER_N or fewer dinos are not already doomed, they count as newborns
     (full value) no matter their age, so the bot chases them instead of ignoring old ones.
  3. Running-line spots: for every Stegosaurus (they rarely stop and run straight) and for every straggler,
     add meteor spots 2-4 tiles ahead along its last move, and score them FIRST with CNN4
     (Cage B only aims where a dino stands now, so it never tried leading a runner).

Use: in bot/bot.py ->  from n1_straggler_hunter import Bot
"""
from game_message import *

import cage_cnn4 as CB

CA, C, FC = CB.CA, CB.C, CB.FC

# ---- knobs ---------------------------------------------------------------
CLEAR_BONUS = 10.0       # was 4.0 (in "newborn kills")
STRAGGLER_N = 3          # this many or fewer live dinos -> treat them as full value
LEAD_STEPS = (2, 3, 4)   # spots this many tiles ahead along a runner's last move
LEAD_SPECIES = {"Stegosaurus"}
# --------------------------------------------------------------------------

FC.CLEAR_BONUS = CLEAR_BONUS
FC.H.BOARD_CLEAR_W = CLEAR_BONUS      # heuristic fallback uses the same bonus
C.BOT_NAME = "n1_straggler_hunter"


class Bot(CB.Bot):
    def __init__(self):
        super().__init__()
        self.names = {}
        self.n1 = dict(straggler_ticks=0, lead_spots=0, lead_fired=0)

    def choose_actions(self, s):
        self.names = {d["id"]: d["name"] for d in s["dinosaurs"]}
        return super().choose_actions(s)

    def live_ids(self):
        """dinos not already standing in a meteor that lands before ours"""
        land = self.t + self.D
        return {d["id"] for d in self.dinos
                if not any(d["pos"] in tiles and L < land for L, _, tiles in self.falling)}

    # 2. stragglers count as newborns
    def value_grid(self):
        live = self.live_ids()
        if 0 < len(live) <= STRAGGLER_N:
            self.n1["straggler_ticks"] += 1
            saved = {d["id"]: d["age"] for d in self.dinos if d["id"] in live}
            for d in self.dinos:
                if d["id"] in live:
                    d["age"] = -self.D                     # value(age + D) = value(0) = full
            g = super().value_grid()
            for d in self.dinos:
                if d["id"] in saved:
                    d["age"] = saved[d["id"]]
            return g
        return super().value_grid()

    # 3. running-line spots first
    def candidates(self, s):
        ranked = super().candidates(s)                     # sets geometry (self.dinos, self.falling, ...)
        live = self.live_ids()
        straggle = 0 < len(live) <= STRAGGLER_N
        lead = []
        for d in self.dinos:
            if d["id"] not in live:
                continue
            if not (self.names.get(d["id"]) in LEAD_SPECIES or straggle):
                continue
            h = list(self.hist4.get(d["id"], ()))
            if len(h) < 2:
                continue
            dx, dy = h[-1][0] - h[-2][0], h[-1][1] - h[-2][1]
            if (dx, dy) == (0, 0):
                continue
            x, y = d["pos"]
            for k in LEAD_STEPS:
                p = (x + k * dx, y + k * dy)
                if self.in_window(*p) and p not in lead:
                    lead.append(p)
        self.lead = set(lead)
        self.n1["lead_spots"] += len(lead)
        rest = [p for p in ranked if p not in self.lead]
        return lead + rest

    def plan_meteor(self, s):
        t = super().plan_meteor(s)
        if t is not None and t in getattr(self, "lead", ()):
            self.n1["lead_fired"] += 1
        return t

    def finish(self, s):
        super().finish(s)
        print(f"[N1] straggler ticks {self.n1['straggler_ticks']} | running-line spots tried {self.n1['lead_spots']} | "
              f"meteors fired at a running-line spot {self.n1['lead_fired']}")
