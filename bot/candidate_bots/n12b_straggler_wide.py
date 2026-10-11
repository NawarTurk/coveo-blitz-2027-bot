"""
N12b straggler wide = N12 modes with every N12 strategy OFF (= N7 targeting + N12 mode logs)
+ ONE change: a WIDER search in straggler mode (1-2 dinos on the board).

Why: N12 (3 server games, 14.3k vs N7 ~17.6k) showed that forcing hand-made spots on CNN4 loses in every mode.
With 1-2 dinos N7 only gives CNN4 ~26-35 spots although it has time for ~60.
N12b fills that spare time with MORE options, nothing is forced:
  extra spots = every blast centre that covers a tile the straggler can reach in the next 3 moves
  (up to ~60 per dino), ranked by the same trap-table prior and merged with N7's spots.
  N6's priority follow-ups stay first; CNN4 scores as many as fit in the time budget and picks the best.

Logs: all N12 logs; straggler shots by kind: "follow" (N6 follow-up), "normal" (N7 spot), "wide" (new spot).
Mode lines: "own spots" = the wide spots chosen and what they killed.
Use: ./scripts/ship.sh n12b_straggler_wide
"""
import n12_modes as N12

for k in N12.ON:
    N12.ON[k] = False                     # all N12 strategies off: plays like N7
N4, CB, C, manh = N12.N4, N12.CB, N12.C, N12.manh
print = N12.print

# ---- knobs ---------------------------------------------------------------
WIDE = True              # the one change (False = N7 + mode logs)
WIDE_MAX = 60            # max spots in the straggler list (CNN4 scores as many as time allows)
# --------------------------------------------------------------------------

C.BOT_NAME = "n12b_straggler_wide"


class Bot(N12.Bot):
    OWN_KINDS = ("wide",)

    def __init__(self):
        super().__init__()
        self.wide = dict(ticks=0, added=0)
        print(f"[N12b] loaded n12b_straggler_wide | N12 strategies off | WIDE {WIDE} | max {WIDE_MAX} spots in straggler mode")

    def candidates(self, s):
        lst = super().candidates(s)                         # N7 list (all N12 strategies are off)
        if not WIDE or self.mode != "straggler" or not lst or self.cool > 0:
            return lst
        have = set(lst)
        extra = set()
        for d in self.dinos:
            for q in d["reach"] + [d["pos"]]:
                for a, b in N12.FOOT:
                    p = (q[0] + a, q[1] + b)
                    if p not in have and self.in_window(*p):
                        extra.add(p)
        for p in extra:
            self.prior[p] = self.cage_prior(p)
        fol = [p for p in lst if p in self.follow]
        rest = sorted([p for p in lst if p not in self.follow] + list(extra), key=lambda p: -self.prior[p])
        out = (fol + rest)[:max(WIDE_MAX, len(fol))]
        self.kinds = {p: ("follow" if p in self.follow else "wide" if p in extra else "normal") for p in out}
        self.wide["ticks"] += 1
        self.wide["added"] += sum(1 for p in out if p in extra)
        return out

    def finish(self, s):
        super().finish(s)
        w = self.wide
        print(f"[N12b] straggler ticks with wide search {w['ticks']} | wide spots added {w['added']} "
              f"({w['added'] / max(w['ticks'], 1):.1f}/tick)")
