"""
BEHAVIOUR MODEL: predicts each dino's next move from the confirmed spec rules (results/dino_behavior_model.md).

For every dino it scores each possible move (stay / up / down / left / right that is not blocked) with rule
features, and turns the scores into probabilities (softmax). The weight of each rule is learned per species
from game logs (training/fit_behavior.py), so a rule only matters for the species where the data says it does.

Rules used (H-numbers from spec_hypothesis.md; the ones that failed are left out):
  escape a falling blast (H4, H15)        avoid stepping into a blast          raptor stops just outside (H8)
  avoid edges, most of all the top (H3, H17)                                   stegosaurus target zone (H2)
  stay with the pack (H9, H10)            T-Rex spreads out (H13)              high ground (H11)
  lava-adjacent tiles (H16)               never on blocked / lava / shared tiles (H18, H19)
  next to a T-Rex = danger (H14)          repeat last move, stay habit, down bias (H1, H6)

Use:
  M = BehaviorModel.load("models/behavior/weights.json")
  board = Board(state)                     # once per tick
  probs = M.move_probs(board, dino_id, last_moves)   -> {tile: probability}
  grid  = M.forecast(board, last_moves, steps=4)     -> {dino_id: {tile: P(dino on tile after 4 moves)}}
"""
import json
import math

NB = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
TARGET = (9.5, 12.0)                   # stegosaurus target zone: (column, row) inside the window (H2)
R_BLAST = 2

FEATS = ["stay", "repeat", "escape", "into_blast", "stop_outside", "top_edge", "other_edge", "target_pull",
         "pack_pull", "spread", "elev_up", "lava_adj", "occupied", "next_to_rex", "down", "sideways"]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Board:
    """one tick of the game, from a game-state dict"""

    def __init__(self, s):
        m = s["map"]
        self.t = s["currentTick"]
        self.ox, self.oy, self.h, self.w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        self.elev, self.block, self.lava = {}, set(), set()
        for i, row in enumerate(m["tiles"]):
            for j, tl in enumerate(row):
                p = (j + self.ox, i + self.oy)
                self.elev[p] = tl.get("elevation", 0)
                if tl["isImpassable"] or tl["isMountain"]:
                    self.block.add(p)
                if tl["hasLava"]:
                    self.lava.add(p)
        self.d = {d["id"]: (SP.get(d["name"], "?"), (d["position"]["x"], d["position"]["y"]))
                  for d in s["dinosaurs"]}
        self.met = [((me["target"]["x"], me["target"]["y"]), me["turnsUntilImpact"]) for me in s["meteors"]]
        self.lava_adj = {(x + a, y + b) for (x, y) in self.lava for a, b in NB + [(0, 0)]}

    def inside(self, p):
        return self.ox <= p[0] < self.ox + self.w and self.oy <= p[1] < self.oy + self.h

    def free(self, p):
        return self.inside(p) and p not in self.block and p not in self.lava

    def options(self, p):
        return [p] + [(p[0] + a, p[1] + b) for a, b in NB if self.free((p[0] + a, p[1] + b))]


def features(b, i, opts, last):
    """rule features for each option of dino i (list of lists, FEATS order)"""
    sp, p = b.d[i]
    blasts = [c for c, _ in b.met if manh(p, c) <= R_BLAST]
    near_c = min((c for c, _ in b.met), key=lambda c: manh(p, c), default=None)
    just_out = near_c is not None and manh(p, near_c) == R_BLAST + 1
    same = [q for j, (s2, q) in b.d.items() if j != i and s2 == sp]
    others = [q for j, (s2, q) in b.d.items() if j != i]
    occ = set(others)
    rex = [q for j, (s2, q) in b.d.items() if j != i and s2 == "R"]
    lm = last.get(i)
    e0 = b.elev.get(p, 0)

    def tdist(q):
        return abs(q[0] - b.ox - TARGET[0]) + abs(q[1] - b.oy - TARGET[1])

    def nd(q, pts):
        return min(manh(q, x) for x in pts) if pts else 0

    out = []
    for o in opts:
        mv = (o[0] - p[0], o[1] - p[1])
        row = o[1] - b.oy
        col = o[0] - b.ox
        esc = sum(manh(o, c) - manh(p, c) for c in blasts)
        into = sum(1 for c, _ in b.met if manh(o, c) <= R_BLAST)
        out.append([
            1.0 if o == p else 0.0,
            1.0 if lm is not None and mv == lm and mv != (0, 0) else 0.0,
            float(esc),
            float(into),
            1.0 if (just_out and o == p) else 0.0,
            float(max(0, 3 - row)),
            float(max(0, 2 - (b.h - 1 - row)) + max(0, 2 - col) + max(0, 2 - (b.w - 1 - col))),
            float(tdist(p) - tdist(o)),
            float(nd(p, same) - nd(o, same)) if same else 0.0,
            float(nd(o, others) - nd(p, others)) if others else 0.0,
            float(b.elev.get(o, e0) - e0),
            1.0 if o in b.lava_adj else 0.0,
            1.0 if (o in occ and o != p) else 0.0,
            1.0 if any(manh(o, r) == 1 for r in rex) else 0.0,
            float(mv[1]),
            1.0 if mv[0] != 0 else 0.0,
        ])
    return out


class BehaviorModel:
    def __init__(self, weights):
        self.w = weights                       # species -> list of len(FEATS)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["weights"])

    def move_probs(self, b, i, last):
        sp, p = b.d[i]
        opts = b.options(p)
        w = self.w.get(sp)
        if w is None:
            return {o: 1 / len(opts) for o in opts}
        sc = [sum(a * c for a, c in zip(w, f)) for f in features(b, i, opts, last)]
        m = max(sc)
        ex = [math.exp(x - m) for x in sc]
        z = sum(ex)
        return {o: e / z for o, e in zip(opts, ex)}

    def forecast(self, b, last, steps=4, only=None):
        """P(dino on tile after `steps` moves) per dino; other dinos and meteors held at their current state"""
        out = {}
        for i in (only or b.d):
            sp, p = b.d[i]
            dist = {p: 1.0}
            for _ in range(steps):
                nxt = {}
                for q, pq in dist.items():
                    if pq < 1e-4:
                        continue
                    b.d[i] = (sp, q)
                    for o, po in self.move_probs(b, i, last).items():
                        nxt[o] = nxt.get(o, 0.0) + pq * po
                dist = nxt
            b.d[i] = (sp, p)
            out[i] = dist
        return out
