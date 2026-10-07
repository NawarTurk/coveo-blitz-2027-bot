"""
Shared meteor-candidate features. USED BY BOTH build_meteor_table.py (training) AND bot_znawar1.py (playing),
so the learned weights always see exactly the same numbers.

  ctx = FeatureCtx(state_dict)          # once per tick
  f   = features(ctx, (x, y))           # per candidate -> dict with HEUR features
  vg  = value_grid(ctx, H, W)           # points a kill is worth on each tile (for the CNN4 score)
"""
import numpy as np

PRE_TRAP = [(0, 0.79), (3, 0.47), (7, 0.26), (14, 0.10), (999, 0.036)]
HEUR = ["n_dinos", "n_left", "n_cover", "n_reach", "v_cover", "v_reach", "trap_prior", "min_exits", "mean_exits",
        "blocked_ring", "chain_n", "scroll_n", "scroll_v", "row_from_back", "dist_block", "n_r3", "n_steg",
        "n_raptor", "n_tri", "n_trex", "trex_prey", "age_min", "age_mean", "meteors_falling"]
BASE = ["cnn_score", "cnn_clear"]


def val(age):
    return 160.0 / (1 + age / 30.0)


def p_trap(ex):
    for hi, p in PRE_TRAP:
        if ex <= hi:
            return p
    return PRE_TRAP[-1][1]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class FeatureCtx:
    def __init__(self, s):
        c = s["constants"]
        self.D, self.R, self.SH = c["meteorDelay"], c["meteorRadius"], c["mapShiftInterval"]
        self.t = s["currentTick"]
        m = s["map"]
        self.ox, self.oy, self.h, self.w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        R, D = self.R, self.D
        self.foot = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]
        self.ring = [(a, b) for a in range(-R - 2, R + 3) for b in range(-R - 2, R + 3) if abs(a) + abs(b) <= R + 2]
        self.near3 = [(a, b) for a in range(-3, 4) for b in range(-3 + abs(a), 4 - abs(a))]
        self.blocked = set()
        for i, row in enumerate(m["tiles"]):
            for j, tl in enumerate(row):
                if tl["isImpassable"] or tl["isMountain"] or tl["hasLava"]:
                    self.blocked.add((j + self.ox, i + self.oy))
        # distance (Manhattan) from every window tile to the nearest blocked tile (40 if none)
        if self.blocked:
            Bk = np.array(list(self.blocked))
            ii, jj = np.mgrid[0:self.h, 0:self.w]
            d = np.abs((jj + self.ox)[..., None] - Bk[:, 0]) + np.abs((ii + self.oy)[..., None] - Bk[:, 1])
            self.dblock = np.minimum(d.min(-1), 40)
        else:
            self.dblock = np.full((self.h, self.w), 40)
        self.land = self.t + D
        self.falling = [(self.t + me["turnsUntilImpact"] - 1, {(p["x"], p["y"]) for p in me["impactedTiles"]})
                        for me in s["meteors"]]
        self.n_meteors = len(s["meteors"])
        self.dinos = [(d["id"], (d["position"]["x"], d["position"]["y"]), d["name"], d["age"]) for d in s["dinosaurs"]]
        self.vals = {d[0]: val(d[3] + D) for d in self.dinos}
        self.doomed = {i for i, p, _, _ in self.dinos if any(p in T and L < self.land for L, T in self.falling)}
        self.n_left = len(self.dinos) - len(self.doomed)
        self.shifts = sum(1 for k in range(self.t + 1, self.land + 1) if k % self.SH == 0)
        self.trex = [p for _, p, n, _ in self.dinos if n == "Tyrannosaurus"]

    def inwin(self, q):
        return 0 <= q[0] - self.ox < self.w and 0 <= q[1] - self.oy < self.h


def features(ctx, c):
    R, D = ctx.R, ctx.D
    cover, reach, r3 = [], [], 0
    for d in ctx.dinos:
        md = manh(d[1], c)
        if md <= R:
            cover.append(d)
        if md <= R + D:
            reach.append(d)
        if md <= 3:
            r3 += 1
    exits = []
    for _, p, _, _ in cover:
        ex = 0
        for a, b in ctx.near3:
            q = (p[0] + a, p[1] + b)
            if q not in ctx.blocked and ctx.inwin(q) and manh(q, c) > R:
                ex += 1
        exits.append(ex)
    nb = 0
    for a, b in ctx.ring:
        q = (c[0] + a, c[1] + b)
        if q in ctx.blocked or not ctx.inwin(q):
            nb += 1
    sc = [d for d in reach if d[1][1] - ctx.oy < ctx.shifts]
    i, j = c[1] - ctx.oy, c[0] - ctx.ox
    dist_block = int(ctx.dblock[i, j]) if (0 <= i < ctx.h and 0 <= j < ctx.w) else 40
    ages = [d[3] for d in reach]
    return dict(
        n_dinos=len(ctx.dinos), n_left=ctx.n_left, n_cover=len(cover), n_reach=len(reach),
        v_cover=sum(ctx.vals[d[0]] for d in cover), v_reach=sum(ctx.vals[d[0]] for d in reach),
        trap_prior=sum(ctx.vals[d[0]] * p_trap(ex) for d, ex in zip(cover, exits)),
        min_exits=min(exits) if exits else 30, mean_exits=sum(exits) / len(exits) if exits else 30,
        blocked_ring=round(nb / len(ctx.ring), 3),
        chain_n=sum(1 for d in reach if d[0] in ctx.doomed),
        scroll_n=len(sc), scroll_v=sum(ctx.vals[d[0]] for d in sc),
        row_from_back=c[1] - ctx.oy, dist_block=dist_block, n_r3=r3,
        n_steg=sum(1 for d in reach if d[2] == "Stegosaurus"),
        n_raptor=sum(1 for d in reach if d[2] == "Velociraptor"),
        n_tri=sum(1 for d in reach if d[2] == "Triceratops"),
        n_trex=sum(1 for d in reach if d[2] == "Tyrannosaurus"),
        trex_prey=sum(1 for d in cover if d[2] != "Tyrannosaurus" and any(manh(d[1], q) <= 1 for q in ctx.trex)),
        age_min=min(ages, default=0), age_mean=sum(ages) / len(ages) if ages else 0,
        meteors_falling=ctx.n_meteors)


def value_grid(ctx, H, W):
    """points of a kill on each tile = mean value of the dinos that can reach it (as Z1)"""
    if not ctx.dinos:
        return np.zeros((H, W), np.float32)
    P = np.array([d[1] for d in ctx.dinos])
    V = np.array([ctx.vals[d[0]] for d in ctx.dinos], np.float32)
    ii, jj = np.mgrid[0:H, 0:W]
    dist = np.abs((jj + ctx.ox)[..., None] - P[:, 0]) + np.abs((ii + ctx.oy)[..., None] - P[:, 1])
    near = dist <= ctx.D + 1
    cnt = near.sum(-1)
    return np.where(cnt > 0, (near * V).sum(-1) / np.maximum(cnt, 1), 0).astype(np.float32)


def cnn_feats(E, V, n_left):
    """the two CNN4 features, from E = sum P(dino) over the blast and V = sum P x value"""
    return dict(cnn_score=V * (1 + 0.5 * max(E - 1, 0)),
                cnn_clear=(min(1.0, E / n_left) ** n_left) if n_left > 0 else 0.0)


class LinearModel:
    """ridge model saved by fit_meteor_models.py: pred = intercept + sum coef_std * (x - mean) / std"""
    def __init__(self, d):
        self.features = d["features"]
        self.mean = np.array(d["mean"]); self.std = np.array(d["std"])
        self.w = np.array(d["coef_std"]) / self.std
        self.b = d["intercept_std"] - float((self.w * self.mean).sum())

    def predict(self, rows):
        X = np.array([[r[f] for f in self.features] for r in rows], float)
        return X @ self.w + self.b
