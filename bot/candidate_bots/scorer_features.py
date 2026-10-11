"""
Features of ONE meteor candidate for the learned shot scorer (XGBoost).
Used by training/build_scorer_dataset.py AND later by the bot (N14), so training and play see identical numbers.

Call after the bot's per-tick setup (cage_planner.setup) has filled: t, D, R, ox, oy, h, w, free, falling, dinos (+reach).
P = CNN4's P(dino) per window tile at impact for THIS candidate (20x20 array), vg = value_grid(), mask = blast tiles.
"""
import numpy as np

MODES = ["straggler", "rex", "pack", "crowd", "normal", "empty"]
SPI = {"Stegosaurus": 0, "Velociraptor": 1, "Triceratops": 2, "Tyrannosaurus": 3}

FEATURES = [
    # CNN4 forecast for this blast
    "cnn_E", "cnn_v", "cnn_pmax", "cnn_pc", "cnn_clear", "cnn_score",
    # trap-table prior (N7's 30% part)
    "prior",
    # dinos now
    "n_dinos", "n_cov", "n_cov1", "cov_S", "cov_V", "cov_T", "cov_R", "val_cov", "min_age_cov",
    "min_exits", "mean_exits", "min_dist", "frac_reach",
    # meteors already falling
    "n_falling", "n_fall_near", "overlap", "land_gap",
    # where the blast is
    "is_follow", "blocked_in_blast", "edge_dist", "row",
    # board mode
    "mode", "trapped", "rex_risk", "pack", "tick_frac",
]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def features(bot, c, P, vg, mask, names, mode, flags, follow, clear_bonus):
    R, D = bot.R, bot.D
    land = bot.t + D
    Pm = P[mask]
    E = float(Pm.sum())
    v = float((Pm * vg[mask]).sum()) * (1 + 0.5 * max(E - 1, 0))
    n_left = sum(1 for d in bot.dinos if not any(d["pos"] in tiles and L < land for L, _, tiles in bot.falling))
    clear = clear_bonus * min(1.0, E / n_left) ** n_left if n_left > 0 else 0.0
    i, j = c[1] - bot.oy, c[0] - bot.ox
    pc = float(P[i, j]) if 0 <= i < bot.h and 0 <= j < bot.w else 0.0

    blast = {(c[0] + a, c[1] + b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R}
    deadly = set()
    for L, _, tiles in bot.falling:
        if land - 1 <= L <= land + 1:
            deadly |= tiles
    cov, cov1, sp, exits = [], 0, [0, 0, 0, 0], []
    for d in bot.dinos:
        dist = manh(d["pos"], c)
        if dist <= R + 1:
            cov1 += 1
        if dist <= R:
            cov.append(d)
            k = SPI.get(names.get(d["id"]), -1)
            if k >= 0:
                sp[k] += 1
            exits.append(sum(1 for q in d["reach"] if q not in blast and q not in deadly))
    near = min(bot.dinos, key=lambda d: manh(d["pos"], c)) if bot.dinos else None
    min_dist = manh(near["pos"], c) if near else 99
    frac = (sum(1 for q in near["reach"] if q in blast) / max(len(near["reach"]), 1)) if near else 0.0
    fall_near = sum(1 for _, cen, _ in bot.falling if near and manh(near["pos"], cen) <= R + 1)
    overlap = sum(1 for _, _, tiles in bot.falling for q in blast if q in tiles)
    gaps = [abs(L - land) for L, _, tiles in bot.falling if tiles & blast]
    blocked = sum(1 for q in blast if q not in bot.free)
    edge = min(c[0] - bot.ox, bot.ox + bot.w - 1 - c[0], c[1] - bot.oy, bot.oy + bot.h - 1 - c[1])
    vals = [160 / (1 + (d["age"] + D) / 30) for d in cov]
    return [
        E, v, float(Pm.max()) if Pm.size else 0.0, pc, clear, v + clear,
        bot.prior.get(c, 0.0) if hasattr(bot, "prior") else 0.0,
        len(bot.dinos), len(cov), cov1, *sp, sum(vals), min((d["age"] for d in cov), default=999),
        min(exits, default=99), (sum(exits) / len(exits)) if exits else 99, min_dist, frac,
        len(bot.falling), fall_near, overlap, min(gaps, default=9),
        int(c in follow), blocked, edge, i,
        MODES.index(mode) if mode in MODES else 4, int(flags.get("trapped", 0)), int(flags.get("rex_risk", 0)),
        int(flags.get("pack", 0)), bot.t / 1000,
    ]


# ---------------------------------------------------------------- version 2: wave inputs
WAVE_FEATURES = ["wave_pos", "wave_since", "wave_to_next", "wave_left", "wave_size", "open_waves", "oldest_left"]
ALL_FEATURES = FEATURES + WAVE_FEATURES          # dataset columns; each model's features.json picks its subset


class WaveTracker:
    """waves from new dino ids (spawns <= 2 ticks apart = one wave); call update(s) EVERY tick"""

    def __init__(self, cadence=100):
        self.seen, self.last_spawn, self.n, self.waves, self.cadence = set(), -99, 0, [], cadence

    def update(self, s):
        T = s["currentTick"]
        new = [d["id"] for d in s["dinosaurs"] if d["id"] not in self.seen]
        if new:
            if T - self.last_spawn > 2:
                self.n += 1
                self.waves.append({"start": T, "ids": set()})
            self.last_spawn = T
            self.seen.update(new)
            self.waves[-1]["ids"].update(new)

    def features(self, s):
        if not self.waves:
            return [0, 0, self.cadence, 0, 0, 0, 0]
        T = s["currentTick"]
        alive = {d["id"] for d in s["dinosaurs"]}
        cur = self.waves[-1]
        since = T - cur["start"]
        left = [len(w["ids"] & alive) for w in self.waves]
        open_ = [x for x in left if x]
        return [(self.n - 1) % 10 + 1, since, max(0, self.cadence - since), left[-1], len(cur["ids"]),
                len(open_), open_[0] if open_ else 0]


# ---------------------------------------------------------------- version 2: rule-model inputs
# A second movement forecast (behavior_model.py, 16 spec rules x learned weights), independent of CNN4.
RULE_FEATURES = ["rule_E", "rule_v", "rule_pmax"]
ALL_FEATURES = ALL_FEATURES + RULE_FEATURES
RULE_STEPS = 3                        # a dino gets 3 moves between seeing a meteor and its impact
NAN3 = [float("nan")] * 3


class MoveTracker:
    """last non-zero move per dino id (the rule model's 'repeat' rule); call update(s) EVERY tick"""

    def __init__(self):
        self.prev, self.last = {}, {}

    def update(self, s):
        now = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        for i, p in now.items():
            q = self.prev.get(i)
            if q is not None:
                mv = (p[0] - q[0], p[1] - q[1])
                if mv != (0, 0) and abs(mv[0]) + abs(mv[1]) == 1:
                    self.last[i] = mv
        self.last = {i: m for i, m in self.last.items() if i in now}
        self.prev = now


def rule_forecast(model, s, last):
    """once per tick: per-dino tile probabilities after RULE_STEPS moves + kill value per dino"""
    import behavior_model as BM
    b = BM.Board(s)
    dists = model.forecast(b, last, RULE_STEPS)
    val = {d["id"]: 160 / (1 + (d["age"] + RULE_STEPS + 1) / 30) for d in s["dinosaurs"]}
    return [(dist, val.get(i, 0.0)) for i, dist in dists.items()]


def rule_features(fc, blast):
    """rule-model inputs for one blast (set of world tiles); fc = rule_forecast(...) or None"""
    if fc is None:
        return NAN3
    E = v = pmax = 0.0
    for dist, val in fc:
        p = sum(pr for t, pr in dist.items() if t in blast)
        E += p
        v += p * val
        pmax = max(pmax, p)
    return [E, v, pmax]
