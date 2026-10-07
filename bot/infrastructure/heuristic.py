"""
Pure heuristic bot (no ML). Every decision comes from game mechanics, species rules,
recent movement history, map geometry, corpses, lava, elevation and hand-tuned scores.

Use it:   cp bot.py bot_v2_best.py && cp heuristic.py bot.py
Scores:   logs/scores_heuristic.csv
"""
from game_message import *
from collections import Counter, deque
from datetime import datetime
import math, os, time

import msgspec

# ============================== TUNABLE WEIGHTS ==============================
BOT_NAME = "heuristic"
LOG_GAMES = True
TIME_BUDGET_MS = 250          # stop evaluating candidates after this

# --- value of a kill -------------------------------------------------------
AGE_HALF = 20.0               # value(age) = 1 / (1 + age / AGE_HALF)
YOUNG_AGE, YOUNG_W = 10, 0.3  # extra for kills younger than YOUNG_AGE at impact
TREX_EAT_RISK = 0.5           # dinos next to a T-Rex may get eaten (0 points)

# --- meteor scoring ----------------------------------------------------------
MULTIKILL_W = 1.0             # scales the game's 0.5*(k-1) same-tick multiplier
SCROLL_SYNC_W = 1.0           # count dinos scrolling out on the impact tick in the multiplier
CLUSTER_W = 0.2               # bonus per extra expected kill beyond the first
UNCERTAINTY_W = 0.3           # penalty * sum p(1-p)
WASTE_PENALTY = 0.4           # if expected kills < MIN_EXPECTED_KILLS (not in straggler mode)
MIN_EXPECTED_KILLS = 0.25
METEOR_MIN_SCORE = 0.05       # below this, don't launch (keep the slot)
LAVA_SYNERGY_W = 0.05         # per lava tile hugging the blast edge
FUTURE_RAPTOR_W = 0.15        # corpses we create * raptors that will come to them

# --- board clear / wave turnover -----------------------------------------------
BOARD_CLEAR_W = 6.0           # bonus * P(every dino dead) — a fresh wave spawns 1 tick later
FEW_DINOS = 3                 # <= this: straggler mode
MEDIUM_DINOS = 9              # <= this: start setting up the clear
STRAGGLER_MULT = 2.0          # kill value multiplier in straggler mode
ISOLATED_W = 0.3              # medium mode: bonus for killing isolated dinos (future stragglers)

# --- species movement ----------------------------------------------------------
STEG_EDGE_ROWS = 4            # Stegosaurus near trailing edge pushes forward
STEG_MOMENTUM = 0.45
MOMENTUM = 0.45               # generic "repeat last move"
HERD_RADIUS = 8
TRI_HIGH_W, TRI_CENTROID_W, TRI_ROT_W = 1.0, 0.6, 0.8
TREX_HUNT_RADIUS = 5
FLEE_BEST_P = 0.75            # probability a threatened dino takes the best escape move

# --- strategy-paragraph mechanics (your logs did NOT confirm these; set the *_BONUS/W to 0 to drop) ---
OPENING_TICKS = 150
NE_ELEVATION = 7
NE_METEOR_BONUS = 0.15        # [unconfirmed] NE / high-elevation opening
NE_VOLCANO_BONUS = 0.2        # [unconfirmed]
TRI_CCW_PROB = 0.83           # [unconfirmed: logs show 50/50] herd rotation
TRI_OFFSET_DEG = 37.0         # [unconfirmed] meteor offset from pack centroid
TRI_OFFSET_BONUS = 0.2
RAPTOR_SCAVENGE_RADIUS = 6.2  # [unconfirmed: logs show no corpse attraction]
RAPTOR_BAIT_BONUS = 0.15
FUNNEL_WIDTH = 4.7            # [unconfirmed] lava funnel
DOUBLE_BOUNCE_DELTA = 3       # [unconfirmed: 0 extra kills in 105k impacts]
DOUBLE_BOUNCE_WEIGHT = 1.0    # chance the outer ring also kills on steep impacts
REPEAT_IMPACT_WINDOW = 15     # [unconfirmed] "seismic cascade"
REPEAT_IMPACT_BONUS = 0.1

# --- volcano scoring -------------------------------------------------------------
LAVA_HORIZON = 8              # ticks of lava flow to simulate
LAVA_AVOID = 0.5              # dinos see lava: chance they still get caught
LAVA_CONTROL_W = 0.08         # per escape tile cut off next to a dino
FUNNEL_BONUS = 0.4
PUSH_BONUS = 0.2              # lava pushing a dino toward an in-flight meteor
PERSIST_W = 0.2               # volcanoes far from the trailing edge live longer
RIDGE_BONUS = 0.15
CHAIN_BONUS = 0.2
VOLCANO_MIN_SCORE = 0.3       # needed to beat a meteor slot
VOLCANO_MIN_SCORE_WHEN_FULL = 0.1

# --- search limits -----------------------------------------------------------------
MAX_CANDIDATES = 30
MAX_VOLCANO_CANDIDATES = 15
MAX_POSITIONS = 5             # positions kept per dino in the probabilistic rollout
# =================================================================================

MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def age_value(age):
    return 1.0 / (1.0 + age / AGE_HALF)


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def sgn(v):
    return (v > 0) - (v < 0)


class Zone:
    __slots__ = ("c", "tiles", "land", "ring")

    def __init__(self, c, tiles, land, ring=frozenset()):
        self.c, self.tiles, self.land, self.ring = c, tiles, land, ring


# ================================ per-tick context ================================
class Ctx:
    def __init__(self, s, hist, recent_impacts):
        c, m = s["constants"], s["map"]
        self.t = s["currentTick"]
        self.D, self.R, self.shift = c["meteorDelay"], c["meteorRadius"], c["mapShiftInterval"]
        self.maxM, self.maxV = c["maxMeteors"], c["maxVolcanoes"]
        self.ox, self.oy, self.H, self.W = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        tiles = m["tiles"]
        self.elev = [[t["elevation"] for t in r] for r in tiles]
        self.imp = [[t["isImpassable"] for t in r] for r in tiles]
        self.lava = [[t["hasLava"] for t in r] for r in tiles]
        self.lava_set = {(j + self.ox, i + self.oy) for i in range(self.H) for j in range(self.W) if self.lava[i][j]}
        self.foot = [(dx, dy) for dx in range(-self.R, self.R + 1) for dy in range(-self.R, self.R + 1)
                     if abs(dx) + abs(dy) <= self.R]
        self.ring = [(dx, dy) for dx in range(-self.R - 1, self.R + 2) for dy in range(-self.R - 1, self.R + 2)
                     if abs(dx) + abs(dy) == self.R + 1]
        self.meteors = [Zone((me["target"]["x"], me["target"]["y"]),
                             frozenset((p["x"], p["y"]) for p in me["impactedTiles"]),
                             self.t + me["turnsUntilImpact"] - 1) for me in s["meteors"]]
        self.corpses = [(p["x"], p["y"]) for p in s["corpses"]]
        self.mountains = [(p["x"], p["y"]) for p in s["mountains"]]
        self.volcanoes = [(v["position"]["x"], v["position"]["y"]) for v in s["volcanoes"]]
        self.recent_impacts = [(x, y) for x, y, tk in recent_impacts if self.t - tk <= REPEAT_IMPACT_WINDOW]

        self.dinos = []
        for d in s["dinosaurs"]:
            pos = (d["position"]["x"], d["position"]["y"])
            h = hist.get(d["id"], ())
            deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(list(h)[:-1], list(h)[1:])]
            last = deltas[-1] if deltas else (0, 0)
            nz = [m_ for m_ in deltas[-5:] if m_ != (0, 0) and m_ in MOVES]
            dom = Counter(nz).most_common(1)[0][0] if nz else (0, 0)
            self.dinos.append({"id": d["id"], "name": d["name"], "age": d["age"], "pos": pos,
                               "last": last if last in MOVES else (0, 0), "dom": dom})
        self._species_context()
        self.cache = {}

    def in_window(self, x, y):
        return 0 <= y - self.oy < self.H and 0 <= x - self.ox < self.W

    def tile(self, x, y):
        return y - self.oy, x - self.ox

    def blocked(self, x, y, oyk):
        i, j = y - self.oy, x - self.ox
        if not (0 <= i < self.H and 0 <= j < self.W) or y < oyk:
            return True
        return self.imp[i][j] or self.lava[i][j]

    def _species_context(self):
        """herd centroids, high ground, corpse targets, prey — computed once per tick"""
        tri = [d for d in self.dinos if d["name"] == "Triceratops"]
        self.herd = {}
        for d in tri:
            mates = [e["pos"] for e in tri if manh(e["pos"], d["pos"]) <= HERD_RADIUS]
            if len(mates) >= 2:
                self.herd[d["id"]] = (sum(p[0] for p in mates) / len(mates), sum(p[1] for p in mates) / len(mates), len(mates))
        self.high = {}
        for d in tri:
            x, y = d["pos"]
            best, bv = None, -1e9
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if abs(dx) + abs(dy) > 4 or not self.in_window(x + dx, y + dy):
                        continue
                    i, j = self.tile(x + dx, y + dy)
                    if self.imp[i][j] or self.lava[i][j]:
                        continue
                    if self.elev[i][j] > bv:
                        best, bv = (x + dx, y + dy), self.elev[i][j]
            self.high[d["id"]] = best
        self.corpse_target = {}
        for d in self.dinos:
            if d["name"] != "Velociraptor" or not self.corpses:
                continue
            c = min(self.corpses, key=lambda p: math.dist(p, d["pos"]))
            if math.dist(c, d["pos"]) <= RAPTOR_SCAVENGE_RADIUS:
                self.corpse_target[d["id"]] = c
        self.prey, self.crowd = {}, {}
        for d in self.dinos:
            if d["name"] != "Tyrannosaurus":
                continue
            others = [e for e in self.dinos if e["id"] != d["id"]]
            prey = [e["pos"] for e in others if e["name"] != "Tyrannosaurus" and manh(e["pos"], d["pos"]) <= TREX_HUNT_RADIUS]
            if prey:
                self.prey[d["id"]] = min(prey, key=lambda p: manh(p, d["pos"]))
            near = [e["pos"] for e in others if manh(e["pos"], d["pos"]) <= 3]
            if near:
                self.crowd[d["id"]] = (sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near))
        trex = [d["pos"] for d in self.dinos if d["name"] == "Tyrannosaurus"]
        self.eat_risk = {d["id"] for d in self.dinos
                         if d["name"] != "Tyrannosaurus" and any(manh(d["pos"], p) <= 2 for p in trex)}


# ============================== movement prediction ==============================
def vec_to_dist(vx, vy, p1, p2, ps):
    d = dict.fromkeys(MOVES, 0.0)
    if abs(vx) + abs(vy) < 1e-9:
        d[(0, 0)] = ps + p1
    else:
        if abs(vx) >= abs(vy):
            prim, sec = (sgn(vx), 0), ((0, sgn(vy)) if abs(vy) > 1e-9 else None)
        else:
            prim, sec = (0, sgn(vy)), ((sgn(vx), 0) if abs(vx) > 1e-9 else None)
        d[prim] += p1
        d[(0, 0)] += ps
        if sec:
            d[sec] += p2
    rest = max(0.0, 1.0 - sum(d.values()))
    zero = [m for m in MOVES if d[m] == 0.0]
    for m in zero:
        d[m] = rest / len(zero)
    if not zero:
        d[(0, 0)] += rest
    return d


def exit_dist(p, zones, R):
    e = 0
    for z in zones:
        if p in z.tiles:
            e = max(e, R + 1 - manh(p, z.c))
    return e


def predict_species_move(ctx, d, pos, oyk, zones):
    """probability of each of the 5 moves for dino d standing at pos"""
    x, y = pos
    R = ctx.R
    sig = tuple(exit_dist((x + mx, y + my), zones, R) for mx, my in MOVES)
    key = (d["id"], pos, oyk, sig)
    if key in ctx.cache:
        return ctx.cache[key]
    legal = [m for m in MOVES if m == (0, 0) or not ctx.blocked(x + m[0], y + m[1], oyk)]

    if sig[0] > 0:                                    # ---------- threatened: flee
        es = {m: sig[MOVES.index(m)] for m in legal}
        best = min(es.values())
        bests = [m for m in legal if es[m] == best]
        if d["name"] == "Velociraptor" and (0, 0) in bests:
            choice = (0, 0)                           # moves as little as necessary
        else:
            choice = max(bests, key=lambda m: min((manh((x + m[0], y + m[1]), z.c) for z in zones), default=0))
        dist = dict.fromkeys(MOVES, 0.0)
        dist[choice] = FLEE_BEST_P
        others_best = [m for m in bests if m != choice]
        rest_moves = [m for m in legal if m != choice and m not in others_best]
        for m in others_best:
            dist[m] += 0.12 / len(others_best)
        left = 1 - FLEE_BEST_P - (0.12 if others_best else 0)
        for m in (rest_moves or [choice]):
            dist[m] += left / len(rest_moves or [choice])
    else:                                             # ---------- calm: species behaviour
        name = d["name"]
        if name == "Stegosaurus":
            if y - oyk < STEG_EDGE_ROWS:
                dist = vec_to_dist(0, 1, 0.7, 0.0, 0.15)
            elif d["dom"] != (0, 0):
                dist = vec_to_dist(*d["dom"], STEG_MOMENTUM, 0.0, 0.25)
                dist[(0, 1)] += 0.1
            else:
                dist = vec_to_dist(0, 1, 0.35, 0.0, 0.35)
        elif name == "Velociraptor" and d["id"] in ctx.corpse_target:
            cx, cy = ctx.corpse_target[d["id"]]
            dist = vec_to_dist(cx - x, cy - y, 0.65, 0.15, 0.1) if (cx, cy) != pos else vec_to_dist(0, 0, 0.0, 0.0, 0.7)
        elif name == "Triceratops" and d["id"] in ctx.herd:
            hx, hy, n = ctx.herd[d["id"]]
            vx = vy = 0.0
            hi = ctx.high.get(d["id"])
            if hi:
                vx += TRI_HIGH_W * (hi[0] - x); vy += TRI_HIGH_W * (hi[1] - y)
            rx, ry = x - hx, y - hy
            r = math.hypot(rx, ry)
            if r > 2:
                vx -= TRI_CENTROID_W * rx / r * 2; vy -= TRI_CENTROID_W * ry / r * 2
            if r > 0.5:                               # tangent, counter-clockwise on screen (y down)
                rot = TRI_ROT_W * (2 * TRI_CCW_PROB - 1)
                vx += rot * ry / r * 2; vy += rot * -rx / r * 2
            dist = vec_to_dist(vx, vy, 0.5, 0.15, 0.2)
        elif name == "Tyrannosaurus" and d["id"] in ctx.prey:
            px, py = ctx.prey[d["id"]]
            dist = vec_to_dist(px - x, py - y, 0.55, 0.15, 0.15)
        elif name == "Tyrannosaurus" and d["id"] in ctx.crowd:
            cx, cy = ctx.crowd[d["id"]]
            dist = vec_to_dist(x - cx, y - cy, 0.45, 0.15, 0.25)
        else:
            dist = vec_to_dist(*d["last"], MOMENTUM, 0.0, 0.3) if d["last"] != (0, 0) else vec_to_dist(0, 0, 0, 0, 0.4)
        # dinos see threats & edges: avoid stepping into danger / onto the trailing row
        for k, m in enumerate(MOVES):
            if m == (0, 0):
                continue
            f = 1.0
            if sig[k] > 0:
                f = 0.1
            elif y + m[1] - oyk <= 0:
                f = 0.3
            moved = dist[m] * (1 - f)
            dist[m] -= moved
            dist[(0, 0)] += moved
    # blocked moves -> stay
    for m in MOVES:
        if m != (0, 0) and m not in legal and dist[m] > 0:
            dist[(0, 0)] += dist[m]
            dist[m] = 0.0
    out = [(m, p) for m, p in dist.items() if p > 1e-6]
    ctx.cache[key] = out
    return out


def predict_dino_heuristic(ctx, d, extra=None, keep_dists=False):
    """roll one dino forward until (and including) the impact tick of `extra` (or t+D).
    returns kill prob by `extra`, death prob by other meteors, scroll-out prob, per-step distributions"""
    dist = {d["pos"]: 1.0}
    oyk = ctx.oy
    kill_extra = dead_other = dead_scroll = scroll_last = 0.0
    dists = [dist] if keep_dists else None
    for k in range(ctx.D + 1):
        tau = ctx.t + k
        for z in ctx.meteors:                         # existing meteors landing now (before moves)
            if z.land == tau:
                hit = sum(p for q, p in dist.items() if q in z.tiles)
                if hit:
                    dead_other += hit
                    dist = {q: p for q, p in dist.items() if q not in z.tiles}
        if k == ctx.D:
            if extra is not None:
                inside = sum(p for q, p in dist.items() if q in extra.tiles)
                ring = sum(p for q, p in dist.items() if q in extra.ring)
                kill_extra = inside + DOUBLE_BOUNCE_WEIGHT * ring
            break
        zones = [z for z in ctx.meteors if z.land > tau] + ([extra] if extra is not None else [])
        new = {}
        for q, p in dist.items():
            for m, pm in predict_species_move(ctx, d, q, oyk, zones):
                nq = (q[0] + m[0], q[1] + m[1])
                new[nq] = new.get(nq, 0.0) + p * pm
        if (tau + 1) % ctx.shift == 0:                # window scrolls: trailing row disappears
            oyk += 1
            gone = sum(p for q, p in new.items() if q[1] < oyk)
            if gone:
                dead_scroll += gone
                if k == ctx.D - 1:
                    scroll_last = gone
                new = {q: p for q, p in new.items() if q[1] >= oyk}
        if len(new) > MAX_POSITIONS:
            total = sum(new.values())
            top = sorted(new.items(), key=lambda kv: -kv[1])[:MAX_POSITIONS]
            kept = sum(p for _, p in top)
            new = {q: p * total / kept for q, p in top}
        dist = new
        if keep_dists:
            dists.append(dist)
    return {"kill": min(kill_extra, 1.0), "other": dead_other, "scroll": dead_scroll,
            "scroll_last": scroll_last, "dists": dists, "alive": sum(dist.values())}


# ================================ meteor targeting ================================
def mode(dist):
    return max(dist.items(), key=lambda kv: kv[1])[0]


def generate_meteor_candidates(ctx, base):
    pts = set()
    modes = {d["id"]: mode(base[d["id"]]["dists"][-1]) if base[d["id"]]["dists"][-1] else d["pos"] for d in ctx.dinos}
    for d in ctx.dinos:
        mx, my = modes[d["id"]]
        pts.add(d["pos"])
        for dx, dy in MOVES:
            pts.add((mx + dx, my + dy))
    ds = ctx.dinos
    for a in range(len(ds)):                          # cluster midpoints
        for b in range(a + 1, len(ds)):
            pa, pb = modes[ds[a]["id"]], modes[ds[b]["id"]]
            if manh(pa, pb) <= 2 * ctx.R + 1:
                pts.add((round((pa[0] + pb[0]) / 2), round((pa[1] + pb[1]) / 2)))
    th = math.radians(TRI_OFFSET_DEG)
    ctx.offset_pts = set()
    for d in ds:                                      # herd centroid + 37 deg CCW offset points
        if d["id"] in ctx.herd:
            hx, hy, _ = ctx.herd[d["id"]]
            pts.add((round(hx), round(hy)))
            vx, vy = d["pos"][0] - hx, d["pos"][1] - hy
            ox_, oy_ = vx * math.cos(th) + vy * math.sin(th), -vx * math.sin(th) + vy * math.cos(th)
            p = (round(hx + ox_), round(hy + oy_))
            pts.add(p); ctx.offset_pts.add(p)
    raptors = [d["pos"] for d in ds if d["name"] == "Velociraptor"]
    for c in ctx.corpses:                             # corpse bait
        if any(math.dist(c, r) <= RAPTOR_SCAVENGE_RADIUS for r in raptors):
            pts.add(c)
    if ctx.t < OPENING_TICKS:                         # NE high ground
        for d in ds:
            x, y = d["pos"]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if ctx.in_window(x + dx, y + dy):
                        i, j = ctx.tile(x + dx, y + dy)
                        if i < ctx.H / 2 and j >= ctx.W / 2 and ctx.elev[i][j] >= NE_ELEVATION:
                            pts.add((x + dx, y + dy))
    pts |= set(ctx.recent_impacts)
    pts = [p for p in pts if ctx.in_window(*p)]

    def quick(p):
        return sum(age_value(d["age"]) for d in ds if manh(modes[d["id"]], p) <= ctx.R) + \
               0.3 * sum(age_value(d["age"]) for d in ds if manh(d["pos"], p) <= ctx.R)
    pts.sort(key=quick, reverse=True)
    return pts[:MAX_CANDIDATES]


def steep_impact(ctx, x, y):
    i, j = ctx.tile(x, y)
    drops = []
    for dx, dy in MOVES[1:]:
        if ctx.in_window(x + dx, y + dy):
            a, b = ctx.tile(x + dx, y + dy)
            if not ctx.imp[a][b]:
                drops.append(abs(ctx.elev[i][j] - ctx.elev[a][b]))
    return bool(drops) and max(drops) > DOUBLE_BOUNCE_DELTA


def board_clear_bonus(ctx, p_death):
    n = len(ctx.dinos)
    if n == 0 or n > MEDIUM_DINOS:
        return 0.0
    p = 1.0
    for v in p_death.values():
        p *= min(1.0, v)
    return BOARD_CLEAR_W * p


def score_meteor(ctx, cand, base):
    cx, cy = cand
    tiles = frozenset((cx + dx, cy + dy) for dx, dy in ctx.foot if ctx.in_window(cx + dx, cy + dy))
    ring = frozenset((cx + dx, cy + dy) for dx, dy in ctx.ring if ctx.in_window(cx + dx, cy + dy)) \
        if steep_impact(ctx, cx, cy) else frozenset()
    extra = Zone(cand, tiles, ctx.t + ctx.D, ring)
    n = len(ctx.dinos)
    reach = ctx.R + ctx.D + 2
    kv = E = unc = young = iso = 0.0
    p_death = {}
    killed_ids = []
    for d in ctx.dinos:
        b = base[d["id"]]
        if manh(d["pos"], cand) > reach:
            p_death[d["id"]] = b["other"] + b["scroll"]
            continue
        r = predict_dino_heuristic(ctx, d, extra)
        p = r["kill"]
        p_death[d["id"]] = r["other"] + r["scroll"] + p
        if p <= 0:
            continue
        val = age_value(d["age"] + ctx.D) * (1 - TREX_EAT_RISK if d["id"] in ctx.eat_risk else 1.0)
        kv += p * val
        E += p
        unc += p * (1 - p)
        if d["age"] + ctx.D < YOUNG_AGE:
            young += p
        if FEW_DINOS < n <= MEDIUM_DINOS and all(manh(d["pos"], e["pos"]) > 5 for e in ctx.dinos if e is not d):
            iso += p
        killed_ids.append(d["id"])
    E_scroll = sum(base[d["id"]]["scroll_last"] for d in ctx.dinos) if (ctx.t + ctx.D) % ctx.shift == 0 else 0.0
    E_tot = E + SCROLL_SYNC_W * E_scroll
    immediate = kv * (STRAGGLER_MULT if n <= FEW_DINOS else 1.0)
    multikill = kv * MULTIKILL_W * 0.5 * max(E_tot - 1, 0)
    clustering = CLUSTER_W * max(E - 1, 0)
    clear = board_clear_bonus(ctx, p_death)
    # documented-strategy bonuses
    species_bonus = 0.0
    if any(manh(cand, p) <= 1 for p in getattr(ctx, "offset_pts", ())):
        species_bonus += TRI_OFFSET_BONUS
    if ctx.t < OPENING_TICKS:
        i, j = ctx.tile(cx, cy)
        if i < ctx.H / 2 and j >= ctx.W / 2 and ctx.elev[i][j] >= NE_ELEVATION:
            species_bonus += NE_METEOR_BONUS
    if any(manh(cand, p) <= 1 for p in ctx.recent_impacts):
        species_bonus += REPEAT_IMPACT_BONUS
    raptors_near = sum(1 for d in ctx.dinos if d["name"] == "Velociraptor" and d["id"] not in killed_ids
                       and math.dist(d["pos"], cand) <= RAPTOR_SCAVENGE_RADIUS)
    if cand in ctx.corpses:
        species_bonus += RAPTOR_BAIT_BONUS * raptors_near
    future = FUTURE_RAPTOR_W * E * raptors_near
    lava_syn = LAVA_SYNERGY_W * sum(1 for dx, dy in ctx.ring if (cx + dx, cy + dy) in ctx.lava_set)
    waste = WASTE_PENALTY if (E < MIN_EXPECTED_KILLS and n > FEW_DINOS) else 0.0
    score = (immediate + multikill + clear + YOUNG_W * young + clustering + lava_syn + future
             + species_bonus + ISOLATED_W * iso - waste - UNCERTAINTY_W * unc)
    return score, {"E": E, "kv": kv, "clear": clear}


# ================================ volcano / lava ================================
def simulate_lava(ctx, src, horizon=LAVA_HORIZON):
    """tile -> tick lava arrives. Each step lava moves to the lowest neighbouring cells (ties split)."""
    arrive, frontier = {}, [src]
    for step in range(1, horizon + 1):
        new = []
        for (x, y) in frontier:
            nbrs = []
            for dx, dy in MOVES[1:]:
                q = (x + dx, y + dy)
                if not ctx.in_window(*q) or q in arrive or q == src or q in ctx.lava_set:
                    continue
                i, j = ctx.tile(*q)
                if ctx.imp[i][j]:
                    continue
                nbrs.append((ctx.elev[i][j], q))
            if nbrs:
                lo = min(e for e, _ in nbrs)
                for e, q in nbrs:
                    if e == lo and q not in arrive:
                        arrive[q] = step
                        new.append(q)
        if not new:
            break
        frontier = new
    return arrive


def score_volcano(ctx, mtn, base, last_volcano):
    lava = simulate_lava(ctx, mtn)
    if not lava:
        return -1.0
    walls = set(lava) | ctx.lava_set
    in_flight = set().union(*[z.tiles for z in ctx.meteors]) if ctx.meteors else set()
    direct = control = funnel = push = 0.0
    for d in ctx.dinos:
        val = age_value(d["age"] + ctx.D)
        dists = base[d["id"]]["dists"]
        p_hit = 0.0
        for step in range(1, LAVA_HORIZON + 1):
            dist = dists[min(step, len(dists) - 1)]
            p_hit = max(p_hit, sum(p for q, p in dist.items() if lava.get(q, 99) <= step))
        direct += (1 - LAVA_AVOID) * p_hit * val
        x, y = d["pos"]
        if min((manh(d["pos"], q) for q in lava), default=99) <= 3:
            cut = sum(1 for dx, dy in MOVES[1:] if (x + dx, y + dy) in lava)
            control += LAVA_CONTROL_W * cut * val
            if any(manh(d["pos"], q) <= 3 for q in in_flight) and min(manh(d["pos"], q) for q in lava) <= 2:
                push += PUSH_BONUS * val
        for axis in ((1, 0), (0, 1)):                 # funnel: corridor between walls <= FUNNEL_WIDTH
            lo = hi = None
            for s in range(1, 7):
                if lo is None and (x - axis[0] * s, y - axis[1] * s) in walls:
                    lo = s
                if hi is None and (x + axis[0] * s, y + axis[1] * s) in walls:
                    hi = s
            if lo and hi and lo + hi - 1 <= FUNNEL_WIDTH and any(
                    (x + sg * axis[0] * s, y + sg * axis[1] * s) in lava for sg, s in ((-1, lo), (1, hi))):
                funnel += FUNNEL_BONUS * val
    i, j = ctx.tile(*mtn)
    persist = PERSIST_W * i / max(ctx.H - 1, 1)
    ridge = RIDGE_BONUS * min(4, sum(1 for m in ctx.mountains if m != mtn and manh(m, mtn) <= 2)) / 4
    chain = CHAIN_BONUS if last_volcano and manh(last_volcano, mtn) <= 3 else 0.0
    ne = NE_VOLCANO_BONUS if (ctx.t < OPENING_TICKS and i < ctx.H / 2 and j >= ctx.W / 2) else 0.0
    if direct + control + funnel + push == 0:
        return 0.0                                    # nothing useful nearby: no positional bonuses
    return direct + control + funnel + push + persist + ridge + chain + ne


# ================================ decision ================================
def choose_action(ctx, last_volcano, t0):
    if not ctx.dinos:
        return []
    base = {d["id"]: predict_dino_heuristic(ctx, d, None, keep_dists=True) for d in ctx.dinos}

    best_m, best_ms = None, -1e9
    if len(ctx.meteors) < ctx.maxM:
        for cand in generate_meteor_candidates(ctx, base):
            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS:
                break
            sc, _ = score_meteor(ctx, cand, base)
            if sc > best_ms:
                best_m, best_ms = cand, sc

    best_v, best_vs = None, -1e9
    if len(ctx.volcanoes) < ctx.maxV and ctx.mountains:
        used = set(ctx.volcanoes)
        mtns = [m for m in ctx.mountains if m not in used and ctx.in_window(*m)]
        mtns.sort(key=lambda m: -sum(1.0 / (1 + manh(m, d["pos"])) for d in ctx.dinos))
        for m in mtns[:MAX_VOLCANO_CANDIDATES]:
            if (time.perf_counter() - t0) * 1000 > TIME_BUDGET_MS * 1.5:
                break
            sc = score_volcano(ctx, m, base, last_volcano)
            if sc > best_vs:
                best_v, best_vs = m, sc

    straggler = len(ctx.dinos) <= FEW_DINOS
    m_floor = 0.0 if straggler else METEOR_MIN_SCORE
    if best_m is not None and best_ms >= m_floor and best_ms >= best_vs:
        return [LaunchMeteorAction(target=WorldPosition(*best_m))]
    v_floor = VOLCANO_MIN_SCORE if best_m is not None else VOLCANO_MIN_SCORE_WHEN_FULL
    if best_v is not None and best_vs >= v_floor:
        return [TriggerVolcanoAction(target=WorldPosition(*best_v))]
    if best_m is not None and best_ms > 0:
        return [LaunchMeteorAction(target=WorldPosition(*best_m))]
    return []


# ================================ bot ================================
class Bot:
    def __init__(self):
        self.hist = {}
        self.prev_meteors = []
        self.recent_impacts = deque(maxlen=20)
        self.last_volcano = None
        self.times = []
        if LOG_GAMES:
            os.makedirs("logs", exist_ok=True)
            self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"logs/game_{self.stamp}.jsonl"
            self.log = open(self.log_path, "ab")

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        t0 = time.perf_counter()
        if game_message.lastTickErrors:
            print(game_message.currentTick, game_message.lastTickErrors)
        s = msgspec.to_builtins(game_message)

        # meteors that showed 1 turn left last tick have landed: remember (seismic cascade)
        for (x, y, turns) in self.prev_meteors:
            if turns == 1:
                self.recent_impacts.append((x, y, s["currentTick"] - 1))
        self.prev_meteors = [(m["target"]["x"], m["target"]["y"], m["turnsUntilImpact"]) for m in s["meteors"]]

        # movement history
        alive = set()
        for d in s["dinosaurs"]:
            alive.add(d["id"])
            self.hist.setdefault(d["id"], deque(maxlen=6)).append((d["position"]["x"], d["position"]["y"]))
        for k in [k for k in self.hist if k not in alive]:
            del self.hist[k]

        try:
            ctx = Ctx(s, self.hist, self.recent_impacts)
            actions = self.validate(choose_action(ctx, self.last_volcano, t0), ctx)
        except Exception as e:                        # never crash mid-game
            print("heuristic error:", repr(e))
            actions = []
        if actions and actions[0].type == "TRIGGER_VOLCANO":
            self.last_volcano = (actions[0].target.x, actions[0].target.y)

        self.times.append(time.perf_counter() - t0)
        if LOG_GAMES:
            rec = {"tick": s["currentTick"], "state": game_message, "actions": actions}
            self.log.write(msgspec.json.encode(rec) + b"\n")
            self.log.flush()
        if s["currentTick"] >= s["constants"]["maxTicks"]:
            self.finish(s)
        return actions

    @staticmethod
    def validate(actions, ctx):
        if not actions:
            return []
        a = actions[0]
        p = (a.target.x, a.target.y)
        if a.type == "LAUNCH_METEOR":
            ok = len(ctx.meteors) < ctx.maxM and ctx.in_window(*p)
        else:
            ok = len(ctx.volcanoes) < ctx.maxV and p in ctx.mountains and p not in ctx.volcanoes
        return [a] if ok else []

    def finish(self, s):
        ms = [1000 * t for t in self.times]
        print(f"Final score: {s['score']} | avg {sum(ms) / len(ms):.0f} ms/tick, max {max(ms):.0f} ms")
        if LOG_GAMES:
            self.log.close()
            path = f"logs/scores_{BOT_NAME}.csv"
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("timestamp,log_file,final_tick,final_score\n")
                f.write(f"{self.stamp},{self.log_path},{s['currentTick']},{s['score']}\n")
