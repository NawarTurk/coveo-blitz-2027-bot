#!/usr/bin/env python3
"""
spawns.py - where dinos spawn.  Answers Q8-Q15 of results/research_questions.md.

A spawn = a dino seen for the first time with age <= 1; positions relative to the visible 20x20 window.

Usage:  python analysis/spawns.py --logs local_game_logs [--games 300]
Writes: results/analysis/spawns.txt
"""
import random
from collections import Counter, defaultdict

from common import Report, args_and_files, run_games, load, events, manh, origin, tile, pct, mean

H = W = 20
FOOT = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if abs(a) + abs(b) <= 2]


def blk(x, y, n):
    return (min(n - 1, y * n // H), min(n - 1, x * n // W))


def one(path):
    recs = load(path)
    if not recs:
        return None
    born, died, waves, alive = events(recs)
    by_t = {s["currentTick"]: s for s, _ in recs}
    rng = random.Random(hash(path) & 0xffff)
    out = dict(sp=[], waves=[], elev=Counter(), elev_all=Counter(), mtn_near=[0, 0], mtn_near_all=[0, 0],
               dist_dino=[], dist_dino_rand=[], on_corpse=[0, 0], corpse_rate=[0, 0], on_lava=0, spawns=0)
    for k, w in enumerate(waves):
        pts = []
        for i in w["ids"]:
            b = born[i]
            if b["age0"] > 1:
                continue
            x, y = b["scr"]
            if not (0 <= x < W and 0 <= y < H):
                continue
            out["sp"].append((x, y, b["sp"], k % 10))
            pts.append((x, y))
            s = by_t.get(b["t"])
            if s is None:
                continue
            out["spawns"] += 1
            t = tile(s, b["p"])
            if t:
                out["elev"][t["elevation"]] += 1
                out["on_lava"] += t["hasLava"]
            ox, oy = origin(s)
            near = any(tile(s, (b["p"][0] + a, b["p"][1] + c)) and tile(s, (b["p"][0] + a, b["p"][1] + c))["isMountain"]
                       for a in range(-2, 3) for c in range(-2, 3) if abs(a) + abs(c) <= 2)
            out["mtn_near"][0] += near; out["mtn_near"][1] += 1
            # baseline: a random free tile of the same state
            free = [(ox + j, oy + i) for i, row in enumerate(s["map"]["tiles"]) for j, tl in enumerate(row)
                    if not (tl["isImpassable"] or tl["isMountain"] or tl["hasLava"])]
            if free:
                q = rng.choice(free)
                tq = tile(s, q)
                out["elev_all"][tq["elevation"]] += 1
                nq = any(tile(s, (q[0] + a, q[1] + c)) and tile(s, (q[0] + a, q[1] + c))["isMountain"]
                         for a in range(-2, 3) for c in range(-2, 3) if abs(a) + abs(c) <= 2)
                out["mtn_near_all"][0] += nq; out["mtn_near_all"][1] += 1
                others = [(d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"] if d["id"] != i]
                if others:
                    out["dist_dino"].append(min(manh(b["p"], o) for o in others))
                    out["dist_dino_rand"].append(min(manh(q, o) for o in others))
                cs = {(c["x"], c["y"]) for c in s.get("corpses", [])}
                out["on_corpse"][0] += b["p"] in cs; out["on_corpse"][1] += 1
                out["corpse_rate"][0] += q in cs; out["corpse_rate"][1] += 1
        out["waves"].append(pts)
    return out


def best_fixed_blast(pts_by_wave):
    """spawns covered by the best single fixed blast centre (count grid + diamond sum)"""
    g = [[0] * W for _ in range(H)]
    for w in pts_by_wave:
        for x, y in w:
            g[y][x] += 1
    return max(sum(g[cy + b][cx + a] for a, b in FOOT if 0 <= cx + a < W and 0 <= cy + b < H)
               for cx in range(W) for cy in range(H))


def main():
    a, files = args_and_files(__doc__)
    G = run_games(one, files, a.workers)
    rep = Report("spawns")
    S = [p for g in G for p in g["sp"]]
    n = len(S)
    rep.line(f"spawns.py | {len(G)} games, {n} spawns from {a.logs}")
    if not n:
        rep.save(); return

    # Q8 exact tile: same cycle position across games, and repeats inside a game
    by_pos = defaultdict(list)
    for gi, g in enumerate(G):
        for k, w in enumerate(g["waves"]):
            by_pos[k % 10].append(set(w))
    ov, exp = [], []
    for p, sets in by_pos.items():
        for i in range(min(len(sets), 60)):
            for j in range(i + 1, min(len(sets), 60)):
                A, B = sets[i], sets[j]
                if A and B:
                    ov.append(len(A & B) / min(len(A), len(B)))
                    exp.append(len(B) / 400)
    rep.q(8, "Can we predict the exact spawn tile?",
          f"same wave number in two games shares {100 * mean(ov):.1f}% of tiles vs {100 * mean(exp):.1f}% by chance"
          if ov else "not enough games")

    # Q9 grids
    for nb in (3, 4):
        c = Counter(blk(x, y, nb) for x, y, _, _ in S)
        vals = [100 * c[(i, j)] / n for i in range(nb) for j in range(nb)]
        noise = 200 * ((1 / nb ** 2) * (1 - 1 / nb ** 2) / n) ** 0.5
        rep.d(f"{nb}x{nb} grid: blocks {min(vals):.1f}-{max(vals):.1f}% (even {100 / nb ** 2:.1f}%, noise +-{noise:.1f})")
        for i in range(nb):
            rep.d("   " + " ".join(f"{100 * c[(i, j)] / n:5.1f}" for j in range(nb)))
    fixed = best_fixed_blast([g["waves"][k] for g in G for k in range(len(g["waves"]))])
    rep.q(9, "Can we predict spawns with a 3x3 or 4x4 grid?",
          f"best single fixed blast (hindsight) covers {pct(fixed, n)} of spawns vs {13 / 4:.1f}% for a random spot")

    # Q10 species
    rep.q(10, "Do spawn positions differ by species?", "4x4 block range per species (even 6.25%):")
    for sp in "SVTR":
        xs = [(x, y) for x, y, s_, _ in S if s_ == sp]
        if len(xs) < 20:
            continue
        c = Counter(blk(x, y, 4) for x, y in xs)
        vals = [100 * c[(i, j)] / len(xs) for i in range(4) for j in range(4)]
        noise = 200 * (0.0625 * 0.9375 / len(xs)) ** 0.5
        rep.d(f"{sp}: n={len(xs)} range {min(vals):.1f}-{max(vals):.1f}% (noise +-{noise:.1f})")

    # Q11 wave number
    rep.q(11, "Do spawn positions depend on the wave number?", "3x3 block range per cycle position (even 11.1%):")
    for p in range(10):
        xs = [(x, y) for x, y, _, q in S if q == p]
        if len(xs) < 20:
            continue
        c = Counter(blk(x, y, 3) for x, y in xs)
        vals = [100 * c[(i, j)] / len(xs) for i in range(3) for j in range(3)]
        noise = 200 * ((1 / 9) * (8 / 9) / len(xs)) ** 0.5
        rep.d(f"wave {p + 1:2d}: n={len(xs)} range {min(vals):.1f}-{max(vals):.1f}% (noise +-{noise:.1f})")

    # Q12 repeat
    hit = tot = 0
    for g in G:
        ws = [w for w in g["waves"] if w]
        for A, B in zip(ws, ws[1:]):
            top = Counter(blk(x, y, 4) for x, y in A).most_common(1)[0][0]
            hit += sum(1 for x, y in B if blk(x, y, 4) == top); tot += len(B)
    rep.q(12, "Does a wave spawn where the previous wave spawned?",
          f"next wave's spawns in the previous wave's busiest 4x4 block: {pct(hit, tot)} (chance 6.25%)")

    # Q13 clustering
    rng = random.Random(0)
    real, rnd = [], []
    for g in G:
        for w in g["waves"]:
            if len(w) >= 3:
                real.append(mean([manh(p, q) for i, p in enumerate(w) for q in w[i + 1:]]))
                r = [(rng.randrange(W), rng.randrange(H)) for _ in w]
                rnd.append(mean([manh(p, q) for i, p in enumerate(r) for q in r[i + 1:]]))
    rep.q(13, "Do dinos of one wave spawn close together?",
          f"mean distance between dinos of a wave {mean(real):.1f} vs {mean(rnd):.1f} for random placement"
          if real else "not enough waves")

    # Q14 terrain
    m = {}
    for g in G:
        for k in ("elev", "elev_all"):
            m.setdefault(k, Counter()).update(g[k])
        for k in ("mtn_near", "mtn_near_all", "on_corpse", "corpse_rate"):
            m[k] = [x + y for x, y in zip(m.get(k, [0, 0]), g[k])]
    ne, na = sum(m["elev"].values()), sum(m["elev_all"].values())
    bins = [(0, 3), (4, 6), (7, 9), (10, 99)]
    rep.q(14, "Do spawns favour certain terrain (elevation, mountains)?",
          "elevation of spawn tiles vs random free tiles: " + ", ".join(
              f"{lo}-{hi if hi < 99 else '+'}: {pct(sum(v for e, v in m['elev'].items() if lo <= e <= hi), ne)} vs "
              f"{pct(sum(v for e, v in m['elev_all'].items() if lo <= e <= hi), na)}" for lo, hi in bins))
    rep.d(f"mountain within 2 tiles: spawns {pct(*m['mtn_near'])} vs random free tiles {pct(*m['mtn_near_all'])}")
    if ne and len(m["elev"]) == 1:
        rep.d("(no elevation in these logs: old team logs without terrain - use local logs or new traces)")

    # Q15 occupied / lava / corpses
    dd = [x for g in G for x in g["dist_dino"]]
    dr = [x for g in G for x in g["dist_dino_rand"]]
    lava = sum(g["on_lava"] for g in G)
    rep.q(15, "Do spawns depend on occupied tiles, lava, corpses?",
          f"distance to the nearest other dino: spawns {mean(dd):.1f} vs random free tile {mean(dr):.1f}; "
          f"spawns on lava {lava}; on a corpse {pct(*m['on_corpse'])} vs random {pct(*m['corpse_rate'])}"
          if dd else "not enough data")
    rep.save()


if __name__ == "__main__":
    main()
