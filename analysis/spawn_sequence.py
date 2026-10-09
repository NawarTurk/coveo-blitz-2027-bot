#!/usr/bin/env python3
"""
Spawn sequence: inside ONE game, can the next wave's spawn spots be predicted from the previous waves?
(e.g. they rotate around the screen, mirror, shift, or always appear in the same part of the window)

Positions are measured RELATIVE TO THE VISIBLE WINDOW (the map scrolls), 20x20.

Predictors tested (each one aims 3 meteor blasts, 13 tiles each, at the best spots for its guess;
score = % of the new wave's dinos inside a blast). Learned on 80% of games, scored on the other 20%:
  blind            3 blasts at the best fixed spots for a uniform map (baseline)
  screen heatmap   3 blasts where spawns are most frequent on the screen overall
  same as last     the next wave spawns where the previous wave spawned
  rot90/180/270    the previous wave's spots rotated around the screen centre
  mirror x / y     the previous wave's spots mirrored
  best shift       the previous wave's spots shifted by the most common (dx, dy)
  perfect          3 blasts placed knowing the real spots (upper limit)

Also prints how the wave's centre moves from wave to wave (angle around the screen centre).

Usage:   python analysis/spawn_sequence.py --logs local_game_logs [--species Stegosaurus]
"""
import argparse, glob, json, os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np

R = 2
H = W = 20
FOOT = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]


SPECIES = None    # set from --species: only spawns of this species


def waves_of(path, species=None):
    """list of waves; a wave = list of (rx, ry) window-relative spawn tiles (spawns <= 2 ticks apart merged)"""
    try:
        seen, spawns = set(), []
        for l in open(path):
            if not l.strip():
                continue
            s = json.loads(l)["state"]
            ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
            for d in s["dinosaurs"]:
                if d["id"] not in seen:
                    seen.add(d["id"])
                    if species is None or d["name"] == species:
                        spawns.append((s["currentTick"], d["position"]["x"] - ox, d["position"]["y"] - oy))
    except Exception:
        return None
    waves, cur, last_t = [], [], None
    for t, x, y in spawns:
        if cur and t - last_t > 2:
            waves.append(cur); cur = []
        cur.append((x, y)); last_t = t
    if cur:
        waves.append(cur)
    return waves[1:] if len(waves) > 2 else None        # drop wave 0 (game start)


def cover_count(centres, pts):
    tiles = {(cx + a, cy + b) for cx, cy in centres for a, b in FOOT}
    return sum(1 for p in pts if p in tiles)


def greedy_centres(weights, k=3):
    """k blast centres maximizing the covered weight (weights: dict tile -> weight)"""
    centres, left = [], dict(weights)
    for _ in range(k):
        best, bv = None, -1
        for cx in range(W):
            for cy in range(H):
                v = sum(left.get((cx + a, cy + b), 0) for a, b in FOOT)
                if v > bv:
                    best, bv = (cx, cy), v
        centres.append(best)
        for a, b in FOOT:
            left.pop((best[0] + a, best[1] + b), None)
    return centres


TRANSFORMS = {
    "same as last": lambda x, y: (x, y),
    "rot90": lambda x, y: (W - 1 - y, x),
    "rot180": lambda x, y: (W - 1 - x, H - 1 - y),
    "rot270": lambda x, y: (y, H - 1 - x),
    "mirror x": lambda x, y: (W - 1 - x, y),
    "mirror y": lambda x, y: (x, H - 1 - y),
}


def angle(pts):
    cx = np.mean([p[0] for p in pts]) - (W - 1) / 2
    cy = np.mean([p[1] for p in pts]) - (H - 1) / 2
    return np.degrees(np.arctan2(cy, cx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--species", default=None, help="Stegosaurus | Velociraptor | Triceratops | Tyrannosaurus")
    args = ap.parse_args()
    global SPECIES
    SPECIES = args.species
    print(f"species: {SPECIES or 'all'}")
    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    with Pool() as pool:
        games = [g for g in pool.map(partial(waves_of, species=SPECIES), files, chunksize=4) if g]
    if not games:
        raise SystemExit("no usable games")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(games))
    cut = int(0.8 * len(games))
    train = [games[i] for i in idx[:cut]]
    test = [games[i] for i in idx[cut:]]
    n_w = sum(len(g) for g in games)
    print(f"{len(games)} games, {n_w} waves ({n_w / len(games):.1f} per game), "
          f"median {np.median([len(w) for g in games for w in g]):.0f} dinos per wave | train {len(train)} / test {len(test)} games")

    # where on the screen do waves spawn?
    heat = Counter(p for g in train for w in g for p in w)
    rows = Counter(p[1] for p in heat.elements())
    tot = sum(rows.values())
    print("\nSPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:")
    print("  " + " ".join(f"{r}:{100 * rows.get(r, 0) / tot:.0f}" for r in range(H)))
    cols = Counter(p[0] for p in heat.elements())
    print("SPAWN COLUMN (0 = left), % of spawns:")
    print("  " + " ".join(f"{c}:{100 * cols.get(c, 0) / tot:.0f}" for c in range(W)))

    # most common shift between consecutive wave centres
    shifts = Counter()
    for g in train:
        for a, b in zip(g, g[1:]):
            ca = (round(np.mean([p[0] for p in a])), round(np.mean([p[1] for p in a])))
            cb = (round(np.mean([p[0] for p in b])), round(np.mean([p[1] for p in b])))
            shifts[(cb[0] - ca[0], cb[1] - ca[1])] += 1
    best_shift = shifts.most_common(1)[0][0]
    TRANSFORMS[f"best shift {best_shift}"] = lambda x, y, s=best_shift: (x + s[0], y + s[1])

    # angle steps between consecutive waves
    steps = Counter()
    for g in games:
        for a, b in zip(g, g[1:]):
            d = (angle(b) - angle(a) + 180) % 360 - 180
            steps[int(45 * round(d / 45))] += 1
    ts = sum(steps.values())
    print("\nWAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:")
    print("  " + " ".join(f"{k:+d}deg:{100 * v / ts:.0f}" for k, v in sorted(steps.items())))
    print("  (a real rotation would put most pairs in one bin; ~equal bins = random)")

    # predictors
    blind = greedy_centres({(x, y): 1 for x in range(W) for y in range(H)})
    heat_c = greedy_centres(heat)
    res = {k: [0, 0] for k in ["blind", "screen heatmap", *TRANSFORMS, "perfect"]}
    for g in test:
        for prev, nxt in zip(g, g[1:]):
            n = len(nxt)
            res["blind"][0] += cover_count(blind, nxt)
            res["screen heatmap"][0] += cover_count(heat_c, nxt)
            for name, f in TRANSFORMS.items():
                guess = Counter(f(x, y) for x, y in prev)
                res[name][0] += cover_count(greedy_centres(guess), nxt)
            res["perfect"][0] += cover_count(greedy_centres(Counter(nxt)), nxt)
            for k in res:
                res[k][1] += n
    print("\n% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):")
    for k, (c, n) in sorted(res.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"  {k:<26} {100 * c / max(n, 1):5.1f}%")
    print("\nIf nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.")


if __name__ == "__main__":
    main()
