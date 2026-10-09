#!/usr/bin/env python3
"""
Spawn blocks: split the screen into 9 blocks and look for patterns in WHICH blocks waves spawn in.

    b1 b2 b3
    b4 b5 b6        (positions relative to the visible 20x20 window)
    b7 b8 b9

Idea 1 - sequence: does the wave's main block follow a pattern from wave to wave (b1 -> b4 -> b5 ...)?
         9x9 transition table of each wave's main block (the block with most spawns), compared with
         the same games with their waves shuffled (= no order). Information in bits: 0 = no pattern.
Idea 2 - memory: if a block was used, is it MORE likely again (repeats) or LESS likely (cycles through
         the other blocks first)? Compared with the overall rate for that block.
Also:   inside one wave, are the dinos spread over more blocks than random would give (one per block)?

Usage:   python analysis/spawn_blocks.py --logs local_game_logs [--species Stegosaurus]
"""
import argparse, glob, json, os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np

H = W = 20


def block(x, y):
    return 3 * min(2, y * 3 // H) + min(2, x * 3 // W) + 1        # 1..9


SPECIES = None    # set from --species: only spawns of this species


def waves_of(path, species=None):
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
                    x, y = d["position"]["x"] - ox, d["position"]["y"] - oy
                    if 0 <= x < W and 0 <= y < H and (species is None or d["name"] == species):
                        spawns.append((s["currentTick"], block(x, y)))
    except Exception:
        return None
    waves, cur, last_t = [], [], None
    for t, b in spawns:
        if cur and t - last_t > 2:
            waves.append(cur); cur = []
        cur.append(b); last_t = t
    if cur:
        waves.append(cur)
    return waves[1:] if len(waves) > 3 else None


def main_block(w):
    c = Counter(w)
    return max(sorted(c), key=lambda b: c[b])


def mutual_info(pairs):
    """bits of information the previous main block gives about the next one"""
    n = len(pairs)
    joint = Counter(pairs)
    a = Counter(p[0] for p in pairs)
    b = Counter(p[1] for p in pairs)
    return sum(c / n * np.log2((c / n) / ((a[x] / n) * (b[y] / n))) for (x, y), c in joint.items())


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
    rng = np.random.default_rng(0)
    allw = [w for g in games for w in g]
    print(f"{len(games)} games, {len(allw)} waves")

    # overall
    tot = Counter(b for w in allw for b in w)
    n = sum(tot.values())
    print("\nWHERE DINOS SPAWN (% per block):")
    for r in range(3):
        print("   " + "  ".join(f"b{3 * r + c + 1}:{100 * tot[3 * r + c + 1] / n:4.1f}%" for c in range(3)))
    print("   (block areas differ slightly: middle row/column are 6 tiles, edges 7)")

    # spread inside a wave vs random
    real = np.mean([len(set(w)) for w in allw])
    probs = np.array([tot[b] / n for b in range(1, 10)])
    sim = np.mean([len(set(rng.choice(9, size=len(w), p=probs))) for w in allw for _ in range(3)])
    print(f"\nINSIDE ONE WAVE: dinos land in {real:.2f} different blocks on average (random would give {sim:.2f})")
    print("   more than random = spread out on purpose; fewer = clustered")

    # idea 1: sequence of main blocks
    pairs = [(main_block(a), main_block(b)) for g in games for a, b in zip(g, g[1:])]
    shuf = []
    for g in games:
        h = list(g); rng.shuffle(h)
        shuf += [(main_block(a), main_block(b)) for a, b in zip(h, h[1:])]
    mi, mi0 = mutual_info(pairs), mutual_info(shuf)
    same = np.mean([a == b for a, b in pairs])
    same0 = np.mean([a == b for a, b in shuf])
    print(f"\nIDEA 1 - SEQUENCE of each wave's main block ({len(pairs)} wave pairs):")
    print(f"   information the previous block gives about the next: {mi:.4f} bits (no-order baseline {mi0:.4f}; max possible ~3.17)")
    print(f"   next main block = same as previous: {100 * same:.1f}% (no-order baseline {100 * same0:.1f}%)")
    T = np.zeros((9, 9))
    for a, b in pairs:
        T[a - 1, b - 1] += 1
    T = T / T.sum(1, keepdims=True).clip(1)
    print("   transition table: row = this wave's main block, column = next wave's (% of rows)")
    print("        " + " ".join(f"  b{j + 1}" for j in range(9)))
    for i in range(9):
        print(f"   b{i + 1}  " + " ".join(f"{100 * T[i, j]:4.0f}" for j in range(9)))
    top = sorted(((T[i, j], i + 1, j + 1) for i in range(9) for j in range(9)), reverse=True)[:3]
    print("   strongest transitions: " + ", ".join(f"b{a}->b{b} {100 * p:.0f}%" for p, a, b in top) + "  (random ~11%)")

    # idea 2: memory - is a block more / less likely again?
    def memory(gs, lag):
        used, unused = [0, 0], [0, 0]
        for g in gs:
            for k in range(lag, len(g)):
                recent = set().union(*[set(g[k - j]) for j in range(1, lag + 1)])
                nxt = set(g[k])
                for bb in range(1, 10):
                    tgt = used if bb in recent else unused
                    tgt[0] += bb in nxt; tgt[1] += 1
        return used[0] / max(used[1], 1), unused[0] / max(unused[1], 1)

    shuffled = []
    for g in games:
        h = list(g); rng.shuffle(h); shuffled.append(h)
    print("\nIDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)")
    rows = []
    for lag_name, lag in (("used in the previous wave", 1), ("used in any of the last 3 waves", 3)):
        pu, pn = memory(games, lag)
        su, sn = memory(shuffled, lag)
        rows.append((pu - pn) - (su - sn))
        print(f"   block {lag_name:<33}: {100 * pu:5.1f}% vs not used {100 * pn:5.1f}%   | shuffled: {100 * su:5.1f}% vs {100 * sn:5.1f}%")
    print("   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle")

    verdict = "NO usable pattern" if (mi - mi0) < 0.02 and max(abs(x) for x in rows) < 0.03 else "POSSIBLE pattern - worth a closer look"
    if len(games) < 100:
        verdict += f" (only {len(games)} games: too few to trust)"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
