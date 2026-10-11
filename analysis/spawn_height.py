#!/usr/bin/env python3
"""
SPAWN HEIGHT: do dinos spawn on high or low tiles more than chance?

For every spawn (a dino id seen for the first time) it compares the height of the spawn tile with the free
tiles it could have spawned on (same window, same tick; free = not impassable / mountain / lava / occupied).
  1. above vs below the window's average height     (chance = share of free tiles above / below average)
  2. the same per species
  3. the same per block: window split into 4 blocks (top-left, top-right, bottom-left, bottom-right),
     plus how often each block gets the spawn vs its share of free tiles
Also: average height percentile of spawn tiles (0.50 = no preference, 1.00 = always the highest tile).

Usage (repo root):
  python analysis/spawn_height.py --logs logs/local_game_logs
  python analysis/spawn_height.py --logs logs/server_jsonl --tag server
Writes results/analysis/<tag>/spawn_height.txt
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
NAME = {"S": "Stegosaurus", "V": "Velociraptor", "T": "Triceratops", "R": "T-Rex"}
BLOCKS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def block(row, col, h, w):
    return BLOCKS[(2 if row >= h // 2 else 0) + (1 if col >= w // 2 else 0)]


def game(path):
    S = []
    for l in open(path, "rb"):
        if l.strip():
            try:
                S.append(json.loads(l)["state"])
            except Exception:
                pass
    S.sort(key=lambda s: s["currentTick"])
    C = defaultdict(Counter)
    seen = set()
    for s in S:
        new = [d for d in s["dinosaurs"] if d["id"] not in seen]
        seen.update(d["id"] for d in s["dinosaurs"])
        if not new:
            continue
        m = s["map"]
        ox, oy, h, w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        occ = {(d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"] if d not in new}
        free = {}
        for i, row in enumerate(m["tiles"]):
            for j, t in enumerate(row):
                p = (j + ox, i + oy)
                if not (t["isImpassable"] or t["isMountain"] or t["hasLava"]) and p not in occ:
                    free[p] = t.get("elevation", 0)
        if not free:
            continue
        avg = sum(free.values()) / len(free)
        vals = sorted(free.values())
        blk_tiles = defaultdict(list)
        for p, e in free.items():
            blk_tiles[block(p[1] - oy, p[0] - ox, h, w)].append(e)
        blk_avg = {b: sum(v) / len(v) for b, v in blk_tiles.items()}
        # chance baselines for this tick
        above_share = sum(1 for e in free.values() if e > avg) / len(free)
        below_share = sum(1 for e in free.values() if e < avg) / len(free)
        for d in new:
            p = (d["position"]["x"], d["position"]["y"])
            if p not in free:
                continue
            sp = SP.get(d["name"], "?")
            e = free[p]
            b = block(p[1] - oy, p[0] - ox, h, w)
            pct = (sum(1 for v in vals if v < e) + 0.5 * sum(1 for v in vals if v == e)) / len(vals)
            for key in ("ALL", sp):
                c = C["sp_" + key]
                c["n"] += 1
                c["above"] += e > avg
                c["below"] += e < avg
                c["chance_above"] += above_share
                c["chance_below"] += below_share
                c["pct"] += pct
            # blocks: where it spawned vs share of free tiles, and above/below inside the block
            c = C["blk_" + b]
            c["n"] += 1
            ba = blk_avg[b]
            bt = blk_tiles[b]
            c["above"] += e > ba
            c["below"] += e < ba
            c["chance_above"] += sum(1 for v in bt if v > ba) / len(bt)
            c["chance_below"] += sum(1 for v in bt if v < ba) / len(bt)
            c["pct"] += pct
            for bb in BLOCKS:
                C["blk_share"][bb + "_tiles"] += len(blk_tiles.get(bb, [])) / len(free)
            C["blk_share"]["spawns"] += 1
            C["blk_sp_" + sp][b] += 1
    return {k: dict(v) for k, v in C.items()}


def line(c):
    n = c.get("n", 0)
    if not n:
        return "no spawns"
    return (f"n={n:6d} | above avg {c['above'] / n:5.1%} (chance {c['chance_above'] / n:5.1%}) | "
            f"below avg {c['below'] / n:5.1%} (chance {c['chance_below'] / n:5.1%}) | "
            f"height percentile {c['pct'] / n:.2f} (0.50 = none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/local_game_logs")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.logs, "*.jsonl")))
    if not files:
        sys.exit(f"no .jsonl games in {a.logs}")
    C = defaultdict(Counter)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(game, files, chunksize=4):
            for k, v in r.items():
                C[k].update(v)
    out = [f"SPAWN HEIGHT | {len(files)} games from {a.logs}",
           "Compares each spawn tile's height with the free tiles it could have spawned on (same window, same tick).",
           "\n1. ALL SPAWNS", "   " + line(C["sp_ALL"]), "\n2. PER SPECIES"]
    for sp in "SVTR":
        out.append(f"   {NAME[sp]:12} " + line(C["sp_" + sp]))
    out.append("\n3. PER BLOCK (window split in 4; above/below = vs that block's own average height)")
    sh = C["blk_share"]
    n = max(sh.get("spawns", 0), 1)
    for b in BLOCKS:
        out.append(f"   {b:12} spawns here {C['blk_' + b].get('n', 0) / n:5.1%} (share of free tiles "
                   f"{sh.get(b + '_tiles', 0) / n:5.1%}) | " + line(C["blk_" + b]))
    out.append("\n   spawns per block, per species:")
    for sp in "SVTR":
        c = C["blk_sp_" + sp]
        t = max(sum(c.values()), 1)
        out.append(f"   {NAME[sp]:12} " + " | ".join(f"{b} {c.get(b, 0) / t:5.1%}" for b in BLOCKS))
    txt = "\n".join(out)
    print(txt)
    path = os.path.join("results", "analysis", a.tag, "spawn_height.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(txt + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
