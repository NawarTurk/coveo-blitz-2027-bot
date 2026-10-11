#!/usr/bin/env python3
"""
MAPS AND SPAWNS: how many different maps / biomes are there, and do spawns depend on the map?

STEP 1 - maps
  Each game's terrain (everything the window showed: height, impassable, mountain; lava is left out because
  volcanoes change it during the game) gives:
    a fingerprint of its first 20 rows  -> identical fingerprints = the same map played again
    terrain numbers (share of impassable / mountain tiles, average height, height spread)
  Games are grouped into terrain types with k-means for k = 2..5 (the "spread" column shows how tight the
  groups are; a big drop that then flattens = the natural number of groups). The 3-group split is used next,
  since the server shows 3 biomes (deep_crevasse, mountain_range, volcanic_plains) - name each group from its
  numbers (many impassable = crevasse, many mountains = mountain range).

STEP 2 - spawns per terrain group
  For every spawn: which block of the window (4 blocks), which row band, its height percentile among the free
  tiles, and its distance to the nearest impassable / mountain tile - each vs. the free tiles it could have
  spawned on (chance).
  If maps repeat: do the same map's games spawn dinos on the same tiles?

Usage (repo root):
  python analysis/maps_and_spawns.py --logs logs/local_game_logs
Writes results/analysis/local/maps_and_spawns.txt
"""
import argparse, glob, hashlib, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
BLOCKS = ["top-left", "top-right", "bottom-left", "bottom-right"]
NB = [(a, b) for a in range(-3, 4) for b in range(-3, 4) if 0 < abs(a) + abs(b) <= 3]


def game(path):
    S = []
    for l in open(path, "rb"):
        if l.strip():
            try:
                S.append(json.loads(l)["state"])
            except Exception:
                pass
    if len(S) < 100:
        return None
    S.sort(key=lambda s: s["currentTick"])
    terr = {}                                              # world tile -> (elev, impassable, mountain)
    spawns, seen = [], set()
    for s in S:
        m = s["map"]
        ox, oy, h, w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        for i, row in enumerate(m["tiles"]):
            for j, t in enumerate(row):
                p = (j + ox, i + oy)
                if p not in terr:
                    terr[p] = (t.get("elevation", 0), int(t["isImpassable"]), int(t["isMountain"]))
        new = [d for d in s["dinosaurs"] if d["id"] not in seen]
        seen.update(d["id"] for d in s["dinosaurs"])
        if not new:
            continue
        occ = {(d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"] if d not in new}
        free = [(j + ox, i + oy) for i, row in enumerate(m["tiles"]) for j, t in enumerate(row)
                if not (t["isImpassable"] or t["isMountain"] or t["hasLava"]) and (j + ox, i + oy) not in occ]
        if not free:
            continue
        elev = {p: terr[p][0] for p in free}
        vals = sorted(elev.values())
        for d in new:
            p = (d["position"]["x"], d["position"]["y"])
            if p not in elev:
                continue
            spawns.append(dict(t=s["currentTick"], p=p, sp=SP.get(d["name"], "?"), row=p[1] - oy, col=p[0] - ox,
                               h=h, w=w, ox=ox, oy=oy, pct=(sum(1 for v in vals if v < elev[p])
                                                            + 0.5 * sum(1 for v in vals if v == elev[p])) / len(vals),
                               free=free))
    # terrain numbers + fingerprint of the first 20 rows
    rows = sorted({p[1] for p in terr})[:20]
    first = sorted((p, terr[p]) for p in terr if p[1] in rows)
    fp = hashlib.md5(json.dumps(first).encode()).hexdigest()[:12]
    T = np.array(list(terr.values()), float)
    feats = [T[:, 1].mean(), T[:, 2].mean(), T[:, 0].mean(), T[:, 0].std()]
    # spawn detail vs chance (computed here so the big 'free' lists never leave the worker)
    blocked = {p for p, v in terr.items() if v[1] or v[2]}

    def dist_block(p):
        for k in range(1, 4):
            if any((p[0] + a, p[1] + b) in blocked for a, b in NB if abs(a) + abs(b) == k):
                return k
        return 4

    out = []
    for sp in spawns:
        f = sp["free"]
        blk = BLOCKS[(2 if sp["row"] >= sp["h"] // 2 else 0) + (1 if sp["col"] >= sp["w"] // 2 else 0)]
        share = Counter(BLOCKS[(2 if (q[1] - sp["oy"]) >= sp["h"] // 2 else 0) + (1 if (q[0] - sp["ox"]) >= sp["w"] // 2 else 0)]
                        for q in f)
        dchance = Counter(dist_block(q) for q in f[::3])            # sampled for speed
        out.append(dict(t=sp["t"], p=sp["p"], sp=sp["sp"], row=sp["row"], blk=blk, pct=sp["pct"],
                        dist=dist_block(sp["p"]),
                        blk_share={b: share[b] / len(f) for b in BLOCKS},
                        dist_share={k: dchance[k] / max(sum(dchance.values()), 1) for k in range(1, 5)}))
    return dict(name=os.path.basename(path), fp=fp, feats=feats, spawns=out)


def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    best = None
    for rep in range(10):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C, C2):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/local_game_logs")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.logs, "*.jsonl")))
    if not files:
        sys.exit(f"no .jsonl games in {a.logs}")
    with Pool(a.workers) as pool:
        G = [g for g in pool.imap_unordered(game, files, chunksize=4) if g]
    out = [f"MAPS AND SPAWNS | {len(G)} games from {a.logs}"]

    # ---- step 1: maps
    fps = Counter(g["fp"] for g in G)
    rep = [c for c in fps.values() if c > 1]
    out.append("\nSTEP 1 - MAPS")
    out.append(f"  different maps (first 20 rows): {len(fps)} for {len(G)} games | maps played more than once: "
               f"{len(rep)} (covering {sum(rep)} games)")
    X = np.array([g["feats"] for g in G])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    out.append("  terrain groups (k-means): k -> spread (lower = tighter groups)")
    for k in range(2, 6):
        out.append(f"    k={k}: {kmeans(Z, k)[0]:.0f}")
    _, lab, C = kmeans(Z, a.groups)
    Craw = C * sd + mu
    out.append(f"  {a.groups} groups (name them from the numbers):")
    for j in range(a.groups):
        n = int((lab == j).sum())
        out.append(f"    group {j}: {n:4d} games ({n / len(G):.0%}) | impassable {Craw[j][0]:.1%} | mountain "
                   f"{Craw[j][1]:.1%} | avg height {Craw[j][2]:.1f} | height spread {Craw[j][3]:.1f}")

    # ---- step 2: spawns per group
    out.append("\nSTEP 2 - SPAWNS PER TERRAIN GROUP (spawn vs chance = the free tiles it could have spawned on)")
    for j in range(a.groups):
        sps = [s for g, l in zip(G, lab) if l == j for s in g["spawns"]]
        if not sps:
            continue
        n = len(sps)
        out.append(f"  group {j}: {n} spawns | height percentile {np.mean([s['pct'] for s in sps]):.2f} (0.50 = none)")
        out.append("    block:          " + " | ".join(
            f"{b} {sum(s['blk'] == b for s in sps) / n:5.1%} (chance {np.mean([s['blk_share'][b] for s in sps]):5.1%})"
            for b in BLOCKS))
        rb = Counter(min(s["row"] // 5, 3) for s in sps)
        out.append("    row band:       " + " | ".join(f"rows {k * 5}-{k * 5 + 4} {rb[k] / n:5.1%}" for k in range(4))
                   + "   (chance ~ share of free tiles per band)")
        out.append("    tiles to nearest impassable/mountain: " + " | ".join(
            f"{k if k < 4 else '4+'}: {sum(s['dist'] == k for s in sps) / n:5.1%} "
            f"(chance {np.mean([s['dist_share'][k] for s in sps]):5.1%})" for k in range(1, 5)))
        spc = Counter(s["sp"] for s in sps)
        out.append("    species: " + ", ".join(f"{k} {v / n:.0%}" for k, v in spc.most_common()))

    # ---- same map, same spawn tiles?
    if rep:
        out.append("\nREPEATED MAPS - do the same map's games spawn on the same tiles?")
        same, tot = 0, 0
        by_fp = defaultdict(list)
        for g in G:
            by_fp[g["fp"]].append(g)
        for fp, gs in by_fp.items():
            if len(gs) < 2:
                continue
            first = {(s["t"], s["p"]) for s in gs[0]["spawns"]}
            tiles0 = {s["p"] for s in gs[0]["spawns"]}
            for g in gs[1:]:
                for s in g["spawns"]:
                    tot += 1
                    same += s["p"] in tiles0
        out.append(f"  spawns on a tile where the first game of that map also had a spawn: {same / max(tot, 1):.1%} "
                   f"(n={tot}) - compare with how many tiles a game's spawns cover")

    txt = "\n".join(out)
    print(txt)
    path = os.path.join("results", "analysis", a.tag, "maps_and_spawns.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(txt + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
