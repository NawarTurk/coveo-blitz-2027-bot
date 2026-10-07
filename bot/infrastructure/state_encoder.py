#!/usr/bin/env python3
"""
CNN dataset: one sample per TICK = the whole map as a stack of 20x20 layers,
label = every dino's next move on its tile (-1 = no dino / ignore).

Usage:
    python state_encoder.py --logs logs
Writes (memory-mapped .npy, loaded lazily by train_cnn1.py):
    data/dataset_X_cnn.npy     (N, C, H, W) uint8   map layers (raw values, normalized at train time)
    data/dataset_y_cnn.npy     (N, H, W)    int8    move class per tile, -1 = ignore
    data/dataset_meta_cnn.npz  game index, tick, ticks_to_shift, tick_mod100, channel names, classes...
"""
import argparse, glob, json, os
from multiprocessing import Pool

import numpy as np

SPECIES = ["Stegosaurus", "Velociraptor", "Triceratops", "Tyrannosaurus"]
CLASSES = ["-1,0", "0,-1", "0,0", "0,1", "1,0"]                 # same order as the XGBoost model
MOVE2CLS = {tuple(map(int, c.split(","))): i for i, c in enumerate(CLASSES)}
CHANNELS = (["inside", "elevation", "impassable", "mountain", "lava", "corpse",
             "danger_turns",      # min turnsUntilImpact of meteors covering the tile (0 = none)
             "new_meteor",        # tile is in the blast of the meteor WE launch this tick
             "volcano"]
            + [f"sp_{s}" for s in SPECIES]
            + ["age",             # dino age on its tile (capped 255)
               "prev_dx", "prev_dy"])   # last move + 2 (1,2,3); 0 = unknown / no dino
C = len(CHANNELS)
CH = {n: i for i, n in enumerate(CHANNELS)}
MIN_TICKS = 900


def footprint_from(recs):
    fp = set()
    for r in recs:
        for me in r["state"]["meteors"]:
            tx, ty = me["target"]["x"], me["target"]["y"]
            offs = {(p["x"] - tx, p["y"] - ty) for p in me["impactedTiles"]}
            if len(offs) > len(fp):
                fp = offs
    return fp


def encode_state(s, actions, prev_pos, footprint, H=None, W=None):
    """Game state dict -> (C,H,W) uint8 array + (ticks_to_shift, tick_mod100). Shared with the bot."""
    m = s["map"]
    ox, oy = m["origin"]["x"], m["origin"]["y"]
    h, w = m["height"], m["width"]
    H, W = H or h, W or w
    X = np.zeros((C, H, W), dtype=np.uint8)
    X[CH["inside"], :h, :w] = 1
    tiles = m["tiles"]
    X[CH["elevation"], :h, :w] = np.clip([[t["elevation"] for t in r] for r in tiles], 0, 255)
    X[CH["impassable"], :h, :w] = [[t["isImpassable"] for t in r] for r in tiles]
    X[CH["mountain"], :h, :w] = [[t["isMountain"] for t in r] for r in tiles]
    X[CH["lava"], :h, :w] = [[t["hasLava"] for t in r] for r in tiles]

    def put(ch, x, y, v=1):
        i, j = y - oy, x - ox
        if 0 <= i < h and 0 <= j < w:
            X[ch, i, j] = v
            return True
        return False

    for p in s["corpses"]:
        put(CH["corpse"], p["x"], p["y"])
    for v in s["volcanoes"]:
        put(CH["volcano"], v["position"]["x"], v["position"]["y"])
    for me in s["meteors"]:
        t = me["turnsUntilImpact"]
        for p in me["impactedTiles"]:
            i, j = p["y"] - oy, p["x"] - ox
            if 0 <= i < h and 0 <= j < w:
                cur = X[CH["danger_turns"], i, j]
                X[CH["danger_turns"], i, j] = t if cur == 0 else min(cur, t)
    if actions and actions[0].get("type") == "LAUNCH_METEOR":
        tx, ty = actions[0]["target"]["x"], actions[0]["target"]["y"]
        for dx, dy in footprint:
            put(CH["new_meteor"], tx + dx, ty + dy)
    for d in s["dinosaurs"]:
        if d["name"] not in SPECIES:
            continue
        x, y = d["position"]["x"], d["position"]["y"]
        if not put(CH[f"sp_{d['name']}"], x, y):
            continue
        put(CH["age"], x, y, min(d["age"], 255))
        pp = prev_pos.get(d["id"]) if prev_pos else None
        if pp is not None:
            put(CH["prev_dx"], x, y, int(np.clip(x - pp[0], -1, 1)) + 2)
            put(CH["prev_dy"], x, y, int(np.clip(y - pp[1], -1, 1)) + 2)

    c = s["constants"]
    tick = s["currentTick"]
    scal = ((c["mapShiftInterval"] - tick % c["mapShiftInterval"]) % c["mapShiftInterval"], tick % 100)
    return X, scal


def process_game(args):
    path, H, W = args
    with open(path) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    if len(recs) < MIN_TICKS:
        return path, None
    fp = footprint_from(recs)
    Xs, ys, ticks, scals = [], [], [], []
    prev = {}
    for cur, nxt in zip(recs, recs[1:]):
        s, s2 = cur["state"], nxt["state"]
        pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        if s2["currentTick"] == s["currentTick"] + 1 and s["dinosaurs"]:
            X, scal = encode_state(s, cur.get("actions"), prev, fp, H, W)
            y = np.full((H, W), -1, dtype=np.int8)
            ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
            nxt_pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s2["dinosaurs"]}
            for d in s["dinosaurs"]:
                if d["id"] not in nxt_pos or d["name"] not in SPECIES:
                    continue                                   # died / eaten / scrolled out
                x0, y0 = pos[d["id"]]
                mv = (nxt_pos[d["id"]][0] - x0, nxt_pos[d["id"]][1] - y0)
                i, j = y0 - oy, x0 - ox
                if mv in MOVE2CLS and 0 <= i < H and 0 <= j < W:
                    y[i, j] = MOVE2CLS[mv]
            if (y >= 0).any():
                Xs.append(X); ys.append(y); ticks.append(s["currentTick"]); scals.append(scal)
        prev = pos
    if not Xs:
        return path, None
    delay = recs[0]["state"]["constants"]["meteorDelay"]
    shift = recs[0]["state"]["constants"]["mapShiftInterval"]
    return path, (np.stack(Xs), np.stack(ys), np.array(ticks, np.int16),
                  np.array(scals, np.uint8), delay, shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    if not files:
        print(f"No game_*.jsonl files in {args.logs}")
        return
    # map size (assumed fixed) + upper bound on samples
    first = json.loads(open(files[0]).readline())["state"]["map"]
    H, W = first["height"], first["width"]
    total = sum(sum(1 for _ in open(f)) for f in files)
    print(f"{len(files)} games, map {H}x{W}, up to {total:,} samples, {C} channels")

    os.makedirs(args.out, exist_ok=True)
    X_path, y_path = f"{args.out}/dataset_X_cnn.npy", f"{args.out}/dataset_y_cnn.npy"
    Xmm = np.lib.format.open_memmap(X_path + ".tmp", mode="w+", dtype=np.uint8, shape=(total, C, H, W))
    ymm = np.lib.format.open_memmap(y_path + ".tmp", mode="w+", dtype=np.int8, shape=(total, H, W))

    n, games, game_idx, ticks, scals, delay, shift = 0, [], [], [], [], 4, 5
    with Pool(args.workers) as pool:
        for path, res in pool.imap(process_game, [(f, H, W) for f in files]):
            name = os.path.splitext(os.path.basename(path))[0]
            if res is None:
                print(f"skip {name}")
                continue
            X, y, t, sc, delay, shift = res
            k = len(X)
            Xmm[n:n + k], ymm[n:n + k] = X, y
            game_idx.append(np.full(k, len(games), np.int32)); ticks.append(t); scals.append(sc)
            games.append(name)
            n += k
            print(f"done {name}: {k} ticks")
    Xmm.flush(); ymm.flush()
    del Xmm, ymm

    # trim to the real number of samples
    for tmp, final, shape_tail, dt in [(X_path + ".tmp", X_path, (C, H, W), np.uint8),
                                       (y_path + ".tmp", y_path, (H, W), np.int8)]:
        src = np.load(tmp, mmap_mode="r")
        dst = np.lib.format.open_memmap(final, mode="w+", dtype=dt, shape=(n, *shape_tail))
        for a in range(0, n, 20000):
            b = min(a + 20000, n)
            dst[a:b] = src[a:b]
        dst.flush(); del dst, src
        os.remove(tmp)

    np.savez(f"{args.out}/dataset_meta_cnn.npz",
             game_idx=np.concatenate(game_idx), games=np.array(games), tick=np.concatenate(ticks),
             scalars=np.concatenate(scals), channels=np.array(CHANNELS), classes=np.array(CLASSES),
             species=np.array(SPECIES), meteor_delay=delay, shift_interval=shift)
    y = np.load(y_path, mmap_mode="r")
    lab = y[y >= 0]
    print(f"\nSaved {n:,} map samples ({lab.size:,} labelled dino tiles) from {len(games)} games -> "
          f"{X_path}, {y_path}, {args.out}/dataset_meta_cnn.npz")
    print("move distribution:", {CLASSES[k]: int(v) for k, v in zip(*np.unique(lab, return_counts=True))})
    print(f"disk: X {os.path.getsize(X_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
