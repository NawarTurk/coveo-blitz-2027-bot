#!/usr/bin/env python3
"""
CNN4 dataset: "where will the dinos be when the meteor lands?"

  input  = the map at tick t, including the meteor launched at t (same 16 layers as the 1-step CNN)
  label  = for every tile, what stands there at tick t + meteorDelay (the impact tick):
           0 = no dino, 1..4 = Stegosaurus / Velociraptor / Triceratops / Tyrannosaurus
          -1 = ignore (a dino that spawned after t: impossible to predict)

Usage:   python build_cnn4_dataset.py --logs logs
Writes:  data/cnn4/dataset_X_cnn4.npy   (N, C, H, W) uint8
         data/cnn4/dataset_y_cnn4.npy   (N, H, W)    int8
         data/cnn4/dataset_meta_cnn4.npz
"""
import argparse, glob, json, os
from multiprocessing import Pool

import numpy as np

from state_encoder import encode_state, footprint_from, CHANNELS, SPECIES, MIN_TICKS

OUT_CLASSES = ["none"] + SPECIES          # label 0 = empty tile, 1..4 = species
SP_ID = {s: i + 1 for i, s in enumerate(SPECIES)}
C = len(CHANNELS)


def process_game(args):
    path, H, W = args
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except Exception:
        return path, None
    if len(recs) < MIN_TICKS:
        return path, None
    D = recs[0]["state"]["constants"]["meteorDelay"]
    fp = footprint_from(recs)
    by_tick = {r["state"]["currentTick"]: r for r in recs}
    Xs, ys, ticks, scals, launched = [], [], [], [], []
    prev = {}
    for r in recs:
        s = r["state"]
        t = s["currentTick"]
        pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        fut, nxt = by_tick.get(t + D), by_tick.get(t + 1)
        if fut is None or nxt is None or not s["dinosaurs"]:
            prev = pos
            continue
        # keep the launch only if the server accepted it (it shows up next tick with turnsUntilImpact == D)
        acts = r.get("actions") or []
        ok = False
        if acts and acts[0].get("type") == "LAUNCH_METEOR":
            tgt = (acts[0]["target"]["x"], acts[0]["target"]["y"])
            ok = any((m["target"]["x"], m["target"]["y"]) == tgt and m["turnsUntilImpact"] == D
                     for m in nxt["state"]["meteors"])
        X, scal = encode_state(s, acts if ok else None, prev, fp, H, W)

        ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
        y = np.zeros((H, W), dtype=np.int8)                  # default: no dino at impact
        for d in fut["state"]["dinosaurs"]:
            i, j = d["position"]["y"] - oy, d["position"]["x"] - ox
            if not (0 <= i < H and 0 <= j < W):
                continue
            if d["id"] not in pos:
                y[i, j] = -1                                  # spawned after t: ignore
            elif d["name"] in SP_ID:
                y[i, j] = SP_ID[d["name"]]
        Xs.append(X); ys.append(y); ticks.append(t); scals.append(scal); launched.append(ok)
        prev = pos
    if not Xs:
        return path, None
    return path, (np.stack(Xs), np.stack(ys), np.array(ticks, np.int16), np.array(scals, np.uint8),
                  np.array(launched, bool), D, recs[0]["state"]["constants"]["mapShiftInterval"],
                  recs[0]["state"]["constants"]["meteorRadius"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--out", default="data/cnn4")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    if not files:
        print(f"No game_*.jsonl files in {args.logs}")
        return
    first = json.loads(open(files[0]).readline())["state"]["map"]
    H, W = first["height"], first["width"]
    total = sum(sum(1 for _ in open(f)) for f in files)
    print(f"{len(files)} games, map {H}x{W}, up to {total:,} samples")

    os.makedirs(args.out, exist_ok=True)
    X_path, y_path = f"{args.out}/dataset_X_cnn4.npy", f"{args.out}/dataset_y_cnn4.npy"
    Xmm = np.lib.format.open_memmap(X_path + ".tmp", mode="w+", dtype=np.uint8, shape=(total, C, H, W))
    ymm = np.lib.format.open_memmap(y_path + ".tmp", mode="w+", dtype=np.int8, shape=(total, H, W))
    n, games, gidx, ticks, scals, launched = 0, [], [], [], [], []
    D = shift = R = None
    with Pool(args.workers) as pool:
        for path, res in pool.imap(process_game, [(f, H, W) for f in files]):
            name = os.path.splitext(os.path.basename(path))[0]
            if res is None:
                print(f"skip {name}")
                continue
            X, y, t, sc, la, D, shift, R = res
            k = len(X)
            Xmm[n:n + k], ymm[n:n + k] = X, y
            gidx.append(np.full(k, len(games), np.int32)); ticks.append(t); scals.append(sc); launched.append(la)
            games.append(name)
            n += k
            print(f"done {name}: {k} samples ({la.mean():.0%} with an accepted launch)")
    Xmm.flush(); ymm.flush(); del Xmm, ymm
    for tmp, final, tail, dt in [(X_path + ".tmp", X_path, (C, H, W), np.uint8), (y_path + ".tmp", y_path, (H, W), np.int8)]:
        src = np.load(tmp, mmap_mode="r")
        dst = np.lib.format.open_memmap(final, mode="w+", dtype=dt, shape=(n, *tail))
        for a in range(0, n, 20000):
            b = min(a + 20000, n)
            dst[a:b] = src[a:b]
        dst.flush(); del dst, src
        os.remove(tmp)
    np.savez(f"{args.out}/dataset_meta_cnn4.npz", game_idx=np.concatenate(gidx), games=np.array(games),
             tick=np.concatenate(ticks), scalars=np.concatenate(scals), launched=np.concatenate(launched),
             channels=np.array(CHANNELS), classes=np.array(OUT_CLASSES), meteor_delay=D, shift_interval=shift,
             meteor_radius=R)
    y = np.load(y_path, mmap_mode="r")
    vals, cnts = np.unique(y[: min(n, 50000)], return_counts=True)
    print(f"\nSaved {n:,} samples from {len(games)} games -> {args.out}/")
    print("label mix (first 50k samples):", {(OUT_CLASSES[v] if v >= 0 else 'ignore'): int(c) for v, c in zip(vals, cnts)})
    print(f"disk: X {os.path.getsize(X_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
