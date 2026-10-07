#!/usr/bin/env python3
"""
One row per ACCEPTED meteor launch in the logs (whatever bot fired it):
  CNN4's prediction for that exact shot (recomputed now with --model, same input as the bot)
  + heuristic facts about the shot + what really happened.

Columns
  game, tick
  cnn_E        CNN4 expected dinos in the blast at impact
  cnn_V        CNN4 expected points in the blast (P x value of dinos that can reach the tile)
  cnn_score    cnn_V x (1 + 0.5 x max(cnn_E - 1, 0))      (Z1's reward without the clear bonus)
  cnn_clear    min(1, cnn_E / n_left) ^ n_left             (Z1's clear-probability proxy)
  n_dinos, n_left, n_cover, n_reach        board size, dinos not already doomed, dinos in blast now, dinos within R+D
  v_cover, v_reach                          points of those dinos (160/(1+age/30) at impact)
  trap_prior   sum over covered dinos of value x measured catch rate for its free escape tiles (Z1's 30% part)
  min_exits, mean_exits                     free escape tiles (within 3 steps, outside the blast) of covered dinos
  blocked_ring fraction of tiles within R+2 of the centre that are wall/mountain/lava/off-map  ("cage strength")
  chain_n      reach dinos standing in a falling meteor that lands before this one
  scroll_n, scroll_v                        reach dinos whose row scrolls away before impact (count, points)
  row_from_back  centre row counted from the trailing (scrolling) edge
  n_steg, n_raptor, n_tri, n_trex           species among reach dinos
  trex_prey    covered non-T-Rex dinos with a T-Rex adjacent
  age_min, age_mean                         of reach dinos
  meteors_falling
  kills, points, any_kill                   OUTCOME: dinos killed by THIS meteor, points (age value x multi-kill)

Usage:
  python build_meteor_table.py --model models/cnn4/moves_cnn4.pt --out data/launch_table_cnn4.csv
  python build_meteor_table.py --model models/cnn4_v2/moves_cnn4.pt --out data/launch_table_cnn4v2.csv
"""
import argparse, csv, glob, json, os, time
from collections import deque
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from state_encoder import encode_state, footprint_from, MIN_TICKS
import meteor_features as MF

def add_history(X, s, hist, K, H, W):
    """extra layers hist{k}_dx/dy (k = 2..K): the move a dino made k ticks ago, +2 (0 = unknown). Same as build_cnn4_dataset"""
    if K <= 1:
        return X
    extra = np.zeros((2 * (K - 1), H, W), np.uint8)
    ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
    for d in s["dinosaurs"]:
        i, j = d["position"]["y"] - oy, d["position"]["x"] - ox
        h = hist.get(d["id"])
        if not (0 <= i < H and 0 <= j < W) or not h:
            continue
        h = list(h)
        for k in range(2, K + 1):
            if len(h) < k + 1:
                break
            (x1, y1), (x0, y0) = h[-k], h[-k - 1]
            extra[2 * (k - 2), i, j] = int(np.clip(x1 - x0, -1, 1)) + 2
            extra[2 * (k - 2) + 1, i, j] = int(np.clip(y1 - y0, -1, 1)) + 2
    return np.concatenate([X, extra], 0)


PRE_TRAP = [(0, 0.79), (3, 0.47), (7, 0.26), (14, 0.10), (999, 0.036)]
FIELDS = ["game", "tick", "cnn_E", "cnn_V"] + MF.BASE + MF.HEUR + ["kills", "points", "any_kill"]


def val(age):
    return 160.0 / (1 + age / 30.0)


def p_trap(ex):
    for hi, p in PRE_TRAP:
        if ex <= hi:
            return p
    return PRE_TRAP[-1][1]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def process_game(args):
    path, H, W, K = args
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except Exception:
        return path, None
    if len(recs) < MIN_TICKS:
        return path, None
    c0 = recs[0]["state"]["constants"]
    D, R, SH = c0["meteorDelay"], c0["meteorRadius"], c0["mapShiftInterval"]
    fp = footprint_from(recs)
    foot = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]
    by_tick = {r["state"]["currentTick"]: r for r in recs}
    game = os.path.basename(path)
    rows, Xs, masks, vgs, scals = [], [], [], [], []
    prev, hist = {}, {}
    for r in recs:
        s = r["state"]
        t = s["currentTick"]
        pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        for i_, p_ in pos.items():
            hist.setdefault(i_, deque(maxlen=K + 1)).append(p_)
        for i_ in [i_ for i_ in hist if i_ not in pos]:
            del hist[i_]
        acts = r.get("actions") or []
        nxt, land_s, after_s = by_tick.get(t + 1), by_tick.get(t + D), by_tick.get(t + D + 1)
        ok = False
        if acts and acts[0].get("type") == "LAUNCH_METEOR" and nxt and land_s and after_s and s["dinosaurs"]:
            tgt = (acts[0]["target"]["x"], acts[0]["target"]["y"])
            ok = any((m["target"]["x"], m["target"]["y"]) == tgt and m["turnsUntilImpact"] == D
                     for m in nxt["state"]["meteors"])
        if not ok:
            prev = pos
            continue
        X, scal = encode_state(s, acts, prev, fp, H, W)
        X = add_history(X, s, hist, K, H, W)
        ctx = MF.FeatureCtx(s)
        c = tgt
        tiles = {(c[0] + a, c[1] + b) for a, b in ctx.foot}
        land = t + D
        row = dict(game=game, tick=t, **MF.features(ctx, c))
        # outcome: dinos standing in the blast at the landing state, gone the state after
        land_d = {d["id"]: ((d["position"]["x"], d["position"]["y"]), d["age"]) for d in land_s["state"]["dinosaurs"]}
        after_ids = {d["id"] for d in after_s["state"]["dinosaurs"]}
        killed = [a for i, (p, a) in land_d.items() if p in tiles and i not in after_ids]
        k = len(killed)
        row.update(kills=k, points=round(sum(MF.val(a) for a in killed) * (1 + 0.5 * max(k - 1, 0)), 1),
                   any_kill=int(k > 0))
        rows.append(row)
        mk = np.zeros((H, W), bool)
        for q in tiles:
            i, j = q[1] - ctx.oy, q[0] - ctx.ox
            if 0 <= i < min(ctx.h, H) and 0 <= j < min(ctx.w, W):
                mk[i, j] = True
        vg = MF.value_grid(ctx, H, W)
        Xs.append(X); masks.append(mk); vgs.append(vg); scals.append(scal)
        prev = pos
    if not rows:
        return path, None
    return path, (rows, np.stack(Xs), np.stack(masks), np.stack(vgs), np.array(scals, np.float32), D, SH)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class DinoCNN(nn.Module):
    def __init__(self, cin, width, blocks, ncls):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, ncls, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--model", default="models/cnn4/moves_cnn4.pt")
    ap.add_argument("--out", default="data/launch_table_cnn4.csv")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--exclude-train-of", default="",
                    help="dataset dir the CNN4 was trained from (e.g. data/cnn4_v2): skip its train+dev games, "
                         "keep only games the model never saw (its test split + any newer games)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    chans = [str(x) for x in ck["channels"]]
    CH = {n: i for i, n in enumerate(chans)}
    K, H, W = int(ck.get("history", 1)), ck["H"], ck["W"]
    net = DinoCNN(len(chans) + 4, ck["width"], ck["blocks"], len(ck["classes"]))
    net.load_state_dict(ck["state_dict"]); net = net.to(dev).eval()
    scale = torch.tensor(np.asarray(ck["scale"], np.float32), device=dev).view(1, -1, 1, 1)
    DX = [CH[x] for x in chans if x.endswith("_dx") or x.endswith("_dy")]
    print(f"device {dev} | model {args.model} (w{ck['width']} b{ck['blocks']}, history {K})")

    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    if args.exclude_train_of:
        meta = glob.glob(os.path.join(args.exclude_train_of, "dataset_meta_*.npz"))[0]
        games = np.array(sorted(np.load(meta)["games"]))
        np.random.default_rng(42).shuffle(games)                 # same split as train_cnn4.py / tune_cnn4.py
        stem = lambda x: os.path.splitext(os.path.basename(str(x)))[0]
        seen = {stem(g) for g in games[:int(0.8 * len(games))]}
        before = len(files)
        files = [f for f in files if stem(f) not in seen]
        print(f"excluding {before - len(files)} games the model trained on -> {len(files)} unseen games")
    if args.max_games:
        files = files[-args.max_games:]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader()
    n_rows, n_games, t0 = 0, 0, time.time()
    with Pool(args.workers) as pool:
        for path, res in pool.imap_unordered(process_game, [(f, H, W, K) for f in files], chunksize=2):
            if res is None:
                continue
            rows, Xs, masks, vgs, scals, D, SH = res
            pd = []
            with torch.no_grad():
                for a in range(0, len(Xs), 256):
                    xb = torch.from_numpy(Xs[a:a + 256]).to(dev).float() / scale
                    d = xb[:, CH["danger_turns"]]
                    xb[:, CH["danger_turns"]] = torch.where(d > 0, (D + 1 - d) / D, torch.zeros_like(d))
                    for cc in DX:
                        v = xb[:, cc]
                        xb[:, cc] = torch.where(v > 0, v - 2, torch.zeros_like(v))
                    B = xb.shape[0]
                    sc = torch.from_numpy(scals[a:a + 256]).to(dev)
                    rr = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, 1, H, W)
                    cl = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, 1, H, W)
                    ts = (sc[:, 0] / SH).view(B, 1, 1, 1).expand(B, 1, H, W)
                    tm = (sc[:, 1] / 100).view(B, 1, 1, 1).expand(B, 1, H, W)
                    p = F.softmax(net(torch.cat([xb, rr, cl, ts, tm], 1)), 1)
                    pd.append((1 - p[:, 0]).cpu().numpy())
            pd = np.concatenate(pd)
            for k_, row in enumerate(rows):
                P = pd[k_][masks[k_]]
                E = float(P.sum())
                V = float((P * vgs[k_][masks[k_]]).sum())
                nl = row["n_left"]
                row.update(cnn_E=round(E, 4), cnn_V=round(V, 2), **MF.cnn_feats(E, V, nl))
                wr.writerow(row)
            n_rows += len(rows); n_games += 1
            if n_games % 50 == 0:
                print(f"  {n_games}/{len(files)} games, {n_rows:,} launches, {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    print(f"wrote {args.out}: {n_rows:,} launches from {n_games} games ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
