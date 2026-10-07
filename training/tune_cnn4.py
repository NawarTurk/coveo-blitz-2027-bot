#!/usr/bin/env python3
"""
CNN4 hyper-parameter tuning with 4-fold cross-validation (folds = whole games, no leakage).

Split: same seed-42 game shuffle as train_cnn4.py.
  - the last 20% of games = TEST: never touched here (train_cnn4.py tests on exactly these games)
  - the other 80% are split into 4 folds; each config trains on 3 folds, is scored on the 4th
Each fold-run is short (TUNE_EPOCHS epochs of at most MAX_TRAIN random samples) so the grid fits in a few hours.
Score per config = mean over folds of:
  corr  : correlation of predicted vs actual kills inside the blast of launched meteors (what picks spots)  <- main
  mae   : mean abs error of predicted kills per launch (calibration)
  loss  : cross-entropy on all tiles
  ms    : CPU speed, 1 thread, per spot (server cost: bigger = fewer spots per tick)
Configs slower than MAX_MS_RATIO x the w32/b4 baseline are not eligible (they would score too few spots on the server).
Then (--final) the best config is trained for real with train_cnn4.py on the normal train/dev split and tested.

Usage:
    python tune_cnn4.py --data data/cnn4_v2                       # grid + CV, results in models/cnn4_tune/results.csv
    python tune_cnn4.py --data data/cnn4_v2 --final               # ... then train the winner -> models/cnn4_v2/moves_cnn4.pt
Resumable: finished (config, fold) rows in results.csv are skipped.
"""
import argparse, copy, csv, itertools, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/cnn4_v2")
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--widths", default="32,48")
ap.add_argument("--blocks", default="4,6")
ap.add_argument("--lrs", default="2e-3,1e-3")
ap.add_argument("--flips", default="0,1", help="left-right mirror augmentation off/on")
ap.add_argument("--tune-epochs", type=int, default=4)
ap.add_argument("--max-train", type=int, default=150_000, help="training samples per epoch per fold-run")
ap.add_argument("--max-val", type=int, default=30_000, help="validation samples per fold (launched ones first)")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--max-ms-ratio", type=float, default=1.6)
ap.add_argument("--outdir", default="models/cnn4_tune")
ap.add_argument("--final", action="store_true", help="train the winner with train_cnn4.py afterwards")
ap.add_argument("--final-out", default="models/cnn4_v2/moves_cnn4.pt")
args = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

# ---------------- data (same as train_cnn4.py)
X = np.load(f"{args.data}/dataset_X_cnn4.npy", mmap_mode="r")
Y = np.load(f"{args.data}/dataset_y_cnn4.npy", mmap_mode="r")
meta = np.load(f"{args.data}/dataset_meta_cnn4.npz")
CHANNELS, CLASSES = list(meta["channels"]), list(meta["classes"])
SPECIES = CLASSES[1:]
CH = {n: i for i, n in enumerate(CHANNELS)}
N, C, H, W = X.shape
delay, shift, R = int(meta["meteor_delay"]), int(meta["shift_interval"]), int(meta["meteor_radius"])
SCAL = meta["scalars"].astype(np.float32)
LAUNCHED = meta["launched"].astype(bool)
DX = [CH[c] for c in CHANNELS if c.endswith("_dx")]
DY = [CH[c] for c in CHANNELS if c.endswith("_dy")]

games = np.array(sorted(meta["games"]))
np.random.default_rng(42).shuffle(games)
n_tr = int(0.8 * len(games))
cv_games, test_games = games[:n_tr], games[n_tr:]
name_of = meta["games"][meta["game_idx"]]
fold_games = np.array_split(cv_games, args.folds)
fold_idx = [np.nonzero(np.isin(name_of, g))[0] for g in fold_games]
print(f"{N:,} samples | CV games {len(cv_games)} in {args.folds} folds {[len(g) for g in fold_games]} | "
      f"test games {len(test_games)} (held out, not used here)")

all_cv = np.concatenate(fold_idx)
sample = X[np.sort(np.random.default_rng(0).choice(all_cv, min(5000, len(all_cv)), replace=False))]
scale = np.ones(C, np.float32)
scale[CH["elevation"]] = max(1.0, float(sample[:, CH["elevation"]].max()))
scale[CH["age"]] = 100.0


def prep(xb, sc):
    xb = xb / torch.from_numpy(scale).to(xb.device).view(1, -1, 1, 1)
    d = xb[:, CH["danger_turns"]]
    xb[:, CH["danger_turns"]] = torch.where(d > 0, (delay + 1 - d) / delay, torch.zeros_like(d))
    for c in DX + DY:
        v = xb[:, c]
        xb[:, c] = torch.where(v > 0, v - 2, torch.zeros_like(v))
    B = xb.shape[0]
    rows = torch.linspace(0, 1, H, device=xb.device).view(1, 1, H, 1).expand(B, 1, H, W)
    cols = torch.linspace(0, 1, W, device=xb.device).view(1, 1, 1, W).expand(B, 1, H, W)
    tshift = (sc[:, 0] / shift).view(B, 1, 1, 1).expand(B, 1, H, W)
    tmod = (sc[:, 1] / 100).view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([xb, rows, cols, tshift, tmod], 1)


def make_batch(ids):
    ids = np.sort(ids)
    xb = torch.from_numpy(np.ascontiguousarray(X[ids])).to(dev).float()
    yb = torch.from_numpy(np.ascontiguousarray(Y[ids])).to(dev).long()
    sc = torch.from_numpy(SCAL[ids]).to(dev)
    return prep(xb, sc), yb, ids


def flip_lr(xb, yb):
    xb = xb.flip(-1)
    xb[:, DX] = -xb[:, DX]
    xb[:, C + 1] = xb[:, C + 1].flip(-1)
    return xb, yb.flip(-1)


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


def cpu_ms_per_spot(width, blocks):
    torch.set_num_threads(1)
    net = DinoCNN(C + 4, width, blocks, len(CLASSES)).eval()
    x = torch.randn(16, C + 4, H, W)
    with torch.inference_mode():
        for _ in range(3):
            net(x)
        t0 = time.perf_counter()
        for _ in range(5):
            net(x)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return 1000 * (time.perf_counter() - t0) / (5 * 16)


k = 2 * R + 1
ii, jj = np.mgrid[-R:R + 1, -R:R + 1]
sp_ch = [CH[f"sp_{s}"] for s in SPECIES]


def run_fold(cfg, f):
    width, blocks, lr, flip = cfg
    tr_idx = np.concatenate([fold_idx[j] for j in range(args.folds) if j != f])
    va = fold_idx[f]
    rng = np.random.default_rng(1000 + f)
    la = va[LAUNCHED[va]]
    rest = va[~LAUNCHED[va]]
    n_la = min(len(la), args.max_val // 2)
    va_ids = np.concatenate([rng.choice(la, n_la, replace=False),
                             rng.choice(rest, min(len(rest), args.max_val - n_la), replace=False)])
    torch.manual_seed(f)
    model = DinoCNN(C + 4, width, blocks, len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.tune_epochs)
    for ep in range(args.tune_epochs):
        model.train()
        ids = rng.choice(tr_idx, min(args.max_train, len(tr_idx)), replace=False)
        for a in range(0, len(ids), args.batch):
            xb, yb, _ = make_batch(ids[a:a + args.batch])
            if flip and rng.random() < 0.5:
                xb, yb = flip_lr(xb, yb)
            out = model(xb)
            loss = F.cross_entropy(out, yb, ignore_index=-1, reduction="sum") / max(int((yb >= 0).sum()), 1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    tl, tn, pk, tk = 0.0, 0, [], []
    with torch.no_grad():
        for a in range(0, len(va_ids), args.batch):
            xb, yb, ids = make_batch(va_ids[a:a + args.batch])
            out = model(xb)
            tl += float(F.cross_entropy(out, yb, ignore_index=-1, reduction="sum")); tn += int((yb >= 0).sum())
            lam = torch.from_numpy(LAUNCHED[ids]).to(dev)
            if lam.any():
                p_dino = 1 - F.softmax(out, 1)[:, 0]
                m = yb >= 0
                raw = torch.from_numpy(np.ascontiguousarray(X[ids])).to(dev).float()
                blast = raw[:, CH["new_meteor"]] > 0
                pk.append((p_dino * blast * m).sum((1, 2))[lam].cpu().numpy())
                tk.append(((yb > 0) & blast).sum((1, 2))[lam].float().cpu().numpy())
    pk, tk = np.concatenate(pk), np.concatenate(tk)
    corr = float(np.corrcoef(pk, tk)[0, 1]) if tk.std() > 0 else float("nan")
    return dict(loss=tl / max(tn, 1), corr=corr, mae=float(np.abs(pk - tk).mean()),
                pred=float(pk.mean()), actual=float(tk.mean()), n_launch=len(pk))


# ---------------- grid
grid = list(itertools.product([int(x) for x in args.widths.split(",")], [int(x) for x in args.blocks.split(",")],
                              [float(x) for x in args.lrs.split(",")], [int(x) for x in args.flips.split(",")]))
os.makedirs(args.outdir, exist_ok=True)
res_path = os.path.join(args.outdir, "results.csv")
done = {}
if os.path.exists(res_path):
    for r in csv.DictReader(open(res_path)):
        done[(int(r["width"]), int(r["blocks"]), float(r["lr"]), int(r["flip"]), int(r["fold"]))] = r
fields = ["width", "blocks", "lr", "flip", "fold", "loss", "corr", "mae", "pred", "actual", "n_launch", "secs"]
new_file = not os.path.exists(res_path)
fh = open(res_path, "a", newline="")
wr = csv.DictWriter(fh, fieldnames=fields)
if new_file:
    wr.writeheader()

speed = {}
base_ms = cpu_ms_per_spot(32, 4)
for w, b in {(c[0], c[1]) for c in grid}:
    speed[(w, b)] = cpu_ms_per_spot(w, b)
print(f"grid: {len(grid)} configs x {args.folds} folds = {len(grid) * args.folds} short runs "
      f"({args.tune_epochs} epochs x {args.max_train:,} samples each)")
print("CPU ms/spot (1 thread, this machine):", {f"w{w}b{b}": round(v, 2) for (w, b), v in sorted(speed.items())})

t_all = time.time()
for ci, cfg in enumerate(grid):
    for f in range(args.folds):
        key = cfg + (f,)
        if key in done:
            continue
        t0 = time.time()
        r = run_fold(cfg, f)
        r.update(width=cfg[0], blocks=cfg[1], lr=cfg[2], flip=cfg[3], fold=f, secs=round(time.time() - t0))
        wr.writerow(r); fh.flush()
        done[key] = r
        print(f"[{ci + 1}/{len(grid)}] w{cfg[0]} b{cfg[1]} lr{cfg[2]:g} flip{cfg[3]} fold{f}: corr {r['corr']:.3f} "
              f"mae {r['mae']:.3f} loss {r['loss']:.4f} pred {r['pred']:.3f}/act {r['actual']:.3f} ({r['secs']}s, "
              f"total {(time.time() - t_all) / 60:.0f} min)", flush=True)
fh.close()

# ---------------- summary
rows = []
for cfg in grid:
    rs = [done[cfg + (f,)] for f in range(args.folds) if cfg + (f,) in done]
    if len(rs) < args.folds:
        continue
    g = lambda k_: np.array([float(r[k_]) for r in rs])
    ms = speed[(cfg[0], cfg[1])]
    rows.append(dict(cfg=cfg, corr=g("corr").mean(), corr_sd=g("corr").std(), mae=g("mae").mean(), loss=g("loss").mean(),
                     ms=ms, ok=ms <= args.max_ms_ratio * base_ms))
rows.sort(key=lambda r: -r["corr"])
print(f"\n{'config':<26}{'corr':>14}{'mae':>8}{'loss':>9}{'ms/spot':>9}  eligible")
for r in rows:
    w, b, lr, fl = r["cfg"]
    print(f"w{w} b{b} lr{lr:g} flip{fl:<10}{r['corr']:>8.3f} ±{r['corr_sd']:.3f}{r['mae']:>8.3f}{r['loss']:>9.4f}{r['ms']:>9.2f}  "
          f"{'yes' if r['ok'] else 'too slow'}")
elig = [r for r in rows if r["ok"]]
if not elig:
    sys.exit("no eligible config")
best = elig[0]
w, b, lr, fl = best["cfg"]
base = next((r for r in rows if r["cfg"][:2] == (32, 4) and r["cfg"][2] == 2e-3 and r["cfg"][3] == 0), None)
print(f"\nBEST: width {w}, blocks {b}, lr {lr:g}, flip {fl}  (CV corr {best['corr']:.3f}"
      + (f" vs current settings {base['corr']:.3f})" if base else ")"))
cmd = [sys.executable, "train_cnn4.py", "--data", args.data, "--width", str(w), "--blocks", str(b), "--lr", str(lr),
       "--out", args.final_out] + (["--flip"] if fl else [])
print("final training command:\n  " + " ".join(cmd))
if args.final:
    subprocess.run(cmd, check=True)
