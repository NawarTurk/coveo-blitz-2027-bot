#!/usr/bin/env python3
"""
CNN4: map at tick t (+ the meteor launched at t) -> for EVERY tile, what stands there at the
impact tick t + meteorDelay: none / Stegosaurus / Velociraptor / Triceratops / Tyrannosaurus.
One pass per meteor candidate: expected kills = sum of P(dino) over the blast tiles.

Same backbone, input layers and game split (seed 42, 70/10/20) as train_cnn1.py.

Usage:
    python build_cnn4_dataset.py --logs logs     # once
    python train_cnn4.py
Outputs:
    models/cnn4/moves_cnn4.pt          weights + normalization + config (for the bot)
    models/cnn4/loss_curve_cnn4.png    train vs dev loss per epoch
"""
import argparse, copy, os, shutil, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/cnn4")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--blocks", type=int, default=4, help="4 blocks = 19x19 receptive field, enough for 4 moves")
ap.add_argument("--none-weight", type=float, default=1.0,
                help="loss weight of 'no dino' tiles. Keep 1.0 so P(dino) stays calibrated for kill counting")
ap.add_argument("--flip", action="store_true", help="augment with left-right mirroring")
ap.add_argument("--max-train", type=int, default=0, help="cap training samples per epoch (0 = all)")
ap.add_argument("--out", default="models/cnn4/moves_cnn4.pt")
args = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

# ---------------- data (memory-mapped, loaded per batch)
X = np.load(f"{args.data}/dataset_X_cnn4.npy", mmap_mode="r")
Y = np.load(f"{args.data}/dataset_y_cnn4.npy", mmap_mode="r")
meta = np.load(f"{args.data}/dataset_meta_cnn4.npz")
CHANNELS, CLASSES = list(meta["channels"]), list(meta["classes"])     # classes: none + 4 species
SPECIES = CLASSES[1:]
CH = {n: i for i, n in enumerate(CHANNELS)}
N, C, H, W = X.shape
delay, shift, R = int(meta["meteor_delay"]), int(meta["shift_interval"]), int(meta["meteor_radius"])
SCAL = meta["scalars"].astype(np.float32)
LAUNCHED = meta["launched"]
print(f"{N:,} map samples, {C} channels, {H}x{W}, classes {CLASSES}, delay {delay}")

games = np.array(sorted(meta["games"]))
np.random.default_rng(42).shuffle(games)
n_tr, n_dev = int(0.8 * len(games)), max(1, int(0.1 * len(games)))
name_of = meta["games"][meta["game_idx"]]
idx_train = np.nonzero(np.isin(name_of, games[:n_tr - n_dev]))[0]
idx_dev = np.nonzero(np.isin(name_of, games[n_tr - n_dev:n_tr]))[0]
idx_test = np.nonzero(np.isin(name_of, games[n_tr:]))[0]
print(f"games {n_tr - n_dev}/{n_dev}/{len(games) - n_tr} | samples {len(idx_train):,}/{len(idx_dev):,}/{len(idx_test):,} (train/dev/test)")

sample = X[np.sort(np.random.default_rng(0).choice(idx_train, min(5000, len(idx_train)), replace=False))]
scale = np.ones(C, np.float32)
scale[CH["elevation"]] = max(1.0, float(sample[:, CH["elevation"]].max()))
scale[CH["age"]] = 100.0
cls_w = torch.tensor([args.none_weight] + [1.0] * len(SPECIES), device=dev)


def prep(xb, sc):
    """raw uint8 layers (B,C,H,W) + scalars (B,2) -> normalized float input (B, C+4, H, W). Same as train_cnn1.py"""
    xb = xb / torch.from_numpy(scale).to(xb.device).view(1, -1, 1, 1)
    d = xb[:, CH["danger_turns"]]
    xb[:, CH["danger_turns"]] = torch.where(d > 0, (delay + 1 - d) / delay, torch.zeros_like(d))
    for c in ("prev_dx", "prev_dy"):
        v = xb[:, CH[c]]
        xb[:, CH[c]] = torch.where(v > 0, v - 2, torch.zeros_like(v))
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
    """mirror left-right: species labels don't change, only the maps and the x-direction layer"""
    xb = xb.flip(-1)
    xb[:, CH["prev_dx"]] = -xb[:, CH["prev_dx"]]
    xb[:, C + 1] = xb[:, C + 1].flip(-1)
    return xb, yb.flip(-1)


# ---------------- model (same as train_cnn1.py)
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


model = DinoCNN(C + 4, args.width, args.blocks, len(CLASSES)).to(dev)
print(f"model: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k params, "
      f"receptive field {3 + 4 * args.blocks}x{3 + 4 * args.blocks} tiles")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)


def run(ids, train):
    """mean cross-entropy per labelled tile (all tiles except 'ignore')"""
    model.train(train)
    tot_loss, tot_n = 0.0, 0
    order = np.random.permutation(ids) if train else ids
    with torch.set_grad_enabled(train):
        for a in range(0, len(order), args.batch):
            xb, yb, _ = make_batch(order[a:a + args.batch])
            if train and args.flip and np.random.rand() < 0.5:
                xb, yb = flip_lr(xb, yb)
            loss = F.cross_entropy(model(xb), yb, weight=cls_w, ignore_index=-1, reduction="sum")
            n = int((yb >= 0).sum())
            if train:
                opt.zero_grad()
                (loss / max(n, 1)).backward()
                opt.step()
            tot_loss += float(loss.detach()); tot_n += n
    return tot_loss / max(tot_n, 1)


# ---------------- train with early stopping on dev loss
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
if os.path.exists(args.out):
    old = torch.load(args.out, map_location="cpu", weights_only=False)
    backup = os.path.splitext(args.out)[0] + f"_w{old.get('width')}_b{old.get('blocks')}_backup.pt"
    if not os.path.exists(backup):
        shutil.copy(args.out, backup)
        print(f"backed up existing model -> {backup}")
hist_tr, hist_dv, best, bad = [], [], float("inf"), 0
for ep in range(1, args.epochs + 1):
    t0 = time.time()
    tr_ids = idx_train if not args.max_train else np.random.choice(idx_train, min(args.max_train, len(idx_train)), replace=False)
    tr = run(tr_ids, True)
    dv = run(idx_dev, False)
    sched.step(dv)
    hist_tr.append(tr); hist_dv.append(dv)
    mark = ""
    if dv < best - 1e-5:
        best, bad = dv, 0
        torch.save({"state_dict": model.state_dict(), "scale": scale, "channels": CHANNELS, "classes": CLASSES,
                    "species": SPECIES, "width": args.width, "blocks": args.blocks, "H": H, "W": W,
                    "meteor_delay": delay, "shift_interval": shift, "meteor_radius": R,
                    "target": "occupancy at t + meteor_delay"}, args.out)
        mark = "  * saved"
    else:
        bad += 1
    print(f"epoch {ep:2d}  train {tr:.4f}  dev {dv:.4f}  lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t0:.0f}s){mark}", flush=True)
    if bad >= args.patience:
        print(f"early stopping: dev loss hasn't improved for {args.patience} epochs")
        break

# ---------------- loss curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ep_axis = range(1, len(hist_tr) + 1)
ax.plot(ep_axis, hist_tr, "o-", label="train")
ax.plot(ep_axis, hist_dv, "o-", label="dev")
b = int(np.argmin(hist_dv)) + 1
ax.axvline(b, color="gray", ls="--", lw=1, label=f"best = epoch {b}")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (all labelled tiles)"); ax.set_title("CNN4 train vs dev loss")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
curve = os.path.join(os.path.dirname(args.out) or ".", "loss_curve_cnn4.png")
plt.savefig(curve, dpi=120)
print("saved", curve)

# ---------------- test with the best checkpoint
ck = torch.load(args.out, map_location=dev, weights_only=False)
model.load_state_dict(ck["state_dict"]); model.eval()

# blast mask (Manhattan diamond) around the new meteor tile, as a conv kernel
k = 2 * R + 1
ii, jj = np.mgrid[-R:R + 1, -R:R + 1]
diamond = torch.tensor((np.abs(ii) + np.abs(jj) <= R).astype(np.float32), device=dev).view(1, 1, k, k)
sp_ch = [CH[f"sp_{s}"] for s in SPECIES]

tp = fp = fn = 0
sp_ok = sp_n = 0
pred_k, true_k, stay_k = [], [], []
with torch.no_grad():
    for a in range(0, len(idx_test), args.batch):
        xb, yb, ids = make_batch(idx_test[a:a + args.batch])
        prob = F.softmax(model(xb), 1)
        p_dino = 1 - prob[:, 0]
        m = yb >= 0
        occ_pred, occ_true = (p_dino > 0.5) & m, (yb > 0)
        tp += int((occ_pred & occ_true).sum()); fp += int((occ_pred & ~occ_true).sum()); fn += int((~occ_pred & occ_true & m).sum())
        both = occ_pred & occ_true
        sp_ok += int((prob[:, 1:].argmax(1) + 1 == yb)[both].sum()); sp_n += int(both.sum())
        # expected kills for accepted launches: model vs "dinos don't move"
        la = torch.from_numpy(LAUNCHED[ids]).to(dev)
        if la.any():
            raw = torch.from_numpy(np.ascontiguousarray(X[ids])).to(dev).float()
            blast = F.conv2d((raw[:, CH["new_meteor"]] > 0).float().unsqueeze(1), diamond, padding=R).squeeze(1) > 0
            now = raw[:, sp_ch].sum(1) > 0
            pred_k.append((p_dino * blast * m).sum((1, 2))[la].cpu().numpy())
            true_k.append(((yb > 0) & blast).sum((1, 2))[la].float().cpu().numpy())
            stay_k.append((now & blast).sum((1, 2))[la].float().cpu().numpy())

prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
print(f"\nCNN4 TEST  (tiles at impact tick, threshold P(dino) > 0.5)")
print(f"  occupancy precision {prec:.3f}   recall {rec:.3f}   F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}")
print(f"  species correct on found dinos {sp_ok / max(sp_n, 1):.3f}")
if pred_k:
    pk, tk, sk = map(np.concatenate, (pred_k, true_k, stay_k))
    print(f"\nKills inside the blast, {len(pk):,} accepted launches:")
    print(f"  actual average        {tk.mean():.3f}")
    print(f"  CNN4 predicted avg    {pk.mean():.3f}   mean abs error {np.abs(pk - tk).mean():.3f}")
    print(f"  'dinos don't move'    {sk.mean():.3f}   mean abs error {np.abs(sk - tk).mean():.3f}")
    print(f"  correlation CNN4 vs actual {np.corrcoef(pk, tk)[0, 1] if tk.std() > 0 else float('nan'):.3f}")
    print("  calibration (predicted bin -> actual avg):")
    for lo, hi in [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 99)]:
        sel = (pk >= lo) & (pk < hi)
        if sel.any():
            print(f"    {lo:>4}-{hi:<4} n={sel.sum():7,d}  predicted {pk[sel].mean():.2f}  actual {tk[sel].mean():.2f}")
print(f"\nsaved {args.out}  (width {args.width}, {args.blocks} blocks)")

# ---------------- server speed check: one CPU thread, BatchNorm folded
torch.set_num_threads(1)
cpu_net = copy.deepcopy(model).cpu().eval()
def _fold(conv, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    f = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding, bias=True)
    f.weight.data = conv.weight * (bn.weight / std).view(-1, 1, 1, 1)
    f.bias.data = (-bn.running_mean) * bn.weight / std + bn.bias
    return f
cpu_net.stem[0] = _fold(cpu_net.stem[0], cpu_net.stem[1]); cpu_net.stem[1] = nn.Identity()
for blk in cpu_net.blocks:
    blk.c1 = _fold(blk.c1, blk.b1); blk.b1 = nn.Identity()
    blk.c2 = _fold(blk.c2, blk.b2); blk.b2 = nn.Identity()
x = torch.randn(25, C + 4, H, W)
with torch.inference_mode():
    for _ in range(3):
        cpu_net(x)
    t0 = time.time()
    for _ in range(5):
        cpu_net(x)
ms = 1000 * (time.time() - t0) / 5
print(f"CPU speed (1 thread): 25 spots x 1 pass = {ms:.0f} ms on this machine (the 1-step CNN needed 4 passes)")
