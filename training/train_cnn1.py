#!/usr/bin/env python3
"""
Fully-convolutional CNN: whole map in -> move probabilities (5) for EVERY tile out.
Loss only on tiles with a dino. Same game split as XGBoost (70 train / 10 dev / 20 test).

Default size is SMALL (width 32, 4 blocks): ~4-5x faster than the first CNN, so it fits the
server's ~150 ms per tick on CPU. The existing models/moves_cnn.pt is backed up first.

Usage:
    pip install torch matplotlib
    python state_encoder.py --logs logs      # once
    python train_cnn1.py                          # small server-friendly CNN
    python train_cnn1.py --width 64 --blocks 6    # the original big CNN
Outputs:
    models/moves_cnn.pt          weights + normalization + config (for the bot)
    models/loss_curve_cnn.png    train vs dev loss per epoch
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--width", type=int, default=32, help="filters per conv layer (64 = original big model)")
ap.add_argument("--blocks", type=int, default=4, help="residual blocks, 2 convs each (6 = original big model)")
ap.add_argument("--flip", action="store_true", help="augment with left-right mirroring")
ap.add_argument("--max-train", type=int, default=0, help="cap training samples per epoch (0 = all)")
ap.add_argument("--out", default="models/moves_cnn.pt", help="where to save the model (the bot loads models/moves_cnn.pt)")
args = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

# ---------------- data (memory-mapped, loaded per batch)
X = np.load(f"{args.data}/dataset_X_cnn.npy", mmap_mode="r")
Y = np.load(f"{args.data}/dataset_y_cnn.npy", mmap_mode="r")
meta = np.load(f"{args.data}/dataset_meta_cnn.npz")
CHANNELS, CLASSES, SPECIES = list(meta["channels"]), list(meta["classes"]), list(meta["species"])
CH = {n: i for i, n in enumerate(CHANNELS)}
N, C, H, W = X.shape
delay, shift = int(meta["meteor_delay"]), int(meta["shift_interval"])
print(f"{N:,} map samples, {C} channels, {H}x{W}, classes {CLASSES}")

# same game split as the tabular models: shuffle(seed 42), 80% train+dev / 20% test, dev = 10% of games
games = np.array(sorted(meta["games"]))
np.random.default_rng(42).shuffle(games)
n_tr, n_dev = int(0.8 * len(games)), max(1, int(0.1 * len(games)))
name_of = meta["games"][meta["game_idx"]]
idx_train = np.nonzero(np.isin(name_of, games[:n_tr - n_dev]))[0]
idx_dev = np.nonzero(np.isin(name_of, games[n_tr - n_dev:n_tr]))[0]
idx_test = np.nonzero(np.isin(name_of, games[n_tr:]))[0]
print(f"games {n_tr - n_dev}/{n_dev}/{len(games) - n_tr} | samples {len(idx_train):,}/{len(idx_dev):,}/{len(idx_test):,} (train/dev/test)")

# per-channel scale for the raw uint8 values
sample = X[np.sort(np.random.default_rng(0).choice(idx_train, min(5000, len(idx_train)), replace=False))]
scale = np.ones(C, np.float32)
scale[CH["elevation"]] = max(1.0, float(sample[:, CH["elevation"]].max()))
scale[CH["age"]] = 100.0


def make_batch(ids, flip=False):
    ids = np.sort(ids)                                    # sorted = faster memmap reads
    xb = torch.from_numpy(np.ascontiguousarray(X[ids])).to(dev).float()
    yb = torch.from_numpy(np.ascontiguousarray(Y[ids])).to(dev).long()
    sc = torch.from_numpy(meta["scalars"][ids].astype(np.float32)).to(dev)
    return prep(xb, sc), yb


def prep(xb, sc):
    """raw uint8 layers (B,C,H,W) + scalars (B,2) -> normalized float input (B, C+4, H, W)"""
    xb = xb / torch.from_numpy(scale).to(xb.device).view(1, -1, 1, 1)
    d = xb[:, CH["danger_turns"]]
    xb[:, CH["danger_turns"]] = torch.where(d > 0, (delay + 1 - d) / delay, torch.zeros_like(d))  # 1 = hits next
    for c in ("prev_dx", "prev_dy"):
        v = xb[:, CH[c]]
        xb[:, CH[c]] = torch.where(v > 0, v - 2, torch.zeros_like(v))
    B = xb.shape[0]
    rows = torch.linspace(0, 1, H, device=xb.device).view(1, 1, H, 1).expand(B, 1, H, W)   # 0 = trailing edge
    cols = torch.linspace(0, 1, W, device=xb.device).view(1, 1, 1, W).expand(B, 1, H, W)
    tshift = (sc[:, 0] / shift).view(B, 1, 1, 1).expand(B, 1, H, W)
    tmod = (sc[:, 1] / 100).view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([xb, rows, cols, tshift, tmod], 1)


def flip_lr(xb, yb):
    """mirror left-right: flip maps, negate prev_dx, swap left/right labels"""
    xb = xb.flip(-1)
    xb[:, CH["prev_dx"]] = -xb[:, CH["prev_dx"]]
    xb[:, C + 1] = xb[:, C + 1].flip(-1)                  # col plane back to 0..1
    l, r = CLASSES.index("-1,0"), CLASSES.index("1,0")
    yb = yb.flip(-1)
    yl, yr = yb == l, yb == r
    yb = yb.masked_fill(yl, r).masked_fill(yr, l)
    return xb, yb


# ---------------- model
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

    def forward(self, x):                                 # (B, cin, H, W) -> (B, ncls, H, W) logits
        return self.head(self.blocks(self.stem(x)))


model = DinoCNN(C + 4, args.width, args.blocks, len(CLASSES)).to(dev)
print(f"model: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k params, "
      f"receptive field {3 + 4 * args.blocks}x{3 + 4 * args.blocks} tiles")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)


def run(ids, train):
    model.train(train)
    tot_loss, tot_n = 0.0, 0
    order = np.random.permutation(ids) if train else ids
    with torch.set_grad_enabled(train):
        for a in range(0, len(order), args.batch):
            xb, yb = make_batch(order[a:a + args.batch])
            if train and args.flip and np.random.rand() < 0.5:
                xb, yb = flip_lr(xb, yb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, ignore_index=-1, reduction="sum")
            n = int((yb >= 0).sum())
            if train:
                opt.zero_grad()
                (loss / max(n, 1)).backward()
                opt.step()
            tot_loss += float(loss.detach()); tot_n += n
    return tot_loss / max(tot_n, 1)


# ---------------- train with early stopping on dev loss
os.makedirs("models", exist_ok=True)
if os.path.exists(args.out):
    old = torch.load(args.out, map_location="cpu", weights_only=False)
    backup = os.path.splitext(args.out)[0] + f"_w{old.get('width')}_b{old.get('blocks')}_backup.pt"
    if not os.path.exists(backup):
        import shutil
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
    if dv < best - 1e-4:
        best, bad = dv, 0
        torch.save({"state_dict": model.state_dict(), "scale": scale, "channels": CHANNELS, "classes": CLASSES,
                    "species": SPECIES, "width": args.width, "blocks": args.blocks, "H": H, "W": W,
                    "meteor_delay": delay, "shift_interval": shift}, args.out)
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
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (dino tiles)"); ax.set_title("CNN train vs dev loss")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); curve = os.path.splitext(args.out)[0].replace("moves_cnn", "loss_curve_cnn") + ".png"
plt.savefig(curve, dpi=120)
print("saved", curve)

# ---------------- test with the best checkpoint
ck = torch.load(args.out, map_location=dev, weights_only=False)
model.load_state_dict(ck["state_dict"]); model.eval()
sp_ch = [CH[f"sp_{s}"] for s in SPECIES]
correct, species, threat = [], [], []
with torch.no_grad():
    for a in range(0, len(idx_test), args.batch):
        ids = np.sort(idx_test[a:a + args.batch])
        xb, yb = make_batch(ids)
        pred = model(xb).argmax(1)
        raw = torch.from_numpy(np.ascontiguousarray(X[ids])).to(dev).float()
        danger = ((raw[:, CH["danger_turns"]] > 0) | (raw[:, CH["new_meteor"]] > 0)).float()
        near = F.max_pool2d(danger.unsqueeze(1), 3, 1, 1).squeeze(1) > 0      # own tile or any of 8 neighbours
        m = yb >= 0
        correct.append((pred == yb)[m].cpu().numpy())
        species.append(raw[:, sp_ch].argmax(1)[m].cpu().numpy())
        threat.append(near[m].cpu().numpy())
correct, species, threat = map(np.concatenate, (correct, species, threat))
ylab = np.concatenate([Y[np.sort(idx_test[a:a + 20000])][Y[np.sort(idx_test[a:a + 20000])] >= 0]
                       for a in range(0, len(idx_test), 20000)])
print(f"\nCNN TEST accuracy    : {correct.mean():.3f}   ({len(correct):,} dino moves)")
print(f"Baseline most-common : {(ylab == np.bincount(ylab).argmax()).mean():.3f}")
print("XGBoost (tuned) was  : 0.674\n")
print(f"{'species':15s} {'acc':>6s} {'n':>9s}")
for k, s in enumerate(SPECIES):
    sel = species == k
    if sel.any():
        print(f"{s:15s} {correct[sel].mean():6.3f} {sel.sum():9,d}")
print("\nCalm vs threatened (blast on/next to dino):")
for t in (False, True):
    sel = threat == t
    if sel.any():
        print(f"  {str(t):5s}  {correct[sel].mean():.3f}  {sel.sum():9,d}")
print(f"\nsaved {args.out}  (width {args.width}, {args.blocks} blocks)")

# ---------------- server speed check: one CPU thread, BatchNorm folded (like the bot)
import copy
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
ms = 1000 * (time.time() - t0) / 5 * 4
print(f"CPU speed (1 thread): 25 spots x 4 passes = {ms:.0f} ms on this machine")
print("  the server was ~1.5x slower than a fast CPU; the bot needs this well under ~90 ms to test ~25 spots")
