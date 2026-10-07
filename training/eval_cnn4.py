#!/usr/bin/env python3
"""
Test a trained CNN4 (or CNN4prev7) on N games, on the REAL 13-tile meteor blast.

For every meteor the bot actually launched (and the server accepted) in those games:
    predicted kills = sum of P(dino) over the 13 blast tiles at the impact tick
    actual kills    = dinos really standing on those tiles at the impact tick
and compared with the naive guess "dinos don't move".
Also: per-tile hit rate and species accuracy at the impact tick.

Usage:
    python eval_cnn4.py                                   # CNN4, 20 games from its TEST split (never trained on)
    python eval_cnn4.py --model models/cnn4prev7/moves_cnn4prev7.pt
    python eval_cnn4.py --games 50
    python eval_cnn4.py --latest                          # the 20 newest logs instead (play fresh games first!)
Needs: state_encoder.py, torch
"""
import argparse, glob, json, os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from state_encoder import encode_state, footprint_from

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/cnn4/moves_cnn4.pt")
ap.add_argument("--logs", default="local_game_logs")
ap.add_argument("--games", type=int, default=20)
ap.add_argument("--latest", action="store_true", help="use the newest logs instead of the model's test split")
args = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(args.model, map_location="cpu", weights_only=False)
CHANNELS, CLASSES = [str(c) for c in ck["channels"]], [str(c) for c in ck["classes"]]
SPECIES = CLASSES[1:]
SP_ID = {s: i + 1 for i, s in enumerate(SPECIES)}
CH = {n: i for i, n in enumerate(CHANNELS)}
K = int(ck.get("history", 1))
H, W, delay, shift = ck["H"], ck["W"], ck["meteor_delay"], ck["shift_interval"]
C = len(CHANNELS)
DX = [CH[c] for c in CHANNELS if c.endswith("_dx")]
DY = [CH[c] for c in CHANNELS if c.endswith("_dy")]
print(f"model {args.model}: {C} layers, {K} moves of history")


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class DinoCNN(nn.Module):
    def __init__(self, cin, width, blocks, ncls):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, ncls, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


net = DinoCNN(C + 4, ck["width"], ck["blocks"], len(CLASSES))
net.load_state_dict(ck["state_dict"])
net = net.to(dev).eval()
scale = torch.tensor(np.asarray(ck["scale"], np.float32), device=dev).view(1, -1, 1, 1)


def prep(xb, sc):
    xb = xb / scale
    d = xb[:, CH["danger_turns"]]
    xb[:, CH["danger_turns"]] = torch.where(d > 0, (delay + 1 - d) / delay, torch.zeros_like(d))
    for c in DX + DY:
        v = xb[:, c]
        xb[:, c] = torch.where(v > 0, v - 2, torch.zeros_like(v))
    B = xb.shape[0]
    rows = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, 1, H, W)
    cols = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, 1, H, W)
    tshift = (sc[:, 0] / shift).view(B, 1, 1, 1).expand(B, 1, H, W)
    tmod = (sc[:, 1] / 100).view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([xb, rows, cols, tshift, tmod], 1)


def add_history(X, s, hist):
    if K <= 1:
        return X
    extra = np.zeros((2 * (K - 1), H, W), np.uint8)
    ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
    for d in s["dinosaurs"]:
        i, j = d["position"]["y"] - oy, d["position"]["x"] - ox
        h = list(hist.get(d["id"], ()))
        if not (0 <= i < H and 0 <= j < W) or not h:
            continue
        for k in range(2, K + 1):
            if len(h) < k + 1:
                break
            (x1, y1), (x0, y0) = h[-k], h[-k - 1]
            extra[2 * (k - 2), i, j] = int(np.clip(x1 - x0, -1, 1)) + 2
            extra[2 * (k - 2) + 1, i, j] = int(np.clip(y1 - y0, -1, 1)) + 2
    return np.concatenate([X, extra], 0)


def game_samples(path):
    """only the ticks with an accepted launch: (X, scalars, label grid at impact tick)"""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if len(recs) < 900:
        return []
    fp = footprint_from(recs)
    by_t = {r["state"]["currentTick"]: r for r in recs}
    out, prev, hist = [], {}, {}
    for r in recs:
        s = r["state"]
        t = s["currentTick"]
        pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        for i_, p_ in pos.items():
            hist.setdefault(i_, deque(maxlen=K + 1)).append(p_)
        for i_ in [i_ for i_ in hist if i_ not in pos]:
            del hist[i_]
        fut, nxt = by_t.get(t + delay), by_t.get(t + 1)
        acts = r.get("actions") or []
        if fut and nxt and s["dinosaurs"] and acts and acts[0].get("type") == "LAUNCH_METEOR":
            tgt = (acts[0]["target"]["x"], acts[0]["target"]["y"])
            if any((m["target"]["x"], m["target"]["y"]) == tgt and m["turnsUntilImpact"] == delay
                   for m in nxt["state"]["meteors"]):
                X, sc = encode_state(s, acts, prev, fp, H, W)
                X = add_history(X, s, hist)
                ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
                y = np.zeros((H, W), np.int8)
                for d in fut["state"]["dinosaurs"]:
                    i, j = d["position"]["y"] - oy, d["position"]["x"] - ox
                    if 0 <= i < H and 0 <= j < W:
                        y[i, j] = -1 if d["id"] not in pos else SP_ID.get(d["name"], -1)
                out.append((X, sc, y))
        prev = pos
    return out


# ---------------- pick games
files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
meta_path = os.path.join(os.path.dirname(args.model).replace("models", "data"),
                         "dataset_meta_" + os.path.basename(os.path.dirname(args.model)) + ".npz")
if not args.latest and os.path.exists(meta_path):
    meta = np.load(meta_path)
    games = np.array(sorted(meta["games"]))
    np.random.default_rng(42).shuffle(games)
    test = set(games[int(0.8 * len(games)):])          # same split as training: never trained on
    pick = [f for f in files if os.path.splitext(os.path.basename(f))[0] in test][: args.games]
    print(f"{len(pick)} games from the model's TEST split ({meta_path})")
else:
    pick = files[-args.games:]
    print(f"{len(pick)} newest games (careful: they may have been in training)")

# ---------------- evaluate
pk, tk, sk, tp, fp_, fn, sp_ok, sp_n = [], [], [], 0, 0, 0, 0, 0
sp_ch = [CH[f"sp_{s}"] for s in SPECIES]
for f in pick:
    S = game_samples(f)
    if not S:
        continue
    for a in range(0, len(S), 256):
        chunk = S[a:a + 256]
        raw = torch.from_numpy(np.stack([x for x, _, _ in chunk])).to(dev).float()
        sc = torch.tensor([s for _, s, _ in chunk], dtype=torch.float32, device=dev)
        yb = torch.from_numpy(np.stack([y for _, _, y in chunk])).to(dev).long()
        with torch.no_grad():
            prob = F.softmax(net(prep(raw.clone(), sc)), 1)
        p_dino = 1 - prob[:, 0]
        m = yb >= 0
        blast = raw[:, CH["new_meteor"]] > 0                       # the real 13 tiles
        now = raw[:, sp_ch].sum(1) > 0
        pk.append((p_dino * blast * m).sum((1, 2)).cpu().numpy())
        tk.append(((yb > 0) & blast).sum((1, 2)).float().cpu().numpy())
        sk.append((now & blast).sum((1, 2)).float().cpu().numpy())
        occ_p, occ_t = (p_dino > 0.5) & m, yb > 0
        tp += int((occ_p & occ_t).sum()); fp_ += int((occ_p & ~occ_t).sum()); fn += int((~occ_p & occ_t).sum())
        both = occ_p & occ_t
        sp_ok += int((prob[:, 1:].argmax(1) + 1 == yb)[both].sum()); sp_n += int(both.sum())
    print(f"  {os.path.basename(f)}: {len(S)} launches")

pk, tk, sk = map(np.concatenate, (pk, tk, sk))
print("\n" + "=" * 70)
print(f"REAL 13-tile blast, {len(pk):,} launches")
print(f"  actual kills per meteor      {tk.mean():.3f}   (meteors that killed something: {(tk > 0).mean():.1%})")
print(f"  CNN4 predicted per meteor    {pk.mean():.3f}   mean abs error {np.abs(pk - tk).mean():.3f}")
print(f"  'dinos don't move' guess     {sk.mean():.3f}   mean abs error {np.abs(sk - tk).mean():.3f}")
if tk.std() > 0:
    print(f"  correlation CNN4 vs actual   {np.corrcoef(pk, tk)[0, 1]:.3f}   (dinos-don't-move: {np.corrcoef(sk, tk)[0, 1]:.3f})")
print("  calibration (CNN4 says -> really happened):")
for lo, hi in [(0, .1), (.1, .25), (.25, .5), (.5, 1), (1, 2), (2, 99)]:
    sel = (pk >= lo) & (pk < hi)
    if sel.any():
        print(f"    {lo:>4}-{hi:<4} n={sel.sum():6,d}  predicted {pk[sel].mean():.2f}  actual {tk[sel].mean():.2f}")
# can it tell good shots from bad ones?
order = np.argsort(-pk)
top = order[: max(1, len(pk) // 10)]
print(f"  its 10% most confident shots kill {tk[top].mean():.2f} on average (all shots: {tk.mean():.2f})")
prec, rec = tp / max(tp + fp_, 1), tp / max(tp + fn, 1)
print(f"\nPer tile at impact (P(dino) > 0.5): precision {prec:.2f}, recall {rec:.2f} | species right {sp_ok / max(sp_n, 1):.1%}")
