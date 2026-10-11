#!/usr/bin/env python3
"""
Fit and test the BEHAVIOUR MODEL (bot/candidate_bots/behavior_model.py): learns, per species, how much each
confirmed spec rule matters for the next move, then tests it on games it never saw.

Tests (unseen games):
  next move     top-1 accuracy and log-loss vs two baselines:
                  uniform  = every possible move equally likely
                  habit    = each species' overall move habits (stay/up/down/left/right), no rules
  4 moves ahead average probability the model gives to the tile the dino really is on 4 ticks later,
                and how often that tile is the model's single most likely tile, vs a random-walk baseline
                (this is the horizon a meteor needs: it lands 4 ticks after launch)
Prints the learned weight of every rule per species (which rules matter for whom).

Usage (repo root):
  python training/fit_behavior.py --logs logs/local_game_logs
  python training/fit_behavior.py --logs logs/local_game_logs --games 400 --test-games 60
Writes bot/models/behavior/weights.json and results/behavior_model_report.txt
"""
import argparse, glob, json, math, os, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bot", "candidate_bots"))
import behavior_model as BM

MOVES = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def load(path):
    S = []
    for l in open(path, "rb"):
        if l.strip():
            try:
                S.append(json.loads(l)["state"])
            except Exception:
                pass
    S.sort(key=lambda s: s["currentTick"])
    return S


def samples(args):
    """(species, features [n_opt x F], chosen index, move type, option move types) for sampled moves of one game"""
    path, rate, seed = args
    rng = random.Random(seed)
    S = load(path)
    out, last = [], {}
    prev = None
    for s in S:
        b = BM.Board(s)
        if prev is not None and b.t == prev.t + 1:
            for i, (sp, p) in prev.d.items():
                if i not in b.d or sp not in "SVTR":
                    continue
                q = b.d[i][1]
                mv = (q[0] - p[0], q[1] - p[1])
                if mv not in MOVES:
                    continue
                if rng.random() < rate:
                    opts = prev.options(p)
                    if q in opts:
                        f = BM.features(prev, i, opts, last)
                        types = [MOVES[(o[0] - p[0], o[1] - p[1])] for o in opts]
                        out.append((sp, f, opts.index(q), MOVES[mv], types))
                if mv != (0, 0):
                    last[i] = mv
        prev = b
    return out


def fit(data, iters=400, lr=0.05, l2=1e-3):
    F = len(BM.FEATS)
    n = len(data)
    X = np.zeros((n, 5, F), np.float32)
    M = np.zeros((n, 5), bool)
    y = np.zeros(n, np.int64)
    for k, (f, c) in enumerate(data):
        X[k, :len(f)] = f
        M[k, :len(f)] = True
        y[k] = c
    mu, sd = X[M].mean(0), X[M].std(0) + 1e-6           # standardise for stable steps, undo at the end
    Xs = np.where(M[..., None], (X - mu) / sd, 0)
    w = np.zeros(F)
    m1, m2 = np.zeros(F), np.zeros(F)
    for it in range(1, iters + 1):
        s = Xs @ w
        s = np.where(M, s, -1e9)
        s -= s.max(1, keepdims=True)
        p = np.exp(s) * M
        p /= p.sum(1, keepdims=True)
        g = (p[..., None] * Xs).sum(1) - Xs[np.arange(n), y]
        g = g.mean(0) + l2 * w
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        w -= lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
    return (w / sd).tolist()           # weights on the raw features (the constant shift cancels in the softmax)


def forecast_test(args):
    """4-moves-ahead test on one unseen game: P(model) on the real tile vs random walk"""
    path, weights, n_max, seed = args
    rng = random.Random(seed)
    M = BM.BehaviorModel(weights)
    S = load(path)
    by_t = {s["currentTick"]: s for s in S}
    res = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])        # sp -> [P model, top hit, P walk, top walk, n]
    last, prev = {}, None
    picks = 0
    for s in S:
        b = BM.Board(s)
        if prev is not None and b.t == prev.t + 1:
            for i, (sp, p) in prev.d.items():
                if i in b.d:
                    mv = (b.d[i][1][0] - p[0], b.d[i][1][1] - p[1])
                    if mv != (0, 0) and mv in MOVES:
                        last[i] = mv
        prev = b
        fut = by_t.get(b.t + 4)
        if fut is None or picks >= n_max or rng.random() > 0.05:
            continue
        real = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in fut["dinosaurs"]}
        ids = [i for i in b.d if i in real and b.d[i][0] in "SVTR"]
        if not ids:
            continue
        i = rng.choice(ids)
        sp = b.d[i][0]
        dm = M.forecast(b, last, 4, only=[i])[i]
        walk = BM.BehaviorModel({k: [0.0] * len(BM.FEATS) for k in "SVTR"}).forecast(b, last, 4, only=[i])[i]
        r = res[sp]
        r[0] += dm.get(real[i], 0.0)
        r[1] += max(dm, key=dm.get) == real[i]
        r[2] += walk.get(real[i], 0.0)
        r[3] += max(walk, key=walk.get) == real[i]
        r[4] += 1
        picks += 1
    return {k: list(v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/local_game_logs")
    ap.add_argument("--games", type=int, default=300, help="training games")
    ap.add_argument("--test-games", type=int, default=50)
    ap.add_argument("--rate", type=float, default=0.15, help="share of moves sampled per game")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "bot", "models", "behavior", "weights.json"))
    ap.add_argument("--report", default="results/behavior_model_report.txt")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.logs, "*.jsonl")))
    random.Random(0).shuffle(files)
    test, train = files[:a.test_games], files[a.test_games:a.test_games + a.games]
    say(f"behaviour model | train {len(train)} games, test {len(test)} unseen games | sampled {a.rate:.0%} of moves")
    with Pool(a.workers) as pool:
        tr = [x for r in pool.imap_unordered(samples, [(f, a.rate, k) for k, f in enumerate(train)], chunksize=4) for x in r]
        te = [x for r in pool.imap_unordered(samples, [(f, a.rate, 10 ** 6 + k) for k, f in enumerate(test)], chunksize=4) for x in r]
    say(f"moves: train {len(tr)}, test {len(te)}")
    W, habit = {}, {}
    for sp in "SVTR":
        d = [(f, c) for s2, f, c, _, _ in tr if s2 == sp]
        if len(d) < 100:
            continue
        W[sp] = fit(d)
        cnt = Counter(m for s2, _, _, m, _ in tr if s2 == sp)
        habit[sp] = [cnt[k] / sum(cnt.values()) for k in range(5)]
    M = BM.BehaviorModel(W)

    say("\nNEXT MOVE on unseen games (accuracy = picks the real move as most likely; log-loss lower is better)")
    for sp in "SVTR":
        rows = [x for x in te if x[0] == sp]
        if not rows or sp not in W:
            continue
        acc = {"model": 0, "habit": 0, "uniform": 0}
        ll = {"model": 0.0, "habit": 0.0, "uniform": 0.0}
        w = np.array(W[sp])
        for _, f, c, _, types in rows:
            s = np.array(f) @ w
            pm = np.exp(s - s.max()); pm /= pm.sum()
            ph = np.array([habit[sp][t] for t in types]) + 1e-6; ph /= ph.sum()
            pu = np.full(len(f), 1 / len(f))
            for k, pr in (("model", pm), ("habit", ph), ("uniform", pu)):
                acc[k] += int(np.argmax(pr) == c)
                ll[k] -= math.log(max(pr[c], 1e-9))
        n = len(rows)
        say(f"  {BM_NAME[sp]:12} n={n:6d} | accuracy model {acc['model'] / n:.1%} habit {acc['habit'] / n:.1%} "
            f"uniform {acc['uniform'] / n:.1%} | log-loss model {ll['model'] / n:.3f} habit {ll['habit'] / n:.3f} "
            f"uniform {ll['uniform'] / n:.3f}")

    say("\n4 MOVES AHEAD on unseen games (what a meteor needs): P given to the real tile | real tile = most likely tile")
    with Pool(a.workers) as pool:
        parts = pool.map(forecast_test, [(f, W, 40, k) for k, f in enumerate(test)])
    tot = defaultdict(lambda: [0.0] * 5)
    for r in parts:
        for sp, v in r.items():
            tot[sp] = [x + y for x, y in zip(tot[sp], v)]
    for sp in "SVTR":
        v = tot.get(sp)
        if v and v[4]:
            say(f"  {BM_NAME[sp]:12} n={int(v[4]):5d} | model P {v[0] / v[4]:.3f}, top tile right {v[1] / v[4]:.1%} | "
                f"random walk P {v[2] / v[4]:.3f}, top tile right {v[3] / v[4]:.1%}")

    say("\nLEARNED RULE WEIGHTS (per unit of the feature; + = makes that move more likely)")
    say("  " + f"{'rule':14}" + "".join(f"{BM_NAME[s][:12]:>14}" for s in "SVTR" if s in W))
    for k, name in enumerate(BM.FEATS):
        say("  " + f"{name:14}" + "".join(f"{W[s][k]:14.2f}" for s in "SVTR" if s in W))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"features": BM.FEATS, "weights": W}, open(a.out, "w"), indent=1)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    open(a.report, "w").write("\n".join(OUT) + "\n")
    print(f"\nweights saved to {os.path.normpath(a.out)}\nreport saved to {a.report}")


BM_NAME = {"S": "Stegosaurus", "V": "Velociraptor", "T": "Triceratops", "R": "T-Rex"}

if __name__ == "__main__":
    main()
