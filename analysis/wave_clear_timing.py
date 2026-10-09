#!/usr/bin/env python3
"""
Wave clear timing: where do the ticks of each wave go?

A wave = the dinos that appear on the same tick (spawns within 2 ticks are merged).
For every wave we measure, in ticks after it spawned:
  first death, 50% dead, 80% dead, all dead (wave cleared)
and how the last dino died (our meteor / lava / scroll / eaten by T-Rex / other).

The top teams score ~10x more than us, which needs ~10x more waves: a new wave only comes 2 ticks
after the board is empty (or every 100 ticks). This script shows whether our time goes into the
first kills or into hunting the last stragglers.

Usage:   python analysis/wave_clear_timing.py --logs local_game_logs
         python analysis/wave_clear_timing.py --logs local_game_logs --recent 200
Writes:  results/wave_timing.csv  (one row per wave)
"""
import argparse, csv, glob, json, os
from collections import Counter
from multiprocessing import Pool

import numpy as np

MERGE_TICKS = 2       # spawns this close together count as one wave
EAT_DIST = 2


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def death_cause(d, prev, now):
    """why dino d (as seen in state prev) is missing from state now"""
    p = (d["position"]["x"], d["position"]["y"])
    for me in prev["meteors"]:
        if me["turnsUntilImpact"] == 1 and any((q["x"], q["y"]) == p for q in me["impactedTiles"]):
            return "meteor"
    m = now["map"]
    ox, oy, h, w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
    for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        i, j = p[1] + b - oy, p[0] + a - ox
        if 0 <= i < h and 0 <= j < w and m["tiles"][i][j]["hasLava"]:
            return "lava"
    if p[1] < oy:
        return "scroll"
    if d["name"] != "Tyrannosaurus" and any(
            manh(p, (t["position"]["x"], t["position"]["y"])) <= EAT_DIST
            for t in prev["dinosaurs"] if t["name"] == "Tyrannosaurus" and t["id"] != d["id"]):
        return "eaten"
    return "other"


def one_game(path):
    try:
        states = [json.loads(l)["state"] for l in open(path) if l.strip()]
    except Exception:
        return None
    if len(states) < 100:
        return None
    born, died, info = {}, {}, {}
    prev = None
    for s in states:
        t = s["currentTick"]
        ids = {d["id"] for d in s["dinosaurs"]}
        for d in s["dinosaurs"]:
            if d["id"] not in born:
                born[d["id"]] = t
                info[d["id"]] = d["name"]
        if prev is not None:
            for d in prev["dinosaurs"]:
                if d["id"] not in ids and d["id"] not in died:
                    died[d["id"]] = (t, death_cause(d, prev, s), d["age"])
        prev = s
    end_tick = states[-1]["currentTick"]
    # group spawns into waves
    order = sorted(born, key=lambda i: born[i])
    waves, cur = [], []
    for i in order:
        if cur and born[i] - born[cur[-1]] > MERGE_TICKS:
            waves.append(cur); cur = []
        cur.append(i)
    if cur:
        waves.append(cur)
    rows = []
    for k, w in enumerate(waves):
        t0 = born[w[0]]
        deaths = sorted(died[i][0] - t0 for i in w if i in died)
        n = len(w)
        done = len(deaths) == n
        q = lambda f: deaths[int(np.ceil(f * n)) - 1] if len(deaths) >= int(np.ceil(f * n)) else None
        last = max((i for i in w if i in died), key=lambda i: died[i][0], default=None)
        # was this wave triggered by a clear (board empty just before) or by the cadence?
        nxt = waves[k + 1] if k + 1 < len(waves) else None
        rows.append(dict(
            game=os.path.basename(path), wave=k, spawn_tick=t0, n=n,
            first=deaths[0] if deaths else None, p50=q(0.5), p80=q(0.8), all=deaths[-1] if done else None,
            cleared=int(done),
            last_cause=died[last][1] if (done and last) else "", last_species=info[last] if (done and last) else "",
            last_age=died[last][2] if (done and last) else "",
            meteor_kills=sum(1 for i in w if i in died and died[i][1] == "meteor"),
            lava_kills=sum(1 for i in w if i in died and died[i][1] == "lava"),
            scroll=sum(1 for i in w if i in died and died[i][1] == "scroll"),
            eaten=sum(1 for i in w if i in died and died[i][1] == "eaten"),
            gap_to_next=(born[nxt[0]] - (t0 + deaths[-1])) if (nxt and done) else None,
            score=states[-1]["score"], _deaths=deaths,
            _dinos=[(info[i], (died[i][0] - born[i]) if i in died else None, died[i][1] if i in died else "alive",
                     i == last and done) for i in w]))
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def dead_curve(waves, X):
    """average share of a wave's dinos dead at each tick after spawn"""
    out = np.zeros(len(X))
    for r in waves:
        d = np.array(r["_deaths"])
        out += np.array([(d <= x).sum() for x in X]) / r["n"]
    return out / max(len(waves), 1)


def plot(rows, sc, k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = np.arange(0, 201)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # left: share of the wave dead vs ticks since spawn
    ax[0].plot(X, 100 * dead_curve(rows, X), lw=2.5, label=f"all waves ({len(rows)})")
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    ranked = sorted(by_game, key=lambda g: by_game[g][0]["score"])
    kk = max(1, len(ranked) // 10)
    for name, gs, st in (("lowest-score 10% games", ranked[:kk], "--"), ("highest-score 10% games", ranked[-kk:], ":")):
        w = [r for g in gs for r in by_game[g]]
        ax[0].plot(X, 100 * dead_curve(w, X), st, lw=2, label=name)
    for y in (50, 80, 100):
        ax[0].axhline(y, color="gray", lw=0.6, ls="--")
    ax[0].set_xlabel("ticks since the wave spawned"); ax[0].set_ylabel("% of the wave dead")
    ax[0].set_title("How fast a wave dies (cumulative)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    # right: share of waves fully cleared by tick x
    cl = np.array([r["all"] for r in rows if r["cleared"]])
    ax[1].plot(X, 100 * np.array([(cl <= x).sum() for x in X]) / len(rows), lw=2.5, color="C3")
    ax[1].axvline(100, color="gray", lw=0.8, ls="--", label="next wave anyway (100-tick schedule)")
    ax[1].set_xlabel("ticks since the wave spawned"); ax[1].set_ylabel("% of waves fully cleared")
    ax[1].set_title("Cumulative: waves cleared by tick x"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--recent", type=int, default=0, help="only the newest N games")
    ap.add_argument("--out", default="results/wave_timing.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    if args.recent:
        files = files[-args.recent:]
    with Pool() as pool:
        games = [g for g in pool.map(one_game, files, chunksize=4) if g]
    rows = [r for g in games for r in g]
    if not rows:
        raise SystemExit("no usable games")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fields = [k for k in rows[0] if not k.startswith("_")]
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)

    G = len(games)
    cl = [r for r in rows if r["cleared"]]
    print(f"{G} games, {len(rows)} waves ({len(rows) / G:.1f} per game), "
          f"{100 * len(cl) / len(rows):.0f}% of waves fully cleared (rest: still alive at game end or overlapping)")
    print(f"dinos per wave: median {med([r['n'] for r in rows]):.0f}")

    print("\nTICKS AFTER A WAVE SPAWNS (cleared waves, median):")
    print(f"  first death   {med([r['first'] for r in cl]):6.0f}")
    print(f"  50% dead      {med([r['p50'] for r in cl]):6.0f}")
    print(f"  80% dead      {med([r['p80'] for r in cl]):6.0f}")
    print(f"  all dead      {med([r['all'] for r in cl]):6.0f}   <- ticks per wave")
    tail = [(r["all"] - r["p80"]) / r["all"] for r in cl if r["all"] and r["p80"] is not None]
    print(f"  share of the wave's time spent on the last 20% of dinos: {100 * med(tail):.0f}% (median)")
    gaps = [r["gap_to_next"] for r in cl if r["gap_to_next"] is not None]
    after_clear = [g for g in gaps if g >= 0]
    print(f"  next wave came after a clear: {len(after_clear)} waves, median gap {med(after_clear):.0f} ticks | "
          f"came on the 100-tick schedule while the previous wave was still alive: {len(gaps) - len(after_clear)}")

    print("\nHOW THE LAST DINO OF EACH WAVE DIED:")
    c = Counter(r["last_cause"] for r in cl)
    for k, v in c.most_common():
        print(f"  {k:<8} {100 * v / len(cl):5.1f}%")
    s = Counter(r["last_species"] for r in cl)
    print("  species of the last dino: " + ", ".join(f"{k} {100 * v / len(cl):.0f}%" for k, v in s.most_common()))

    print("\nALL DEATHS IN WAVES:")
    tot = {k: sum(r[k] for r in rows) for k in ("meteor_kills", "lava_kills", "scroll", "eaten")}
    n_all = sum(r["n"] for r in rows)
    for k, v in tot.items():
        print(f"  {k:<13} {100 * v / n_all:5.1f}%")

    # per species
    from collections import defaultdict
    sp = defaultdict(lambda: {"n": 0, "life": [], "cause": Counter(), "last": 0})
    for r in rows:
        for name, life, cause, is_last in r["_dinos"]:
            x = sp[name]
            x["n"] += 1
            if life is not None:
                x["life"].append(life)
            x["cause"][cause] += 1
            x["last"] += int(bool(is_last))
    n_all_d = sum(x["n"] for x in sp.values())
    n_last = sum(x["last"] for x in sp.values())
    print("\nPER SPECIES:")
    print(f"  {'species':<14}{'spawns':>8}{'life (med)':>11}{'meteor':>8}{'lava':>7}{'scroll':>8}{'eaten':>7}{'alive':>7}"
          f"{'is last':>9}{'last/share':>12}")
    for name, x in sorted(sp.items(), key=lambda kv: -kv[1]["n"]):
        share = x["n"] / n_all_d
        last_share = x["last"] / max(n_last, 1)
        c = x["cause"]
        pc = lambda k: f"{100 * c[k] / x['n']:.0f}%"
        print(f"  {name:<14}{100 * share:7.0f}%{med(x['life']):11.0f}{pc('meteor'):>8}{pc('lava'):>7}{pc('scroll'):>8}"
              f"{pc('eaten'):>7}{pc('alive'):>7}{100 * last_share:8.0f}%{last_share / max(share, 1e-9):11.1f}x")
    print("  last/share > 1 = this species is the straggler more often than its numbers explain")

    # do faster waves mean higher scores?
    per_game = {}
    for r in rows:
        g = per_game.setdefault(r["game"], {"score": r["score"], "waves": 0, "all": []})
        g["waves"] += 1
        if r["cleared"]:
            g["all"].append(r["all"])
    sc = sorted(per_game.values(), key=lambda g: g["score"])
    k = max(1, len(sc) // 10)
    for name, part in (("lowest 10% games", sc[:k]), ("highest 10% games", sc[-k:])):
        print(f"\n{name}: score {np.mean([g['score'] for g in part]):,.0f} | waves/game {np.mean([g['waves'] for g in part]):.1f} | "
              f"ticks per wave {med([x for g in part for x in g['all']]):.0f}")
    plot(rows, sc, k, os.path.splitext(args.out)[0] + ".png")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
