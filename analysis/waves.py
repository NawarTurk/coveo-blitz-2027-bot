#!/usr/bin/env python3
"""
waves.py - wave rules.  Answers Q1-Q7 of results/research_questions.md.

Usage:  python analysis/waves.py --logs local_game_logs [--games 300] [--server server_logs]
        --server = folder of downloaded server TEAM logs (*.txt with T| trace lines), used for Q7
Writes: results/analysis/waves.txt
"""
import glob, json, os
from collections import Counter, defaultdict

from common import Report, args_and_files, run_games, load, events, CYCLE, recipe, pct, med


def one(path):
    recs = load(path)
    if not recs:
        return None
    born, died, waves, alive = events(recs)
    return dict(waves=[{k: w.get(k) for k in ("t", "n", "recipe", "trigger", "alive_before", "gap_after_empty", "c")}
                       for w in waves],
                max_alive=max(alive.values()) if alive else 0)


def server_recipes(folder):
    """recipe per wave number from server team logs (T| lines)"""
    by_wave = defaultdict(Counter)
    for f in glob.glob(os.path.join(folder, "*.txt")):
        R = []
        for l in open(f, errors="ignore"):
            if l.startswith("T|"):
                try:
                    R.append(json.loads(l[2:]))
                except Exception:
                    pass
        if not R:
            continue
        R.sort(key=lambda r: r["t"])
        seen, waves, last = set(), [], -99
        for r in R:
            new = [d for d in r["d"] if d[0] not in seen]
            for d in new:
                seen.add(d[0])
            if new:
                if r["t"] - last > 2:
                    waves.append(Counter())
                last = r["t"]
                for d in new:
                    waves[-1][d[1]] += 1
        for k, c in enumerate(waves, 1):
            by_wave[k][recipe(c)] += 1
    return by_wave


def main():
    a, files = args_and_files(__doc__, lambda ap: ap.add_argument("--server", default=""))
    G = run_games(one, files, a.workers)
    rep = Report("waves")
    rep.line(f"waves.py | {len(G)} games from {a.logs}")
    W = [(gi, k, w) for gi, g in enumerate(G) for k, w in enumerate(g["waves"])]
    allw = [w for _, _, w in W]

    # Q1
    gaps = Counter(w["gap_after_empty"] for w in allw if w["trigger"] == "clear")
    trig = Counter(w["trigger"] for w in allw)
    rep.q(1, "When does a new wave arrive?",
          f"after a clear: {dict(sorted(gaps.items()))} ticks after the board went empty (n={sum(gaps.values())}); "
          f"otherwise on the timer. Triggers: {dict(trig)}")

    # Q2
    timed = [(gi, k, w) for gi, k, w in W if w["trigger"] == "timer"]
    mod = Counter(w["t"] % 100 for _, _, w in timed)
    rep.q(2, "Does the wave timer reset after we clear the board?",
          f"timed waves come exactly 100 ticks after the previous wave started: {len(timed)} waves; "
          f"their ticks mod 100 spread over {len(mod)} different values "
          f"({'resets: not fixed multiples of 100' if len(mod) > 3 else 'possibly fixed schedule'})")

    # Q3
    cap = max(g["max_alive"] for g in G)
    part = [w for w in allw if w["alive_before"] > 0]
    fills = sum(1 for w in part if w["n"] + w["alive_before"] == 20)
    rep.q(3, "How many dinos can be alive at once, and what happens to a wave at the cap?",
          f"max alive seen {cap}; waves arriving with dinos still alive: {len(part)}, of which "
          f"{fills} filled exactly up to 20")
    rep.d("waves arriving with N alive -> new dinos: " +
          ", ".join(f"{k}:{med([w['n'] for w in part if w['alive_before'] == k])}"
                    for k in sorted({w['alive_before'] for w in part})[:12]))

    # Q4
    miss = Counter()
    nm = 0
    for gi, k, w in W:
        exp = CYCLE[k % 10]
        if w["recipe"] != exp and w["alive_before"] > 0:
            ec = Counter({s: int(x[:-1]) for x in exp.split() for s in [x[-1]]})
            for s in "SVTR":
                if ec[s] > w["c"][s]:
                    miss[s] += ec[s] - w["c"][s]
            nm += 1
    rep.q(4, "At the 20 cap, which members of a wave are omitted?",
          f"{nm} partial waves; missing dinos by species: {dict(miss)}" if nm else "no partial waves found")

    # Q5
    by_k = defaultdict(Counter)
    for gi, k, w in W:
        by_k[k + 1][w["recipe"]] += 1
    exact = [(k, *by_k[k].most_common(1)[0], sum(by_k[k].values())) for k in sorted(by_k) if sum(by_k[k].values()) >= 5]
    rep.q(5, "Is the species mix of each wave fixed?",
          "yes" if exact and all(c / n > 0.6 for _, _, c, n in exact[:10]) else "not clearly")
    for k, r, c, n in exact[:20]:
        rep.d(f"wave {k:2d}: {r:14s} {pct(c, n)} of {n} games")

    # Q6
    same = tot = 0
    for g in G:
        ws = g["waves"]
        for k in range(len(ws) - 10):
            tot += 1
            same += ws[k]["recipe"] == ws[k + 10]["recipe"]
    rep.q(6, "Does the mix repeat in a cycle? How long?",
          f"wave k = wave k+10 in {pct(same, tot)} of {tot} pairs; cycle: " + " | ".join(CYCLE))

    # Q7
    if a.server and os.path.isdir(a.server):
        sv = server_recipes(a.server)
        agree = [(k, sv[k].most_common(1)[0][0], by_k[k].most_common(1)[0][0]) for k in sorted(sv)
                 if k in by_k and sum(sv[k].values()) >= 3]
        ok = sum(1 for _, s_, l_ in agree if s_ == l_)
        rep.q(7, "Same cycle locally and on the server?", f"same top recipe for {ok}/{len(agree)} wave numbers")
        for k, s_, l_ in agree:
            if s_ != l_:
                rep.d(f"wave {k}: server {s_} | local {l_}")
    else:
        rep.q(7, "Same cycle locally and on the server?", "pass --server <folder of team logs> to compare")

    rep.save()


if __name__ == "__main__":
    main()
