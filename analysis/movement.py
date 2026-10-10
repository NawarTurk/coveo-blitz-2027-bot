#!/usr/bin/env python3
"""
movement.py - how dinos move and react.  Answers Q16-Q25 of results/research_questions.md.

calm       = the dino is not inside any falling meteor's blast and no meteor centre is within 5 tiles
threatened = the dino stands inside a falling meteor's blast
first move vs the blast centre: away (distance grows) / sideways (same distance) / toward / stay

Usage:  python analysis/movement.py --logs local_game_logs [--games 300]
Writes: results/analysis/movement.txt
"""
from collections import Counter, defaultdict, deque

from common import Report, args_and_files, run_games, load, events, manh, pos, origin, SP, pct, mean, med

DIRS = {(0, 0): "stay", (0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}


def kind(p0, p1, c):
    if p1 == p0:
        return "stay"
    d0, d1 = manh(p0, c), manh(p1, c)
    return "away" if d1 > d0 else "toward" if d1 < d0 else "sideways"


def one(path):
    recs = load(path)
    if not recs:
        return None
    born, died, waves, alive = events(recs)
    st = [s for s, _ in recs]
    by_t = {s["currentTick"]: s for s in st}
    D = st[0]["constants"]["meteorDelay"]
    o = dict(calm=defaultdict(Counter), threat=defaultdict(Counter), by_d0=defaultdict(Counter),
             by_turns=defaultdict(Counter), end_d=Counter(), straight=[0, 0],
             pred=defaultdict(lambda: [0, 0, 0, 0]),          # (k, species, predictor) -> hits, n
             edge=defaultdict(Counter), scroll=Counter(), row0=[0, 0])
    hist = defaultdict(lambda: deque(maxlen=4))
    for s in st:
        t = s["currentTick"]
        for d in s["dinosaurs"]:
            hist[d["id"]].append((t, pos(d)))
        s2 = by_t.get(t + 1)
        if not s2:
            continue
        nxt = {d["id"]: pos(d) for d in s2["dinosaurs"]}
        ox, oy = origin(s)
        blasts = [((m["target"]["x"], m["target"]["y"]), m["turnsUntilImpact"],
                   {(q["x"], q["y"]) for q in m["impactedTiles"]}) for m in s["meteors"]]
        shift = origin(s2)[1] > oy
        for d in s["dinosaurs"]:
            i, sp, p0 = d["id"], SP.get(d["name"], "?"), pos(d)
            # Q25 scroll: dinos on the top row just before the window shifts
            if shift and p0[1] - oy == 0:
                o["row0"][1] += 1
                o["row0"][0] += i in died and died[i]["cause"] == "scroll"
            if i not in nxt:
                continue
            p1 = nxt[i]
            mv = (p1[0] - p0[0], p1[1] - p0[1])
            hit = [b for b in blasts if p0 in b[2]]
            if hit:
                c, turns, _ = min(hit, key=lambda b: b[1])
                k = kind(p0, p1, c)
                o["threat"][sp][k] += 1
                o["by_d0"][manh(p0, c)][k] += 1
                o["by_turns"][turns][k] += 1
            elif not any(manh(p0, b[0]) <= 5 for b in blasts):
                o["calm"][sp][DIRS.get(mv, "other")] += 1
                row = p0[1] - oy
                band = "top 0-2" if row <= 2 else "bottom 17-19" if row >= 17 else "middle"
                o["edge"][(sp, band)][DIRS.get(mv, "other")] += 1
            # Q22/23 prediction 1-4 ticks ahead
            h = list(hist[i])
            if len(h) >= 2 and h[-2][0] == t - 1:
                v = (p0[0] - h[-2][1][0], p0[1] - h[-2][1][1])
                maj = Counter((h[j + 1][1][0] - h[j][1][0], h[j + 1][1][1] - h[j][1][1]) for j in range(len(h) - 1)
                              if h[j + 1][0] == h[j][0] + 1).most_common(1)[0][0]
                for k in range(1, 5):
                    sk = by_t.get(t + k)
                    pk = next((pos(e) for e in sk["dinosaurs"] if e["id"] == i), None) if sk else None
                    if pk is None:
                        continue
                    for name, guess in (("stay", p0), ("last move", (p0[0] + k * v[0], p0[1] + k * v[1])),
                                        ("majority of last 3", (p0[0] + k * maj[0], p0[1] + k * maj[1]))):
                        r = o["pred"][(k, sp, name)]
                        r[0] += guess == pk; r[1] += 1; r[2] += manh(guess, pk)
        # Q21 end distance: meteors first seen at t with turns == D
        for c, turns, tiles in blasts:
            if turns != D:
                continue
            sL = by_t.get(t + D - 1)
            if not sL:
                continue
            posL = {e["id"]: pos(e) for e in sL["dinosaurs"]}
            for d in s["dinosaurs"]:
                p0 = pos(d)
                if p0 in tiles and d["id"] in posL and posL[d["id"]] not in tiles:
                    de = manh(posL[d["id"]], c)
                    o["end_d"][de] += 1
                    o["straight"][1] += 1
                    o["straight"][0] += de >= manh(p0, c) + (D - 1) - 1
    o["scroll"] = Counter(v["sp"] for v in died.values() if v["cause"] == "scroll")
    o["deaths"] = Counter(v["sp"] for v in died.values())
    o["pred"] = dict(o["pred"])
    return o


def merge_all(G):
    m = dict(calm=defaultdict(Counter), threat=defaultdict(Counter), by_d0=defaultdict(Counter),
             by_turns=defaultdict(Counter), edge=defaultdict(Counter), end_d=Counter(), straight=[0, 0],
             pred=defaultdict(lambda: [0, 0, 0, 0]), scroll=Counter(), deaths=Counter(), row0=[0, 0])
    for g in G:
        for k in ("calm", "threat", "by_d0", "by_turns", "edge"):
            for kk, c in g[k].items():
                m[k][kk].update(c)
        for k in ("end_d", "scroll", "deaths"):
            m[k].update(g[k])
        for k in ("straight", "row0"):
            m[k] = [x + y for x, y in zip(m[k], g[k])]
        for kk, r in g["pred"].items():
            m["pred"][kk] = [x + y for x, y in zip(m["pred"][kk], r)]
    return m


def shares(c, keys):
    n = sum(c.values())
    return " ".join(f"{k} {pct(c[k], n)}" for k in keys) + f" (n={n})"


def main():
    a, files = args_and_files(__doc__)
    G = run_games(one, files, a.workers)
    m = merge_all(G)
    rep = Report("movement")
    rep.line(f"movement.py | {len(G)} games from {a.logs}")
    MOVES = ("stay", "up", "down", "left", "right")
    FLEE = ("away", "sideways", "toward", "stay")

    rep.q(16, "How does each species move with no threat?", "calm moves per species (down = +y = the way the window scrolls):")
    for sp in "SVTR":
        rep.d(f"{sp}: " + shares(m["calm"][sp], MOVES))

    rep.q(17, "How does each species react to a meteor launched near it?", "first move while inside a falling blast:")
    for sp in "SVTR":
        rep.d(f"{sp}: " + shares(m["threat"][sp], FLEE) +
              f" | calm stay {pct(m['calm'][sp]['stay'], sum(m['calm'][sp].values()))}")

    rep.q(18, "Does reaction depend on distance from the blast centre?", "by distance from the centre (0 = on it):")
    for dd in sorted(m["by_d0"]):
        rep.d(f"{dd}: " + shares(m["by_d0"][dd], FLEE))

    rep.q(19, "Does reaction change with 4 / 3 / 2 / 1 ticks to impact?", "by turns until impact:")
    for tt in sorted(m["by_turns"], reverse=True):
        rep.d(f"{tt}: " + shares(m["by_turns"][tt], FLEE))

    allt = Counter()
    for c in m["threat"].values():
        allt.update(c)
    rep.q(20, "When fleeing: straight away, sideways, toward, or stay?", shares(allt, FLEE))

    ne = sum(m["end_d"].values())
    rep.q(21, "How far from the centre does a fleeing dino end up?",
          ", ".join(f"d={k}: {pct(v, ne)}" for k, v in sorted(m["end_d"].items())[:8]) +
          f" | ran straight away {pct(*m['straight'])}" if ne else "no escapes found")

    rep.q(22, "How predictable is a dino 1-4 ticks ahead, by species?",
          "exact tile hit rate, predictor 'last move' (stay in brackets), mean error in tiles:")
    for sp in "SVTR":
        parts = []
        for k in range(1, 5):
            h, n, e, _ = m["pred"].get((k, sp, "last move"), [0, 0, 0, 0])
            hs, ns, _, _ = m["pred"].get((k, sp, "stay"), [0, 0, 0, 0])
            if n:
                parts.append(f"{k}t: {pct(h, n)} ({pct(hs, ns)}) err {e / n:.1f}")
        rep.d(f"{sp}: " + " | ".join(parts))

    rep.q(23, "How much does movement history help prediction?", "exact hit rate 4 ticks ahead, all species:")
    for name in ("stay", "last move", "majority of last 3"):
        h = sum(m["pred"].get((4, sp, name), [0, 0])[0] for sp in "SVTR")
        n = sum(m["pred"].get((4, sp, name), [0, 0])[1] for sp in "SVTR")
        rep.d(f"{name:20s} {pct(h, n)}")

    rep.q(24, "How does the window edge affect movement per species?", "calm moves by screen row band:")
    for sp in "SVTR":
        for band in ("top 0-2", "middle", "bottom 17-19"):
            c = m["edge"].get((sp, band))
            if c:
                rep.d(f"{sp} {band:13s}: " + shares(c, MOVES))

    ns = sum(m["scroll"].values())
    rep.q(25, "Which dinos die by scrolling, and can it be predicted?",
          f"{ns} scroll deaths (" + ", ".join(f"{sp} {m['scroll'][sp]} of {m['deaths'][sp]} deaths" for sp in "SVTR") +
          f"); dinos on the top row just before a shift die by scroll {pct(*m['row0'])}")
    rep.save()


if __name__ == "__main__":
    main()
