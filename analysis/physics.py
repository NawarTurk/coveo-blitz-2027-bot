#!/usr/bin/env python3
"""
physics.py - meteors, lava and scoring rules.  Answers Q37-Q45 of results/research_questions.md.

For every meteor (first seen with turnsUntilImpact == meteorDelay) we follow the dinos inside its blast
at launch until the landing tick (turnsUntilImpact == 1: the blast hits before dinos move).
free escape tiles = open tiles the dino can reach before impact that are outside the blast.

Usage:  python analysis/physics.py --logs local_game_logs [--games 300]
Writes: results/analysis/physics.txt
"""
from collections import Counter, defaultdict

from common import Report, args_and_files, run_games, load, events, manh, pos, origin, tile, val, SP, pct, mean, med

BUCKETS = [(0, 0), (1, 3), (4, 7), (8, 14), (15, 99)]


def bucket(x):
    return next(f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in BUCKETS if lo <= x <= hi)


def one(path):
    recs = load(path)
    if not recs:
        return None
    born, died, waves, alive = events(recs)
    st = [s for s, _ in recs]
    by_t = {s["currentTick"]: s for s in st}
    D = st[0]["constants"]["meteorDelay"]
    R = st[0]["constants"]["meteorRadius"]
    o = dict(exits=defaultdict(lambda: [0, 0]), occ=defaultdict(lambda: [0, 0]), obst=defaultdict(lambda: [0, 0]),
             fate=Counter(), end_d=Counter(), follow3=[0, 0], own=[0, 0], gap=defaultdict(lambda: [0, 0]),
             lava=defaultdict(lambda: [0, 0]), single=[], multi=[], base=[])
    landings = defaultdict(list)                      # landing tick -> [(centre, tiles)]
    launches = []
    seen = set()
    for s in st:
        t = s["currentTick"]
        for m in s["meteors"]:
            L = t + m["turnsUntilImpact"] - 1
            c = (m["target"]["x"], m["target"]["y"])
            if (c, L) in seen:
                continue
            seen.add((c, L))
            tiles = frozenset((q["x"], q["y"]) for q in m["impactedTiles"])
            landings[L].append((c, tiles))
            if m["turnsUntilImpact"] == D:
                launches.append((t, L, c, tiles))
    for t0, L, c, tiles in launches:
        s0, sL = by_t.get(t0), by_t.get(L)
        if not s0 or not sL:
            continue
        posL = {d["id"]: pos(d) for d in sL["dinosaurs"]}
        others = {pos(d) for d in s0["dinosaurs"]}
        ox, oy = origin(s0)
        lava = {(ox + j, oy + i) for i, row in enumerate(s0["map"]["tiles"]) for j, tl in enumerate(row) if tl["hasLava"]}
        moves = L - t0
        for d in s0["dinosaurs"]:
            p0 = pos(d)
            if p0 not in tiles:
                continue
            reach = [(p0[0] + a, p0[1] + b) for a in range(-moves, moves + 1) for b in range(-moves, moves + 1)
                     if abs(a) + abs(b) <= moves]
            free = [q for q in reach if tile(s0, q) and not (tile(s0, q)["isImpassable"] or tile(s0, q)["isMountain"]
                                                            or tile(s0, q)["hasLava"])]
            esc = [q for q in free if q not in tiles]
            caught = d["id"] in posL and posL[d["id"]] in tiles
            gone = d["id"] not in posL
            key = "caught" if caught else "died another way" if gone else "escaped"
            o["fate"][key] += 1
            if gone:
                continue
            o["exits"][bucket(len(esc))][0] += caught; o["exits"][bucket(len(esc))][1] += 1
            # Q38 other dinos on escape tiles
            occ = sum(1 for q in esc if q in others and q != p0)
            k = "some escape tiles occupied" if occ else "none occupied"
            o["occ"][(bucket(len(esc)), k)][0] += caught; o["occ"][(bucket(len(esc)), k)][1] += 1
            # Q39 obstacle type near the dino
            near = [(p0[0] + a, p0[1] + b) for a in range(-2, 3) for b in range(-2, 3) if abs(a) + abs(b) <= 2]
            tags = []
            if any(tile(s0, q) is None for q in near):
                tags.append("window edge")
            if any(tile(s0, q) and (tile(s0, q)["isImpassable"] or tile(s0, q)["isMountain"]) for q in near):
                tags.append("wall / mountain")
            if any(q in lava for q in near):
                tags.append("lava")
            for tg in tags or ["open ground"]:
                o["obst"][tg][0] += caught; o["obst"][tg][1] += 1
            # Q43 lava inside the reach area
            lk = ("lava in reach" if any(q in lava for q in reach) else "no lava") + f" | exits {bucket(len(esc))}"
            o["lava"][lk][0] += caught; o["lava"][lk][1] += 1
            if not caught:
                pe = posL[d["id"]]
                o["end_d"][manh(pe, c)] += 1
                # Q41: a second blast 3 tiles out along the dino's axis vs at its start tile, does it cover the end tile?
                vx, vy = p0[0] - c[0], p0[1] - c[1]
                if (vx, vy) != (0, 0):
                    ax, ay = ((vx > 0) - (vx < 0), 0) if abs(vx) >= abs(vy) else (0, (vy > 0) - (vy < 0))
                    q = (p0[0], p0[1])
                    while manh(q, c) < 3:
                        q = (q[0] + ax, q[1] + ay)
                    o["follow3"][0] += manh(pe, q) <= R; o["follow3"][1] += 1
                    o["own"][0] += manh(pe, p0) <= R; o["own"][1] += 1
                # Q42: follow-up meteors landing 1-3 ticks later near the first
                for g in (1, 2, 3):
                    for c2, t2 in landings.get(L + g, []):
                        if manh(c2, c) > 2 * R + 3:
                            continue
                        sg = by_t.get(L + g)
                        pg = next((pos(e) for e in sg["dinosaurs"] if e["id"] == d["id"]), None) if sg else None
                        o["gap"][g][1] += 1
                        o["gap"][g][0] += pg is not None and pg in t2
                        break
    # Q44/45 scoring: score change on ticks with meteor kills vs ticks with no deaths
    deaths_at = defaultdict(list)
    for i, v in died.items():
        deaths_at[v["t"]].append(v)
    for s in st:
        t = s["currentTick"]
        p = by_t.get(t - 1)
        if not p:
            continue
        ds = s["score"] - p["score"]
        dl = deaths_at.get(t, [])
        if not dl:
            o["base"].append(ds)
        elif all(v["cause"] == "meteor" for v in dl):
            # group kills by the meteor that hit them
            groups = defaultdict(list)
            for v in dl:
                c = next((cc for cc, tl in landings.get(t - 1, []) if v["p"] in tl), None)
                groups[c].append(v)
            if len(groups) == 1:
                ks = list(groups.values())[0]
                if len(ks) == 1:
                    o["single"].append((ds, ks[0]["age"]))
                else:
                    o["multi"].append((ds, [v["age"] for v in ks]))
    for k in ("exits", "occ", "obst", "gap", "lava"):
        o[k] = dict(o[k])                             # lambdas cannot be sent back from worker processes
    return o


def main():
    a, files = args_and_files(__doc__)
    G = run_games(one, files, a.workers)
    rep = Report("physics")
    rep.line(f"physics.py | {len(G)} games from {a.logs}")
    m = dict(fate=Counter(), end_d=Counter(), follow3=[0, 0], own=[0, 0], single=[], multi=[], base=[])
    for k in ("exits", "occ", "obst", "gap", "lava"):
        m[k] = defaultdict(lambda: [0, 0])
    for g in G:
        m["fate"].update(g["fate"]); m["end_d"].update(g["end_d"])
        for k in ("follow3", "own"):
            m[k] = [x + y for x, y in zip(m[k], g[k])]
        for k in ("single", "multi", "base"):
            m[k] += g[k]
        for k in ("exits", "occ", "obst", "gap", "lava"):
            for kk, v in g[k].items():
                m[k][kk] = [x + y for x, y in zip(m[k][kk], v)]
    order = [bucket(lo) for lo, _ in BUCKETS]

    rep.q(37, "How do free escape tiles affect catch probability?",
          " | ".join(f"{b}: {pct(*m['exits'][b])} (n={m['exits'][b][1]})" for b in order if m["exits"][b][1]))
    rep.q(38, "Do other dinos block escapes enough to matter?", "catch rate by free escape tiles, with / without other dinos on them:")
    for b in order:
        x, y = m["occ"][(b, "some escape tiles occupied")], m["occ"][(b, "none occupied")]
        if x[1] or y[1]:
            rep.d(f"exits {b:5s}: occupied {pct(*x)} (n={x[1]}) | free {pct(*y)} (n={y[1]})")
    rep.q(39, "Do walls, mountains, lava, edges raise catch probability?",
          " | ".join(f"{k}: {pct(*v)} (n={v[1]})" for k, v in sorted(m["obst"].items())))
    nf = sum(m["fate"].values())
    rep.q(40, "When a dino is inside a falling blast, how often does it survive?",
          ", ".join(f"{k} {pct(v, nf)}" for k, v in m["fate"].most_common()) + f" (n={nf})")
    ne = sum(m["end_d"].values())
    rep.q(41, "Where does a dino go after escaping; where should a second meteor land?",
          "distance from the first centre at impact: " +
          ", ".join(f"d={k}: {pct(v, ne)}" for k, v in sorted(m["end_d"].items())[:7]) +
          f" | a blast 3 tiles out along its axis covers it {pct(*m['follow3'])} vs one on its start tile {pct(*m['own'])}")
    rep.q(42, "Catch rate of a follow-up meteor by gap: 1, 2, 3 ticks?",
          " | ".join(f"{g} tick{'s' if g > 1 else ''} later: {pct(*m['gap'][g])} of escapees caught (n={m['gap'][g][1]})"
                     for g in (1, 2, 3)))
    rep.q(43, "Does lava block escape routes (funnel), and how much does it raise catch probability?",
          "catch rate with / without lava inside the dino's reach, same escape-tile band:")
    for b in order:
        x, y = m["lava"][f"lava in reach | exits {b}"], m["lava"][f"no lava | exits {b}"]
        if x[1]:
            rep.d(f"exits {b:5s}: lava {pct(*x)} (n={x[1]}) | no lava {pct(*y)} (n={y[1]})")
    base = med(m["base"]) or 0
    ratios = [(ds - base) / val(age) for ds, age in m["single"] if val(age) > 0]
    ratios1 = [(ds - base) / val(age + 1) for ds, age in m["single"] if val(age + 1) > 0]
    rep.q(44, "Exact age-to-points formula?",
          f"single meteor kills (n={len(ratios)}): (score change - {base} per tick) / [160/(1+age/30)] median "
          f"{med([round(r, 3) for r in ratios])} (with age+1: {med([round(r, 3) for r in ratios1])}); 1.000 = formula exact"
          if ratios else "no clean single kills found")
    for k in (2, 3):
        mm = [(ds - base) / sum(val(a_) for a_ in ages) for ds, ages in m["multi"] if len(ages) == k]
        if mm:
            rep.d(f"{k} kills by one meteor: points / sum of single values = {med([round(x, 3) for x in mm])} (n={len(mm)})")
    mm2 = [(ds - base) / sum(val(a_) for a_ in ages) for ds, ages in m["multi"] if len(ages) == 2]
    rep.q(45, "Exact multi-kill multiplier?",
          f"2 kills: x{med([round(x, 3) for x in mm2])} (rule 1 + 0.5 per extra kill = x1.5)" if mm2 else "no clean multi-kills found")
    rep.save()


if __name__ == "__main__":
    main()
