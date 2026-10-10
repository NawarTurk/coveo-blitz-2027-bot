#!/usr/bin/env python3
"""
species.py - species-specific behaviour and the site's strategy claims.  Answers Q26-Q36.

"lift" = how much more often a dino picks a move than a random choice among its open moves would
(1.0 = no preference). Open moves = stay + the 4 neighbours that are inside the window and not
impassable / mountain.

Usage:  python analysis/species.py --logs local_game_logs [--games 300]
Writes: results/analysis/species.txt
"""
from collections import Counter, defaultdict

from common import Report, args_and_files, run_games, load, events, manh, pos, origin, tile, SP, pct, mean

STEPS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def open_moves(s, p):
    out = []
    for a, b in STEPS:
        q = (p[0] + a, p[1] + b)
        t = tile(s, q)
        if t is not None and not t["isImpassable"] and not t["isMountain"]:
            out.append(q)
    return out


def one(path):
    recs = load(path)
    if not recs:
        return None
    born, died, waves, alive = events(recs)
    st = [s for s, _ in recs]
    by_t = {s["currentTick"]: s for s in st}
    o = dict(quad=Counter(), hi_d=[0, 0], hi_t=[0, 0], rot=Counter(), other_deaths=[0, 0, 0],
             tri_up=[0.0, 0.0, 0], tri_dz=[0.0, 0.0], spread=[], cen_err=[0.0, 0.0, 0],
             rap=defaultdict(lambda: [0.0, 0.0, 0]), rap_threat=[0.0, 0.0, 0],
             eat_dist=Counter(), eat_sp=Counter(), adj=[0, 0], lava=defaultdict(lambda: [0.0, 0.0, 0]))
    cent = {}
    for s in st:
        t = s["currentTick"]
        ox, oy = origin(s)
        s2 = by_t.get(t + 1)
        danger = {(q["x"], q["y"]) for m in s["meteors"] for q in m["impactedTiles"]}
        corpses = [(c["x"], c["y"]) for c in s.get("corpses", [])]
        lava = [(ox + j, oy + i) for i, row in enumerate(s["map"]["tiles"]) for j, tl in enumerate(row) if tl["hasLava"]]
        # Q26 quadrants + elevation 7+
        for i, row in enumerate(s["map"]["tiles"]):
            for tl in row:
                o["hi_t"][0] += tl["elevation"] >= 7; o["hi_t"][1] += 1
        for d in s["dinosaurs"]:
            x, y = pos(d)
            o["quad"][("N" if y - oy < 10 else "S") + ("W" if x - ox < 10 else "E")] += 1
            tl = tile(s, (x, y))
            if tl:
                o["hi_d"][0] += tl["elevation"] >= 7; o["hi_d"][1] += 1
        tri = [d for d in s["dinosaurs"] if d["name"] == "Triceratops"]
        if len(tri) >= 2:
            cx, cy = mean([pos(d)[0] for d in tri]), mean([pos(d)[1] for d in tri])
            o["spread"].append(mean([abs(pos(d)[0] - cx) + abs(pos(d)[1] - cy) for d in tri]))
            cent[t] = (cx, cy)
        if not s2:
            continue
        nxt = {d["id"]: pos(d) for d in s2["dinosaurs"]}
        ids2 = set(nxt)
        # Q35 eating: adjacency to a T-Rex and what happened
        rex = [pos(d) for d in s["dinosaurs"] if d["name"] == "Tyrannosaurus"]
        for d in s["dinosaurs"]:
            if d["name"] == "Tyrannosaurus" or not rex:
                continue
            dm = min(manh(pos(d), r) for r in rex)
            if dm == 1:
                o["adj"][1] += 1
                o["adj"][0] += d["id"] in died and died[d["id"]]["cause"] == "eaten" and died[d["id"]]["t"] == t + 1
        for i, v in died.items():
            if v["t"] == t + 1 and v["cause"] == "eaten":
                o["eat_sp"][v["sp"]] += 1
                o["eat_dist"][min((manh(v["p"], r) for r in rex), default=99)] += 1
            if v["t"] == t + 1 and v["cause"] == "other":
                o["other_deaths"][0] += 1
                landed = [(m["target"]["x"], m["target"]["y"]) for m in s["meteors"] if m["turnsUntilImpact"] == 1]
                o["other_deaths"][1] += any(manh(v["p"], c) <= 5 for c in landed)
        # moves
        for d in s["dinosaurs"]:
            i, sp, p0 = d["id"], SP.get(d["name"], "?"), pos(d)
            if i not in nxt:
                continue
            p1 = nxt[i]
            opts = open_moves(s, p0)
            if p1 not in opts:
                opts.append(p1)
            threatened = p0 in danger
            t0 = tile(s, p0)
            # Q27 rotation around the pack centroid
            if sp == "T" and len(tri) >= 2 and p1 != p0:
                rx, ry = p0[0] - cx, p0[1] - cy
                vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                cr = rx * vy - ry * vx                  # screen y points down: < 0 = counter-clockwise on screen
                o["rot"]["ccw" if cr < 0 else "cw" if cr > 0 else "none"] += 1
            # Q29/30 Triceratops uphill (calm)
            if sp == "T" and not threatened and t0:
                el = {q: tile(s, q)["elevation"] for q in opts if tile(s, q)}
                if p1 in el:
                    up = [q for q in el if el[q] > t0["elevation"]]
                    o["tri_up"][0] += p1 in up
                    o["tri_up"][1] += len(up) / len(el)
                    o["tri_up"][2] += 1
                    o["tri_dz"][0] += el[p1] - t0["elevation"]
                    o["tri_dz"][1] += mean([el[q] - t0["elevation"] for q in el])
            # Q32-34 raptors and corpses
            if sp == "V" and corpses:
                dc = min(manh(p0, c) for c in corpses)
                if dc > 0:
                    toward = [q for q in opts if min(manh(q, c) for c in corpses) < dc]
                    k = p1 in toward
                    exp = len(toward) / len(opts)
                    if threatened:
                        r = o["rap_threat"]
                    else:
                        r = o["rap"]["1-3" if dc <= 3 else "4-6" if dc <= 6 else "7-12" if dc <= 12 else "13+"]
                    r[0] += k; r[1] += exp; r[2] += 1
            # Q36 lava avoidance (calm)
            if lava and not threatened:
                dl = min(manh(p0, q) for q in lava)
                if 1 <= dl <= 4:
                    closer = [q for q in opts if min(manh(q, l) for l in lava) < dl]
                    r = o["lava"][dl]
                    r[0] += p1 in closer; r[1] += len(closer) / len(opts); r[2] += 1
    o["rap"] = dict(o["rap"]); o["lava"] = dict(o["lava"])
    # Q31 centroid prediction (constant velocity vs stay), from the stored centroids
    for t, c in cent.items():
        if t - 1 in cent and t + 4 in cent:
            v = (c[0] - cent[t - 1][0], c[1] - cent[t - 1][1])
            f = cent[t + 4]
            o["cen_err"][0] += abs(c[0] + 4 * v[0] - f[0]) + abs(c[1] + 4 * v[1] - f[1])
            o["cen_err"][1] += abs(c[0] - f[0]) + abs(c[1] - f[1])
            o["cen_err"][2] += 1
    return o


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def main():
    a, files = args_and_files(__doc__)
    G = run_games(one, files, a.workers)
    rep = Report("species")
    rep.line(f"species.py | {len(G)} games from {a.logs}")
    m = dict(quad=Counter(), rot=Counter(), eat_sp=Counter(), eat_dist=Counter(), rap={}, lava={})
    for k in ("hi_d", "hi_t", "other_deaths", "tri_up", "tri_dz", "cen_err", "rap_threat", "adj"):
        m[k] = [0] * len(G[0][k])
    m["spread"] = []
    for g in G:
        for k in ("quad", "rot", "eat_sp", "eat_dist"):
            m[k].update(g[k])
        for k in ("hi_d", "hi_t", "other_deaths", "tri_up", "tri_dz", "cen_err", "rap_threat", "adj"):
            m[k] = add(m[k], g[k])
        m["spread"] += g["spread"]
        for k in ("rap", "lava"):
            for kk, v in g[k].items():
                m[k][kk] = add(m[k].get(kk, [0, 0, 0]), v)
    nq = sum(m["quad"].values())

    rep.q(26, "Do dinos gather in the NE / near elevation 7+?",
          " ".join(f"{q} {pct(m['quad'][q], nq)}" for q in ("NW", "NE", "SW", "SE")) +
          f" | on elevation 7+: dinos {pct(*m['hi_d'])} vs tiles {pct(*m['hi_t'])}")
    nr = m["rot"]["ccw"] + m["rot"]["cw"]
    rep.q(27, "Do Triceratops herds rotate counter-clockwise?",
          f"counter-clockwise {pct(m['rot']['ccw'], nr)} vs clockwise {pct(m['rot']['cw'], nr)} (n={nr}; claim: 83%)")
    od = m["other_deaths"]
    rep.q(28, 'Does a "double-bounce" shockwave exist?',
          f"unexplained deaths: {od[0]}, of which near a landing meteor: {od[1]} "
          f"({'no sign of a shockwave' if od[1] == 0 else 'check these'})")
    up = m["tri_up"]
    rep.q(29, "Do Triceratops move toward high ground?",
          f"calm Triceratops step uphill {pct(up[0], up[2])} vs {pct(up[1], up[2])} for a random open move "
          f"(lift x{up[0] / max(up[1], 1e-9):.2f})")
    dz = m["tri_dz"]
    rep.q(30, "Does Triceratops movement follow increasing elevation?",
          f"mean elevation change per move {dz[0] / max(up[2], 1):+.2f} vs {dz[1] / max(up[2], 1):+.2f} for a random open move")
    ce = m["cen_err"]
    rep.q(31, "How tight are Triceratops packs; is the centroid predictable 4 ticks ahead?",
          f"mean distance to the pack centre {mean(m['spread']):.1f} tiles; centroid error 4 ticks ahead: "
          f"constant velocity {ce[0] / max(ce[2], 1):.1f} vs stay {ce[1] / max(ce[2], 1):.1f} tiles (n={ce[2]})"
          if m["spread"] else "no Triceratops packs")
    rep.q(32, "Do raptors move toward corpses? Radius?", "calm raptors stepping closer to the nearest corpse, by distance:")
    for kk in ("1-3", "4-6", "7-12", "13+"):
        r = m["rap"].get(kk)
        if r and r[2]:
            rep.d(f"corpse {kk:5s} tiles: {pct(r[0], r[2])} vs random open move {pct(r[1], r[2])} "
                  f"(lift x{r[0] / max(r[1], 1e-9):.2f}, n={r[2]})")
    tot = [sum(m["rap"][k][j] for k in m["rap"]) for j in range(3)]
    rep.q(33, "Is distance to the nearest corpse predictive of raptor moves?",
          f"overall lift x{tot[0] / max(tot[1], 1e-9):.2f} (1.0 = no effect); see the distances above" if tot[2] else "no raptors near corpses")
    rt = m["rap_threat"]
    rep.q(34, "Does corpse attraction stop when a raptor is threatened?",
          f"threatened raptors step toward a corpse {pct(rt[0], rt[2])} vs random {pct(rt[1], rt[2])} "
          f"(lift x{rt[0] / max(rt[1], 1e-9):.2f}, n={rt[2]})" if rt[2] else "no threatened raptors near corpses")
    ne = sum(m["eat_sp"].values())
    rep.q(35, "When exactly does a T-Rex eat another dino?",
          f"{ne} meals; prey: {dict(m['eat_sp'])}; T-Rex distance the tick before: {dict(sorted(m['eat_dist'].items()))}; "
          f"a dino next to a T-Rex is eaten the next tick {pct(*m['adj'])}")
    rep.q(36, "How strongly / from how far do dinos avoid lava?", "calm dinos stepping closer to lava, by distance:")
    for dl in sorted(m["lava"]):
        r = m["lava"][dl]
        rep.d(f"lava {dl} tiles away: {pct(r[0], r[2])} vs random open move {pct(r[1], r[2])} "
              f"(lift x{r[0] / max(r[1], 1e-9):.2f}, n={r[2]})")
    rep.save()


if __name__ == "__main__":
    main()
