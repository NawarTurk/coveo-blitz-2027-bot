#!/usr/bin/env python3
"""
MODE REPORT: how well does the bot do in each board situation ("mode")?

Works on ANY team log with T| trace lines (N4 ... N12), so old games give the baseline
and N12 games can be compared mode by mode against it.

Every tick is labelled from the board only (same rules N12 uses):
  primary mode (first match wins)
    straggler   1-2 dinos on the board
    rex         a T-Rex is ADJACENT (distance 1) to another dino
    pack        2+ Triceratops within 4 tiles of each other
    crowd       6+ dinos on the board
    normal      anything else
    empty       0 dinos (between waves)
  flags (can overlap with any mode)
    trapped     some dino has <= 3 free escape tiles for its best blast centre
    rex_risk    a T-Rex is 2 tiles from another dino
    pack        Triceratops cluster exists (even when another mode won)

Per mode it reports: share of ticks, shots, kills, kills/shot, zero-kill %, points from kills,
how long a stay in the mode lasts, and (straggler) how a stay ended.
Kills are matched to the shot whose blast covered the dino on its landing tick.

Usage:
  python analysis/mode_report.py logs/server_game_logs/*.txt
  python analysis/mode_report.py logs/server_game_logs/*.txt --by-bot        # one table per bot
Writes: results/analysis/server/mode_report.txt (also printed)
"""
import argparse, glob, hashlib, json, os, re
from collections import Counter, defaultdict

R, DELAY, MOVES = 2, 4, 3
FOOT = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]
REACH = [(a, b) for a in range(-MOVES, MOVES + 1) for b in range(-MOVES, MOVES + 1) if abs(a) + abs(b) <= MOVES]
MODES = ["straggler", "rex", "pack", "crowd", "normal", "empty"]
FLAGS = ["trapped", "rex_risk", "pack"]
TRAP_MAX, PACK_DIST, CROWD_MIN = 3, 4, 6


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def value(age):
    return 160 / (1 + age / 30)


def read(path):
    recs, bot, final = [], None, None
    for l in open(path, errors="ignore"):
        if l.startswith("T|"):
            try:
                recs.append(json.loads(l[2:]))
            except Exception:
                pass
            continue
        m = re.search(r"\[(N\d+)\] loaded", l)
        if m:
            bot = m.group(1)                       # last "[Nxx] loaded" line = the actual bot
        m = re.search(r"Final score: (\d+)", l)
        if m:
            final = int(m.group(1))
    recs.sort(key=lambda r: r["t"])
    return recs, bot, final


def label(r, blocked, lava):
    """primary mode + flags for one tick"""
    ds = [(d[1], (d[2], d[3])) for d in r["d"]]
    n = len(ds)
    ox, oy = r["o"]

    def free(p):
        return ox <= p[0] < ox + 20 and oy <= p[1] < oy + 20 and p not in blocked and p not in lava

    occupied = {p for _, p in ds}
    trapped = False
    for _, p in ds:
        reach = [(p[0] + a, p[1] + b) for a, b in REACH]
        reach = [q for q in reach if free(q)]
        best = min(sum(1 for q in reach if manh(q, (p[0] + a, p[1] + b)) > R) for a, b in FOOT)
        if best <= TRAP_MAX:
            trapped = True
            break
    rex = [p for s, p in ds if s == "R"]
    others = [p for s, p in ds]
    rex_adj = any(manh(x, o) == 1 for x in rex for o in others if o != x)
    rex_risk = any(manh(x, o) == 2 for x in rex for o in others if o != x)
    tri = [p for s, p in ds if s == "T"]
    pack = any(manh(a, b) <= PACK_DIST for i, a in enumerate(tri) for b in tri[i + 1:])
    if n == 0:
        mode = "empty"
    elif n <= 2:
        mode = "straggler"
    elif rex_adj:
        mode = "rex"
    elif pack:
        mode = "pack"
    elif n >= CROWD_MIN:
        mode = "crowd"
    else:
        mode = "normal"
    return mode, {"trapped": trapped, "rex_risk": rex_risk or rex_adj, "pack": pack}


def game(recs):
    """one row per tick: mode, flags, shot target, kills/points credited to that shot"""
    blocked, lava, rows = set(), set(), []
    by_t = {r["t"]: r for r in recs}
    for r in recs:
        for y, row in (r.get("tr") or {}).items():
            for x, (elev, fl) in enumerate(row):
                if fl & 3:
                    blocked.add((r["o"][0] + x, int(y)))
        if "lava" in r:
            lava = {tuple(p) for p in r["lava"]}
        mode, flags = label(r, blocked, lava)
        a = r.get("a") or []
        shot = (a[1], a[2]) if a and a[0] == "M" else None
        kills, pts = 0, 0.0
        if shot:
            L = r["t"] + DELAY
            pre, post = by_t.get(L), by_t.get(L + 1)
            if pre and post:
                gone = {d[0] for d in pre["d"]} - {d[0] for d in post["d"]}
                eaten = {e[1] for e in post.get("ev", []) if e[0] == "D" and e[2] != "m"}
                hit = [d for d in pre["d"] if d[0] in gone and d[0] not in eaten and manh((d[2], d[3]), shot) <= R]
                kills, pts = len(hit), sum(value(d[4] + 1) for d in hit)
        deaths = Counter(e[2] for e in r.get("ev", []) if e[0] == "D")
        tsp, extra = None, False                   # straggler shot: species of the targeted dino, meteor already on it?
        if shot and mode == "straggler":
            d = min(r["d"], key=lambda d: manh((d[2], d[3]), shot))
            tsp = d[1]
            extra = any(manh((d[2], d[3]), (m[0], m[1])) <= R + 1 for m in r["m"])
        rows.append(dict(t=r["t"], mode=mode, flags=flags, n=len(r["d"]), shot=shot, kills=kills, pts=pts,
                         eaten=deaths.get("e", 0), score=r["s"], sp=[d[1] for d in r["d"]], tsp=tsp, extra=extra))
    return rows


def summarise(rows_by_game, title):
    out = [f"\n=== {title} | {len(rows_by_game)} games ==="]
    tot = sum(len(g) for g in rows_by_game)
    st = {m: Counter() for m in MODES + ["ALL"]}
    stays = defaultdict(list)
    ends = Counter()
    fl = {f: Counter() for f in FLAGS}
    for rows in rows_by_game:
        cur, start = None, 0
        for i, x in enumerate(rows):
            for key in (x["mode"], "ALL"):
                s = st[key]
                s["ticks"] += 1
                s["shots"] += bool(x["shot"])
                s["kills"] += x["kills"]
                s["zero"] += bool(x["shot"]) and x["kills"] == 0
                s["pts"] += x["pts"]
                s["eaten"] += x["eaten"]
            for f in FLAGS:
                if x["flags"][f]:
                    c = fl[f]
                    c["ticks"] += 1
                    c["shots"] += bool(x["shot"])
                    c["kills"] += x["kills"]
            if x["mode"] != cur:
                if cur is not None:
                    stays[cur].append(i - start)
                    if cur == "straggler":
                        ends["cleared (board empty)" if x["mode"] == "empty" else "next wave arrived"] += 1
                cur, start = x["mode"], i
        if cur is not None:
            stays[cur].append(len(rows) - start)
            if cur == "straggler":
                ends["game ended"] += 1
    out.append(f"{'mode':10} {'ticks':>6} {'share':>6} {'shots':>6} {'shots/t':>7} {'kills':>6} "
               f"{'kill/shot':>9} {'zero%':>6} {'points':>7} {'pts/tick':>8} {'eaten':>5} {'avg stay':>8} {'stays':>5}")
    for m in MODES + ["ALL"]:
        s = st[m]
        if not s["ticks"]:
            continue
        v = stays.get(m, [])
        out.append(f"{m:10} {s['ticks']:6d} {s['ticks'] / tot:6.1%} {s['shots']:6d} {s['shots'] / s['ticks']:7.2f} "
                   f"{s['kills']:6d} {s['kills'] / max(s['shots'], 1):9.3f} {s['zero'] / max(s['shots'], 1):6.0%} "
                   f"{s['pts']:7.0f} {s['pts'] / s['ticks']:8.1f} {s['eaten']:5d} "
                   f"{(sum(v) / len(v) if v else 0):8.1f} {len(v):5d}")
    out.append("flags (overlap with modes):")
    for f in FLAGS:
        c = fl[f]
        out.append(f"  {f:9} ticks {c['ticks']:6d} ({c['ticks'] / tot:.0%}) | shots {c['shots']} | "
                   f"kills {c['kills']} = {c['kills'] / max(c['shots'], 1):.3f}/shot")
    if ends:
        out.append("straggler stays ended by: " + ", ".join(f"{k} {v}" for k, v in ends.most_common()))
    v = sorted(stays.get("straggler", []))
    if v:
        out.append(f"straggler stay length: median {v[len(v) // 2]} | 80% {v[int(len(v) * .8)]} | max {v[-1]} ticks")
    out += straggler_detail(rows_by_game)
    return out


def rate(k, n):
    return f"{k / n:.3f}" if n else "  -  "


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0


def straggler_detail(rows_by_game):
    """1 vs 2 dinos, species, first vs extra shots, last-dino chases"""
    out = ["straggler detail:"]
    by_n = {1: Counter(), 2: Counter()}
    by_sp = defaultdict(Counter)
    kind = {"first": Counter(), "extra": Counter()}
    chases = defaultdict(list)                     # species -> [(ticks, shots, cleared?)]
    for rows in rows_by_game:
        chase = None
        for x in rows + [dict(mode="end", n=-1, sp=[], shot=None, kills=0)]:
            if x["mode"] == "straggler":
                c = by_n[x["n"]]
                c["ticks"] += 1; c["shots"] += bool(x["shot"]); c["kills"] += x["kills"]
                for sp in set(x["sp"]):
                    by_sp[sp]["ticks"] += 1
                if x["shot"]:
                    b = by_sp[x["tsp"]]
                    b["shots"] += 1; b["kills"] += x["kills"]
                    k = kind["extra" if x["extra"] else "first"]
                    k["shots"] += 1; k["kills"] += x["kills"]
            if x["n"] == 1:                            # last dino on the board
                if chase is None:
                    chase = dict(sp=x["sp"][0], ticks=0, shots=0)
                chase["ticks"] += 1
                chase["shots"] += bool(x["shot"])
            elif chase is not None:
                chases[chase["sp"]].append((chase["ticks"], chase["shots"], x["n"] == 0))
                chase = None
    for n in (1, 2):
        c = by_n[n]
        out.append(f"  {n} dino{'s' if n > 1 else ' '} left: ticks {c['ticks']:5d} | shots {c['shots']:5d} | "
                   f"kills/shot {rate(c['kills'], c['shots'])}")
    for k in ("first", "extra"):
        c = kind[k]
        what = "no meteor on that dino yet" if k == "first" else "a meteor already falling on it"
        out.append(f"  {k:5} shots ({what}): {c['shots']:5d} | kills/shot {rate(c['kills'], c['shots'])}")
    out.append("  by species (ticks it was a straggler | shots aimed at it | kills/shot):")
    for sp in "SVTR":
        c = by_sp.get(sp)
        if c:
            out.append(f"    {sp}: ticks {c['ticks']:5d} | shots {c['shots']:5d} | kills/shot {rate(c['kills'], c['shots'])}")
    out.append("  last-dino chases (1 dino on the board; cleared = it died and the board emptied):")
    for sp in "SVTR":
        v = chases.get(sp)
        if v:
            cl = [x for x in v if x[2]]
            out.append(f"    {sp}: {len(v):4d} chases | cleared {len(cl) / len(v):4.0%} | median ticks {med([x[0] for x in v]):3d} "
                       f"| median shots to finish {med([x[1] for x in cl]):3d} | worst {max(x[0] for x in v)} ticks")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--by-bot", action="store_true", help="one table per bot as well")
    ap.add_argument("--out", default="results/analysis/server/mode_report.txt")
    a = ap.parse_args()
    paths = sorted({p for g in a.logs for p in glob.glob(g)})
    games, seen, lines = defaultdict(list), set(), []
    for p in paths:
        recs, bot, final = read(p)
        if len(recs) < 100:
            continue
        key = hashlib.md5(json.dumps(recs[:5]).encode()).hexdigest()
        if key in seen:                                   # same game uploaded twice
            continue
        seen.add(key)
        rows = game(recs)
        games[bot or "?"].append(rows)
        has_tr = any("tr" in r for r in recs)
        lines.append(f"{os.path.basename(p)[:30]:30} bot {bot or '?':4} score {final or recs[-1]['s']:6} "
                     f"{'' if has_tr else '(no terrain in log: trapped flag less accurate)'}")
    allg = [g for v in games.values() for g in v]
    if not allg:
        print("no T| traces found")
        return
    out = ["MODE REPORT", *lines]
    out += summarise(allg, "ALL BOTS")
    if a.by_bot:
        for b in sorted(games):
            out += summarise(games[b], f"bot {b}")
    txt = "\n".join(out)
    print(txt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
