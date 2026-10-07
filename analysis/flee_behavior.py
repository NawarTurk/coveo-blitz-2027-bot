#!/usr/bin/env python3
"""
How do dinos react to a falling meteor?  (input for the "cage" heuristic)

For every meteor launch (first seen with turnsUntilImpact == meteorDelay) we follow the dinos that
were INSIDE its 13-tile blast at launch, until the landing tick (turnsUntilImpact == 1, hits before moves).

  1. HIT RATE      kills per meteor, and % of dinos inside the blast at launch that are still inside at impact
  2. REACTION      first move of a threatened dino: away from the centre / sideways / toward / stay
                   (vs the same species when NOT threatened)
  3. ESCAPE PATH   do they run straight away (maximise distance) or wander? distance from centre at impact
  4. WALLS         caught rate vs how many free tiles the dino has around it (cornered dinos = easy kills)
  5. CHAINS        caught rate when a 2nd meteor lands 1-2 ticks later next to the first (a "cage")

Usage:  python flee_behavior.py --logs logs
"""
import argparse, glob, json, os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default="local_game_logs")
ap.add_argument("--workers", type=int, default=os.cpu_count())
args = ap.parse_args()


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def free_tiles(s):
    ox, oy = s["map"]["origin"]["x"], s["map"]["origin"]["y"]
    free = set()
    for i, row in enumerate(s["map"]["tiles"]):
        for j, t in enumerate(row):
            if not (t["isImpassable"] or t["isMountain"] or t["hasLava"]):
                free.add((j + ox, i + oy))
    return free


def read_game(path):
    try:
        recs = [json.loads(l)["state"] for l in open(path) if l.strip()]
    except Exception:
        return None
    if len(recs) < 900:
        return None
    c = recs[0]["constants"]
    D, R = c["meteorDelay"], c["meteorRadius"]
    by_t = {s["currentTick"]: s for s in recs}
    out = {"threat": [], "calm": Counter(), "calm_sp": defaultdict(Counter), "meteors": [], "chain": []}
    seen_m = set()
    landings = []                                      # (landing tick, centre, tiles)
    for s in recs:
        t = s["currentTick"]
        for m in s["meteors"]:
            key = (m["target"]["x"], m["target"]["y"], t + m["turnsUntilImpact"] - 1)
            if key in seen_m:
                continue
            seen_m.add(key)
            if m["turnsUntilImpact"] != D:
                continue
            landings.append((key[2], (key[0], key[1]), frozenset((p["x"], p["y"]) for p in m["impactedTiles"])))
    land_by_t = defaultdict(list)
    for L, cen, tiles in landings:
        land_by_t[L].append((cen, tiles))

    # calm baseline: moves of dinos not inside any falling meteor's blast
    for s in recs:
        t = s["currentTick"]
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        danger = set()
        for m in s["meteors"]:
            danger |= {(p["x"], p["y"]) for p in m["impactedTiles"]}
        npos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in nxt["dinosaurs"]}
        for d in s["dinosaurs"]:
            p = (d["position"]["x"], d["position"]["y"])
            if p in danger or d["id"] not in npos:
                continue
            mv = (npos[d["id"]][0] - p[0], npos[d["id"]][1] - p[1])
            k = "stay" if mv == (0, 0) else "move"
            out["calm"][k] += 1
            out["calm_sp"][d["name"]][k] += 1

    for L, cen, tiles in landings:
        t0 = L - D + 1                                   # first tick the meteor is visible
        s0, sL = by_t.get(t0), by_t.get(L)
        if not s0 or not sL:
            continue
        posL = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in sL["dinosaurs"]}
        killed = sum(1 for p in posL.values() if p in tiles)
        out["meteors"].append({"kills": killed, "n_inside0": sum((d["position"]["x"], d["position"]["y"]) in tiles
                                                                  for d in s0["dinosaurs"])})
        free0 = free_tiles(s0)
        # is there another meteor landing 1-2 ticks later, overlapping/adjacent?
        follow = [(c2, t2, L + dt) for dt in (1, 2) for c2, t2 in land_by_t.get(L + dt, []) if manh(c2, cen) <= 2 * R + 2]
        for d in s0["dinosaurs"]:
            p0 = (d["position"]["x"], d["position"]["y"])
            if p0 not in tiles:
                continue
            # path t0..L
            path = [p0]
            for tk in range(t0 + 1, L + 1):
                st = by_t.get(tk)
                q = next(((e["position"]["x"], e["position"]["y"]) for e in st["dinosaurs"] if e["id"] == d["id"]), None) if st else None
                if q is None:
                    break
                path.append(q)
            if len(path) < 2:
                continue
            p1 = path[1]
            d0, d1 = manh(p0, cen), manh(p1, cen)
            first = "stay" if p1 == p0 else ("away" if d1 > d0 else ("toward" if d1 < d0 else "sideways"))
            alive_at_L = d["id"] in posL
            caught = alive_at_L and posL[d["id"]] in tiles
            pend = posL.get(d["id"], path[-1])
            # max possible distance from centre reachable in the available moves (ignoring walls)
            moves = L - t0
            exits = sum(1 for q in free0 if manh(q, p0) <= moves and manh(q, cen) > R)
            caught_chain = None
            if follow and alive_at_L and not caught:
                # did the follow-up meteor get it?
                for c2, t2, L2 in follow:
                    st = by_t.get(L2)
                    if st:
                        q = next(((e["position"]["x"], e["position"]["y"]) for e in st["dinosaurs"] if e["id"] == d["id"]), None)
                        if q is not None and q in t2:
                            caught_chain = True
                caught_chain = bool(caught_chain)
            out["threat"].append({"sp": d["name"], "first": first, "d0": d0, "caught": caught,
                                  "gone_other": not alive_at_L, "dend": manh(pend, cen), "moves": moves,
                                  "exits": exits, "follow": bool(follow), "caught_chain": caught_chain,
                                  "dir": (int(np.sign(pend[0] - cen[0])), int(np.sign(pend[1] - cen[1])))})
    return out


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    print(f"reading {len(files)} games...")
    with Pool(args.workers) as pool:
        res = [r for r in pool.map(read_game, files, chunksize=4) if r]
    print(f"{len(res)} complete games")
    T = [x for r in res for x in r["threat"]]
    M = [x for r in res for x in r["meteors"]]
    calm = Counter(); calm_sp = defaultdict(Counter)
    for r in res:
        calm.update(r["calm"])
        for k, v in r["calm_sp"].items():
            calm_sp[k].update(v)

    print("\n" + "=" * 74 + "\n1. HIT RATE")
    k = np.array([m["kills"] for m in M])
    print(f"   meteors: {len(M):,} | kills per meteor {k.mean():.2f} | meteors with 0 kills {pct((k == 0).mean())}"
          f" | 2+ kills {pct((k >= 2).mean())}")
    caught = np.array([x["caught"] for x in T]); other = np.array([x["gone_other"] for x in T])
    print(f"   dinos inside the blast when it appears: {len(T):,}")
    print(f"     still inside at impact (killed): {pct(caught.mean())} | escaped: {pct((~caught & ~other).mean())}"
          f" | died another way first: {pct(other.mean())}")
    print("   by distance from centre at launch (0 = on the centre):")
    for dd in range(0, 3):
        sel = np.array([x["d0"] == dd for x in T])
        if sel.any():
            print(f"     {dd}: caught {pct(caught[sel].mean())}  (n={sel.sum():,})")

    print("\n" + "=" * 74 + "\n2. REACTION: first move after the meteor appears")
    fc = Counter(x["first"] for x in T)
    tot = sum(fc.values())
    print("   threatened: " + "  ".join(f"{k} {pct(fc[k] / tot)}" for k in ("away", "sideways", "toward", "stay")))
    print(f"   calm dinos stay {pct(calm['stay'] / max(1, sum(calm.values())))} of the time")
    print(f"   {'species':15s} {'away':>7s} {'side':>7s} {'toward':>7s} {'stay':>7s} {'calm stay':>10s} {'caught':>7s}")
    for sp in sorted({x["sp"] for x in T}):
        xs = [x for x in T if x["sp"] == sp]
        c = Counter(x["first"] for x in xs); n = len(xs)
        cs = calm_sp[sp]
        print(f"   {sp:15s} " + " ".join(f"{pct(c[k] / n):>7s}" for k in ("away", "sideways", "toward", "stay"))
              + f" {pct(cs['stay'] / max(1, sum(cs.values()))):>10s} {pct(np.mean([x['caught'] for x in xs])):>7s}")

    print("\n" + "=" * 74 + "\n3. ESCAPE: distance from the blast centre at impact (escaped dinos)")
    esc = [x for x in T if not x["caught"] and not x["gone_other"]]
    if esc:
        dc = Counter(x["dend"] for x in esc)
        print("   " + "  ".join(f"d={k}: {pct(v / len(esc))}" for k, v in sorted(dc.items())[:8]))
        print(f"   average {np.mean([x['dend'] for x in esc]):.1f} tiles (blast radius is 2: 3 = just outside)")
        straight = np.mean([x["dend"] >= x["d0"] + x["moves"] - 1 for x in esc])
        print(f"   ran (almost) straight away from the centre: {pct(straight)}")

    print("\n" + "=" * 74 + "\n4. WALLS: caught rate vs free escape tiles around the dino")
    ex = np.array([x["exits"] for x in T])
    for lo, hi in [(0, 1), (1, 4), (4, 8), (8, 15), (15, 25), (25, 99)]:
        sel = (ex >= lo) & (ex < hi)
        if sel.any():
            print(f"   {lo:>2}-{hi - 1:<2} free escape tiles: caught {pct(caught[sel].mean())}  (n={sel.sum():,})")

    print("\n" + "=" * 74 + "\n5. CHAINS: a 2nd meteor landing 1-2 ticks later next to the first")
    f = [x for x in T if x["follow"]]
    nf = [x for x in T if not x["follow"]]
    if f:
        tot_f = np.mean([x["caught"] or bool(x["caught_chain"]) for x in f])
        print(f"   single meteor: caught {pct(np.mean([x['caught'] for x in nf]))} (n={len(nf):,})")
        print(f"   with follow-up: caught by 1st {pct(np.mean([x['caught'] for x in f]))}, "
              f"by 1st OR 2nd {pct(tot_f)} (n={len(f):,})")
    else:
        print("   no chained meteors found")


if __name__ == "__main__":
    main()
