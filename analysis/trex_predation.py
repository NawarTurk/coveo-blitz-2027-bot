#!/usr/bin/env python3
"""
How much do T-Rexes cost us? Counts, in recorded games, every dino that disappears between two ticks and
classifies WHY:
  meteor : it stood in the blast of a meteor landing that tick
  lava   : lava on/next to its last tile
  scroll : its row scrolled out of the window
  eaten  : not any of the above and a T-Rex was within EAT_DIST tiles   (eaten = 0 points for us)
  other  : none of the above
For eaten dinos: points we could have had = 160 / (1 + age/30).

Usage:  python trex_predation.py logs            (all logs/game_*.jsonl)
        python trex_predation.py logs --max 300
"""
import argparse, glob, json, os
from collections import Counter, defaultdict
from multiprocessing import Pool

EAT_DIST = 2


def val(age):
    return 160.0 / (1 + age / 30.0)


def pos(d):
    return d["position"]["x"], d["position"]["y"]


def one_game(path):
    out = Counter()
    eaten_pts, eaten_age, eaten_sp, eat_dist = 0.0, [], Counter(), Counter()
    trex_meals = defaultdict(int)
    trex_ids = set()
    trex_life = {}
    prev, score = None, 0
    try:
        for line in open(path):
            s = json.loads(line)["state"]
            if prev is not None and s["currentTick"] == prev["currentTick"] + 1:
                now_ids = {d["id"] for d in s["dinosaurs"]}
                m = s["map"]
                ox, oy, h, w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]

                def lava_near(p):
                    for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                        i, j = p[1] + b - oy, p[0] + a - ox
                        if 0 <= i < h and 0 <= j < w and m["tiles"][i][j]["hasLava"]:
                            return True
                    return False

                landing = [frozenset((t["x"], t["y"]) for t in me["impactedTiles"])
                           for me in prev["meteors"] if me["turnsUntilImpact"] == 1]
                trexes = [(d["id"], pos(d)) for d in prev["dinosaurs"] if d["name"] == "Tyrannosaurus"]
                for d in prev["dinosaurs"]:
                    if d["id"] in now_ids:
                        continue
                    p = pos(d)
                    out["dead_total"] += 1
                    if any(p in L for L in landing):
                        out["meteor"] += 1
                    elif lava_near(p):
                        out["lava"] += 1
                    elif p[1] < oy:
                        out["scroll"] += 1
                    else:
                        near = [(abs(p[0] - q[0]) + abs(p[1] - q[1]), i) for i, q in trexes if i != d["id"]]
                        near = [x for x in near if x[0] <= EAT_DIST]
                        if near and d["name"] != "Tyrannosaurus":
                            dist, tid = min(near)
                            out["eaten"] += 1
                            eaten_pts += val(d["age"])
                            eaten_age.append(d["age"])
                            eaten_sp[d["name"]] += 1
                            eat_dist[dist] += 1
                            trex_meals[tid] += 1
                        else:
                            out["other"] += 1
            for d in s["dinosaurs"]:
                if d["name"] == "Tyrannosaurus":
                    trex_ids.add(d["id"])
                    trex_life[d["id"]] = trex_life.get(d["id"], 0) + 1
            score = s["score"]
            prev = s
    except Exception as e:
        return None
    return dict(c=out, pts=eaten_pts, ages=eaten_age, sp=eaten_sp, dist=eat_dist, n_trex=len(trex_ids),
                meals=[trex_meals.get(i, 0) for i in trex_ids], life=list(trex_life.values()), score=score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="?", default="local_game_logs")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.logs, "game_*.jsonl")))
    if args.max:
        files = files[-args.max:]
    with Pool() as pool:
        res = [r for r in pool.map(one_game, files, chunksize=4) if r is not None]
    res = [r for r in res if r["c"]["dead_total"] > 0 and r["c"]["meteor"] > 0]   # skip empty/test games
    n = len(res)
    if not n:
        raise SystemExit("no usable games")
    tot = Counter()
    for r in res:
        tot.update(r["c"])
    pts = [r["pts"] for r in res]
    scores = [r["score"] for r in res]
    ages = sorted(a for r in res for a in r["ages"])
    sp, dist = Counter(), Counter()
    for r in res:
        sp.update(r["sp"]); dist.update(r["dist"])
    meals = [m for r in res for m in r["meals"]]
    life = [l for r in res for l in r["life"]]
    D = tot["dead_total"]
    print(f"{n} games\n")
    print("How dinos die (per game avg, % of all deaths):")
    for k in ("meteor", "lava", "scroll", "eaten", "other"):
        print(f"  {k:<7} {tot[k] / n:6.1f}   {100 * tot[k] / D:5.1f}%")
    print(f"\nEATEN: {tot['eaten'] / n:.1f} dinos per game")
    print(f"  points lost per game: avg {sum(pts) / n:.0f}  (median {sorted(pts)[n // 2]:.0f}, max {max(pts):.0f})"
          f"  = {100 * sum(pts) / max(sum(scores), 1):.1f}% of our average score {sum(scores) / n:.0f}")
    if ages:
        print(f"  age of eaten dinos: median {ages[len(ages) // 2]}, mean {sum(ages) / len(ages):.0f}  "
              f"(value at median age {val(ages[len(ages) // 2]):.0f} pts)")
    print(f"  species eaten: {dict(sp)}")
    print(f"  T-Rex distance just before: {dict(sorted(dist.items()))}")
    print(f"\nT-REX: {sum(r['n_trex'] for r in res) / n:.1f} per game | meals per T-Rex: avg {sum(meals) / max(len(meals), 1):.2f}, "
          f"max {max(meals) if meals else 0} | % of T-Rexes that eat at least once: "
          f"{100 * sum(1 for m in meals if m > 0) / max(len(meals), 1):.0f}% | ticks alive avg {sum(life) / max(len(life), 1):.0f}")
    if meals and life:
        rate = sum(meals) / max(sum(life), 1)
        print(f"  meals per T-Rex per 100 ticks alive: {100 * rate:.2f}")
        print(f"  -> killing a T-Rex ~X ticks earlier saves about X * {rate:.4f} meals x ~{val(ages[len(ages) // 2]) if ages else 0:.0f} pts")
