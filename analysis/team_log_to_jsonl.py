#!/usr/bin/env python3
"""
Convert downloaded server TEAM logs (T| trace lines) into local-style game logs, so every analysis
script can run on real server games.

Usage:  python analysis/team_log_to_jsonl.py server_logs/*.txt --out server_game_logs
        python analysis/waves.py --logs server_game_logs
Writes: <out>/game_<name>.jsonl  (one {"tick", "state", "actions"} line per tick)

What the trace has: dinos, meteors (centre + turns), volcanoes, lava, our action, score,
and (newer bots) corpses "c", terrain rows "tr", errors "err".
Old logs without "tr": tiles get elevation 0 and nothing blocked; without "c": no corpses.
"""
import argparse, json, os

R = 2
FOOT = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]
NAME = {"S": "Stegosaurus", "V": "Velociraptor", "T": "Triceratops", "R": "Tyrannosaurus"}
CONST = {"maxTicks": 1000, "maxMeteors": 3, "maxVolcanoes": 1, "meteorDelay": 4, "meteorRadius": 2,
         "mapShiftInterval": 5, "waveCadence": 100, "dinosPerWave": 10, "activeDinoCap": 20}


def convert(path, out_dir):
    recs = []
    for l in open(path, errors="ignore"):
        if l.startswith("T|"):
            try:
                recs.append(json.loads(l[2:]))
            except Exception:
                pass
    if not recs:
        return 0
    recs.sort(key=lambda r: r["t"])
    lava, corpses, terrain, const = set(), [], {}, dict(CONST)
    name = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(out_dir, f"game_{name}.jsonl"), "w") as f:
        for r in recs:
            if "lava" in r:
                lava = {tuple(p) for p in r["lava"]}
            if "c" in r:
                corpses = r["c"]
            for y, row in (r.get("tr") or {}).items():
                terrain[int(y)] = row
            if "k" in r:
                const.update(r["k"])
            ox, oy = r["o"]
            tiles = []
            for i in range(20):
                row = terrain.get(oy + i)
                tiles.append([{"position": {"x": ox + j, "y": oy + i},
                               "elevation": row[j][0] if row else 0,
                               "isImpassable": bool(row[j][1] & 1) if row else False,
                               "isMountain": bool(row[j][1] & 2) if row else False,
                               "hasLava": (ox + j, oy + i) in lava} for j in range(20)])
            st = {"tick": r["t"], "currentTick": r["t"], "score": r["s"], "lastTickErrors": r.get("err", []),
                  "constants": const,
                  "map": {"width": 20, "height": 20, "origin": {"x": ox, "y": oy}, "tiles": tiles},
                  "dinosaurs": [{"id": d[0], "name": NAME.get(d[1], d[1]), "position": {"x": d[2], "y": d[3]},
                                 "age": d[4]} for d in r["d"]],
                  "meteors": [{"target": {"x": m[0], "y": m[1]}, "turnsUntilImpact": m[2],
                               "impactedTiles": [{"x": m[0] + a, "y": m[1] + b} for a, b in FOOT]} for m in r["m"]],
                  "volcanoes": [{"position": {"x": v[0], "y": v[1]}} for v in r["v"]],
                  "mountains": [{"x": ox + j, "y": oy + i} for i in range(20) for j in range(20) if tiles[i][j]["isMountain"]],
                  "corpses": [{"x": c[0], "y": c[1]} for c in corpses]}
            a = r.get("a") or []
            acts = ([{"type": "LAUNCH_METEOR", "target": {"x": a[1], "y": a[2]}}] if a and a[0] == "M" else
                    [{"type": "TRIGGER_VOLCANO", "target": {"x": a[1], "y": a[2]}}] if a and a[0] == "V" else [])
            f.write(json.dumps({"tick": r["t"], "state": st, "actions": acts}) + "\n")
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="server_game_logs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for p in a.logs:
        k = convert(p, a.out)
        n += k > 0
        print(f"{os.path.basename(p)}: {k} ticks" if k else f"{os.path.basename(p)}: no trace lines (skipped)")
    print(f"\n{n} games -> {a.out}/")


if __name__ == "__main__":
    main()
