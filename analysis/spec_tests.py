#!/usr/bin/env python3
"""
SPEC TESTS: tests every hypothesis in results/spec_hypothesis.md (H1-H30) on game logs.

Each test prints the measured numbers next to a BASELINE (what you'd see by chance, or for the other species),
so you can judge each hypothesis yourself. Nothing is marked true/false automatically.

Input: game logs with one full state per tick (local: logs/local_game_logs/*.jsonl,
       server: team logs converted with analysis/team_log_to_jsonl.py).
Usage (repo root):
  python analysis/spec_tests.py --logs logs/local_game_logs
  python analysis/spec_tests.py --logs logs/server_jsonl --tag server
  python analysis/spec_tests.py --logs logs/local_game_logs --max-games 100      # quicker
Writes: results/analysis/<tag>/spec_tests.txt  (tag default: local)

Definitions used throughout
  move        world-coordinate step of a dino between tick t and t+1: stay / up (y-1) / down (y+1) / left / right
  window      20x20 visible map; it scrolls toward +y, so the TOP row (smallest y) is the trailing edge
  threat      a falling meteor whose blast covers the dino or a tile next to it, or a T-Rex within 2 tiles (non-Rex)
  safe        no threat
  free tile   inside the window, not impassable, not mountain, not lava
  options     the free tiles among the dino's 4 neighbours + its own tile (what it could have done)
"""
import argparse, glob, json, os, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
SPECIES = "SVTR"
NAME = {"S": "Stegosaurus", "V": "Velociraptor", "T": "Triceratops", "R": "T-Rex"}
STEPS = {(0, 0): "stay", (0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
NB = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def load(path):
    recs = []
    for l in open(path, "rb"):
        if l.strip():
            try:
                recs.append(json.loads(l)["state"])
            except Exception:
                pass
    recs.sort(key=lambda s: s["currentTick"])
    return recs


class Tick:
    """everything we need from one state"""

    def __init__(self, s):
        m = s["map"]
        self.t = s["currentTick"]
        self.ox, self.oy = m["origin"]["x"], m["origin"]["y"]
        self.h, self.w = m["height"], m["width"]
        self.elev, self.block, self.lava = {}, set(), set()
        for i, row in enumerate(m["tiles"]):
            for j, tl in enumerate(row):
                p = (j + self.ox, i + self.oy)
                self.elev[p] = tl.get("elevation", 0)
                if tl["isImpassable"] or tl["isMountain"]:
                    self.block.add(p)
                if tl["hasLava"]:
                    self.lava.add(p)
        self.d = {d["id"]: (SP.get(d["name"], "?"), (d["position"]["x"], d["position"]["y"]), d["age"])
                  for d in s["dinosaurs"]}
        self.met = [((me["target"]["x"], me["target"]["y"]), me["turnsUntilImpact"],
                     {(q["x"], q["y"]) for q in me["impactedTiles"]}) for me in s["meteors"]]
        self.corpses = {(c["x"], c["y"]) for c in s.get("corpses", [])}

    def inside(self, p):
        return self.ox <= p[0] < self.ox + self.w and self.oy <= p[1] < self.oy + self.h

    def free(self, p):
        return self.inside(p) and p not in self.block and p not in self.lava

    def options(self, p):
        return [p] + [(p[0] + a, p[1] + b) for a, b in NB if self.free((p[0] + a, p[1] + b))]

    def in_blast(self, p):
        return [m for m in self.met if p in m[2]]

    def threatened(self, i):
        sp, p, _ = self.d[i]
        near = any(manh(p, q) <= 1 for m in self.met for q in m[2] if manh(p, m[0]) <= 4)
        rex = sp != "R" and any(s2 == "R" and manh(p, p2) <= 2 for j, (s2, p2, _) in self.d.items() if j != i)
        return near or rex

    def reach(self, p, k=2):
        seen, front = {p}, [p]
        for _ in range(k):
            nxt = []
            for q in front:
                for a, b in NB:
                    r = (q[0] + a, q[1] + b)
                    if r not in seen and self.free(r):
                        seen.add(r)
                        nxt.append(r)
            front = nxt
        return len(seen)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def game(path):
    """all counters for one game (summed over games later)"""
    S = load(path)
    if len(S) < 50:
        return None
    C = defaultdict(Counter)             # name -> Counter
    L = defaultdict(list)                # name -> list of values (kept small: means only)
    T = [Tick(s) for s in S]
    rng = random.Random(0)
    last = {}                            # id -> last non-zero move
    # ---- per tick pair
    for a, b in zip(T, T[1:]):
        if b.t != a.t + 1:
            continue
        landing = set().union(*[m[2] for m in a.met if m[1] == 1]) if a.met else set()
        occupied = {p for _, p, _ in a.d.values()}
        # H20-H21: window
        C["window"]["dy_" + str(b.oy - a.oy)] += 1
        if b.oy != a.oy:
            C["window"]["shift_tick_mod5_" + str(b.t % 5)] += 1
        # H18: shared tiles
        pos = Counter(p for _, p, _ in b.d.values())
        C["collide"]["ticks"] += 1
        C["collide"]["shared_tiles"] += sum(1 for v in pos.values() if v > 1)
        # H19, H16, H22, H24: where dinos stand
        for i, (sp, p, _) in b.d.items():
            C["stand"]["dinos"] += 1
            C["stand"]["on_blocked"] += p in b.block
            C["stand"]["on_lava"] += p in b.lava
            C["stand"]["outside_window"] += not b.inside(p)
            C["stand"]["on_corpse"] += p in b.corpses
        # H24 lava over corpse, H25 corpse inside landing blast, H23/H26 corpse lifetime
        C["corpse"]["lava_on_corpse"] += len(b.lava & a.corpses)
        for c in a.corpses:
            C["corpse"]["seen"] += 1
            gone = c not in b.corpses
            if c in landing:
                C["corpse"]["in_blast"] += 1
                C["corpse"]["in_blast_gone"] += gone
            if gone:
                C["corpse"]["gone"] += 1
                C["corpse"]["gone_scrolled_out"] += not b.inside(c)
        for c in b.corpses - a.corpses:
            C["corpse"]["new"] += 1
            C["corpse"]["new_on_death_tile"] += any(p == c and i not in b.d for i, (_, p, _) in a.d.items())
        # deaths (H14, H22)
        for i, (sp, p, age) in a.d.items():
            if i in b.d:
                continue
            if p in landing:
                cause = "meteor"
            elif not b.inside(p):
                cause = "scroll"
            elif p in b.lava or p in a.lava:
                cause = "lava"
            else:
                cause = "other"
            C["death"][cause] += 1
            if cause == "scroll":
                C["death"]["scroll_was_top_row"] += p[1] < b.oy
        # H14: P(non-meteor, non-scroll death next tick) by distance to nearest T-Rex
        rexes = [p for sp, p, _ in a.d.values() if sp == "R"]
        for i, (sp, p, _) in a.d.items():
            if sp == "R" or not rexes or p in landing:
                continue
            k = min(manh(p, r) for r in rexes)
            k = str(k) if k <= 3 else "4+"
            C["eat"]["n_" + k] += 1
            C["eat"]["dead_" + k] += i not in b.d and b.inside(p) and p not in b.lava
        # ---- per dino moves
        for i, (sp, p, age) in a.d.items():
            if i not in b.d:
                continue
            q = b.d[i][1]
            mv = (q[0] - p[0], q[1] - p[1])
            if mv not in STEPS:
                C["moves"]["jump"] += 1
                continue
            name = STEPS[mv]
            opts = a.options(p)
            safe = not a.threatened(i)
            blast = a.in_blast(p)
            C["moves"]["all"] += 1
            # H1 / H6: move mix when safe vs threatened, repeat-last-move
            key = "safe" if safe else "threat"
            C[f"mix_{key}_{sp}"][name] += 1
            if safe and mv != (0, 0):
                if i in last:
                    C[f"repeat_{sp}"]["n"] += 1
                    C[f"repeat_{sp}"]["same"] += last[i] == mv
            if mv != (0, 0):
                last[i] = mv
            row, col = p[1] - a.oy, p[0] - a.ox
            # H2 / H3 / H17: move direction by window position (safe ticks)
            if safe:
                rb = f"row{min(row // 5, 3) * 5:02d}-{min(row // 5, 3) * 5 + 4:02d}"
                cb = f"col{min(col // 5, 3) * 5:02d}-{min(col // 5, 3) * 5 + 4:02d}"
                C[f"rowdir_{sp}_{rb}"][name] += 1
                C[f"coldir_{sp}_{cb}"][name] += 1
                for edge, dist, away in (("top", row, "down"), ("bottom", a.h - 1 - row, "up"),
                                         ("left", col, "right"), ("right", a.w - 1 - col, "left")):
                    db = "0-1" if dist <= 1 else "2-4" if dist <= 4 else "5+"
                    C[f"edge_{sp}_{edge}_{db}"]["n"] += 1
                    C[f"edge_{sp}_{edge}_{db}"]["away"] += name == away
                    C[f"edge_{sp}_{edge}_{db}"]["toward"] += name == {"down": "up", "up": "down",
                                                                      "right": "left", "left": "right"}[away]
            # H4 / H15: flee from a blast it stands in
            if blast:
                cen, turns, _ = blast[0]
                d0, d1 = manh(p, cen), manh(q, cen)
                k = "first_tick" if turns == max(m[1] for m in a.met) and turns >= 3 else "later"
                C[f"flee_{sp}"]["n"] += 1
                C[f"flee_{sp}"]["away"] += d1 > d0
                C[f"flee_{sp}"]["stay"] += d1 == d0
                C[f"flee_when_{k}"]["n"] += 1
                C[f"flee_when_{k}"]["away"] += d1 > d0
            # H8: just outside a falling blast (distance R+1 = safe by 1 tile): does it stop?
            for cen, turns, tiles in a.met:
                if p not in tiles and manh(p, cen) == 3:
                    C[f"edgeofblast_{sp}"]["n"] += 1
                    C[f"edgeofblast_{sp}"]["stay"] += mv == (0, 0)
                    break
            # H5 / H7: corpses (safe ticks, corpse within 8 tiles)
            if safe and a.corpses:
                near_c = min(a.corpses, key=lambda c: manh(p, c))
                if manh(p, near_c) <= 8:
                    ch = manh(q, near_c) - manh(p, near_c)
                    base = mean([manh(o, near_c) - manh(p, near_c) for o in opts])
                    C[f"corpse_{sp}"]["n"] += 1
                    C[f"corpse_{sp}"]["closer"] += ch < 0
                    L[f"corpse_{sp}_chosen"].append(ch)
                    L[f"corpse_{sp}_options"].append(base)
            # H11: elevation (safe, moved)
            if safe and len(opts) > 1:
                e0 = a.elev.get(p, 0)
                ch = a.elev.get(q, e0) - e0
                L[f"elev_{sp}_chosen"].append(ch)
                L[f"elev_{sp}_options"].append(mean([a.elev.get(o, e0) - e0 for o in opts]))
                mx = max(a.elev.get(o, e0) for o in opts)
                C[f"elev_{sp}"]["n"] += 1
                C[f"elev_{sp}"]["picked_highest"] += a.elev.get(q, e0) == mx
                C[f"elev_{sp}"]["chance_highest"] += sum(1 for o in opts if a.elev.get(o, e0) == mx) / len(opts)
            # H12: survival (free tiles reachable in 2 moves after the move)
            if len(opts) > 1 and rng.random() < 0.25:            # sampled: it is the slowest test
                r_ch = a.reach(q)
                r_opt = mean([a.reach(o) for o in opts])
                L[f"reach_{sp}_chosen"].append(r_ch)
                L[f"reach_{sp}_options"].append(r_opt)
            # H13: spreading away from other dinos (no dino adjacent)
            others = [p2 for j, (s2, p2, _) in a.d.items() if j != i]
            if others and min(manh(p, o) for o in others) >= 2:
                nd = lambda x: min(manh(x, o) for o in others)
                L[f"spread_{sp}_chosen"].append(nd(q) - nd(p))
                L[f"spread_{sp}_options"].append(mean([nd(o) - nd(p) for o in opts]))
            # H16: next to lava
            if a.lava:
                lv = lambda x: any((x[0] + u, x[1] + v) in a.lava for u, v in NB + [(0, 0)])
                if any(lv(o) for o in opts):
                    C[f"lava_{sp}"]["n"] += 1
                    C[f"lava_{sp}"]["chosen_near"] += lv(q)
                    C[f"lava_{sp}"]["chance_near"] += sum(lv(o) for o in opts) / len(opts)
            # H18: blocked by another dino in its last direction
            if i in last and safe:
                ahead = (p[0] + last[i][0], p[1] + last[i][1])
                if a.free(ahead):
                    k = "occupied" if ahead in occupied else "empty"
                    C[f"ahead_{k}"]["n"] += 1
                    C[f"ahead_{k}"]["stay"] += mv == (0, 0)
            # H9 / H10: packs
            if sp == "T":
                tri = [p2 for j, (s2, p2, _) in a.d.items() if s2 == "T" and j != i]
                oth = [p2 for j, (s2, p2, _) in a.d.items() if s2 != "T" and j != i]
                if tri:
                    L["pack_tri_to_tri"].append(min(manh(p, x) for x in tri))
                if oth:
                    L["pack_tri_to_other"].append(min(manh(p, x) for x in oth))
                for j, (s2, p2, _) in a.d.items():
                    if j > i and s2 == "T" and j in b.d:
                        mv2 = (b.d[j][1][0] - p2[0], b.d[j][1][1] - p2[1])
                        k = "close" if manh(p, p2) <= 4 else "far"
                        C[f"packmove_{k}"]["n"] += 1
                        C[f"packmove_{k}"]["same"] += mv == mv2
            elif sp in "SVR":
                tri = [p2 for j, (s2, p2, _) in a.d.items() if s2 == "T"]
                if tri:
                    L["pack_other_to_tri"].append(min(manh(p, x) for x in tri))
    # ---- waves (H27-H30)
    seen, last_spawn, waves = set(), -99, []
    for tk in T:
        new = [i for i in tk.d if i not in seen]
        if new:
            if tk.t - last_spawn > 2:
                waves.append({"start": tk.t, "ids": set(), "mix": Counter(), "end": None})
            last_spawn = tk.t
            seen.update(new)
            for i in new:
                waves[-1]["ids"].add(i)
                waves[-1]["mix"][tk.d[i][0]] += 1
        C["alive"]["max_" + str(min(len(tk.d), 25))] += 1
        for w in waves:
            if w["end"] is None and not (w["ids"] & set(tk.d)) and tk.t > w["start"]:
                w["end"] = tk.t
    recipes = []
    for k, w in enumerate(waves):
        rec = " ".join(f"{w['mix'][s]}{s}" for s in SPECIES if w["mix"][s])
        recipes.append(rec)
        C["recipe_pos" + str(k % 10 + 1)][rec] += 1
        if k + 1 < len(waves):
            nxt = waves[k + 1]["start"]
            board_clear_before = all(x["end"] is not None and x["end"] <= nxt for x in waves[:k + 1])
            if board_clear_before:
                last_end = max(x["end"] for x in waves[:k + 1])
                C["wave_gap"]["after_clear_+" + str(min(nxt - last_end, 9))] += 1
            else:
                C["wave_gap"]["timer_" + str(min(nxt - w["start"], 120) // 10 * 10)] += 1
        C["wave_size"][str(len(w["ids"]))] += 1
    for k in range(len(recipes) - 10):
        C["cycle"]["n"] += 1
        C["cycle"]["same"] += recipes[k] == recipes[k + 10]
    L = {k: (sum(v), len(v)) for k, v in L.items()}
    return {k: dict(v) for k, v in C.items()}, L


def pct(c, k, n="n"):
    return f"{c.get(k, 0) / c[n]:.1%}" if c.get(n) else "  -  "


def mix_line(c):
    n = sum(c.values())
    return " ".join(f"{k} {c.get(k, 0) / n:.0%}" for k in ("stay", "up", "down", "left", "right")) + f" (n={n})" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/local_game_logs")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.logs, "*.jsonl")))
    if a.max_games:
        files = files[:a.max_games]
    if not files:
        sys.exit(f"no .jsonl games in {a.logs}")
    print(f"{len(files)} games, {a.workers} workers ...")
    C, L, n = defaultdict(Counter), defaultdict(lambda: [0.0, 0]), 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(game, files, chunksize=4):
            if r is None:
                continue
            n += 1
            for k, v in r[0].items():
                C[k].update(v)
            for k, (s, m) in r[1].items():
                L[k][0] += s
                L[k][1] += m
            if n % 100 == 0:
                print(f"  {n} games", flush=True)
    M = lambda k: (L[k][0] / L[k][1]) if L[k][1] else float("nan")
    out = []
    say = out.append
    say(f"SPEC TESTS | {n} games from {a.logs} | {C['moves']['all']} dino moves")
    say("Each line: measured value vs BASELINE (chance = average over the moves the dino could have made, "
        "or the other species). You judge.")

    say("\n## Stegosaurus")
    say("H1  move mix when safe (and how often it repeats its last non-stay move):")
    for sp in SPECIES:
        r = C[f"repeat_{sp}"]
        say(f"    {NAME[sp]:12} {mix_line(C[f'mix_safe_{sp}'])} | repeats last move {pct(r, 'same')}")
    say("H2  direction by row in the window (safe; row 0 = top/trailing edge) - Stegosaurus vs Triceratops:")
    for sp in "ST":
        for rb in ("row00-04", "row05-09", "row10-14", "row15-19"):
            say(f"    {NAME[sp]:12} {rb}: {mix_line(C[f'rowdir_{sp}_{rb}'])}")
    say("    direction by column (safe):")
    for cb in ("col00-04", "col05-09", "col10-14", "col15-19"):
        say(f"    Stegosaurus  {cb}: {mix_line(C[f'coldir_S_{cb}'])}")
    say("H3  near the TOP (trailing) edge: P(move away = down) vs P(move toward = up), by distance (safe):")
    for sp in SPECIES:
        say(f"    {NAME[sp]:12} " + " | ".join(
            f"dist {db}: away {pct(C[f'edge_{sp}_top_{db}'], 'away')} toward {pct(C[f'edge_{sp}_top_{db}'], 'toward')}"
            for db in ("0-1", "2-4", "5+")))
    say("H4  standing inside a falling blast: next move goes farther from the centre / stays:")
    for sp in SPECIES:
        c = C[f"flee_{sp}"]
        say(f"    {NAME[sp]:12} away {pct(c, 'away')} | stays at same distance {pct(c, 'stay')} | n={c.get('n', 0)}")

    say("\n## Velociraptor")
    say("H5/H7 safe, a corpse within 8 tiles: P(move closer) and average distance change, chosen vs options:")
    for sp in SPECIES:
        c = C[f"corpse_{sp}"]
        say(f"    {NAME[sp]:12} closer {pct(c, 'closer')} | change chosen {M(f'corpse_{sp}_chosen'):+.3f} vs options "
            f"{M(f'corpse_{sp}_options'):+.3f} | n={c.get('n', 0)}")
    say("H6  stay rate safe vs threatened:")
    for sp in SPECIES:
        s1, s2 = C[f"mix_safe_{sp}"], C[f"mix_threat_{sp}"]
        say(f"    {NAME[sp]:12} safe {s1.get('stay', 0) / max(sum(s1.values()), 1):.0%} | "
            f"threatened {s2.get('stay', 0) / max(sum(s2.values()), 1):.0%}")
    say("H8  just outside a falling blast (distance 3 from its centre): P(stay):")
    for sp in SPECIES:
        c = C[f"edgeofblast_{sp}"]
        say(f"    {NAME[sp]:12} stays {pct(c, 'stay')} | n={c.get('n', 0)}")

    say("\n## Triceratops")
    say(f"H9  nearest Triceratops: from a Triceratops {M('pack_tri_to_tri'):.2f} tiles | from other species "
        f"{M('pack_other_to_tri'):.2f} | Triceratops to nearest non-Triceratops {M('pack_tri_to_other'):.2f}")
    a1, a2 = C["packmove_close"], C["packmove_far"]
    say(f"H10 two Triceratops make the SAME move: within 4 tiles {pct(a1, 'same')} (n={a1.get('n', 0)}) | "
        f"farther apart {pct(a2, 'same')} (n={a2.get('n', 0)})")
    say("H11 elevation (safe): change of the chosen move vs average of the options, and P(picks the highest option):")
    for sp in SPECIES:
        c = C[f"elev_{sp}"]
        say(f"    {NAME[sp]:12} chosen {M(f'elev_{sp}_chosen'):+.3f} vs options {M(f'elev_{sp}_options'):+.3f} | "
            f"picks highest {pct(c, 'picked_highest')} vs chance {c.get('chance_highest', 0) / max(c.get('n', 1), 1):.1%}")

    say("\n## T-Rex")
    say("H12 free tiles reachable within 2 moves after the move: chosen vs options (sampled 25%):")
    for sp in SPECIES:
        say(f"    {NAME[sp]:12} chosen {M(f'reach_{sp}_chosen'):.2f} vs options {M(f'reach_{sp}_options'):.2f}")
    say("H13 no dino adjacent: change in distance to the nearest other dino, chosen vs options:")
    for sp in SPECIES:
        say(f"    {NAME[sp]:12} chosen {M(f'spread_{sp}_chosen'):+.3f} vs options {M(f'spread_{sp}_options'):+.3f}")
    e = C["eat"]
    say("H14 P(dies next tick, not by meteor / scroll / lava) by distance to the nearest T-Rex:")
    say("    " + " | ".join(f"dist {k}: {e.get('dead_' + k, 0) / e['n_' + k]:.1%} (n={e['n_' + k]})"
                            for k in ("1", "2", "3", "4+") if e.get("n_" + k)))

    say("\n## All dinosaurs")
    f1, f2 = C["flee_when_first_tick"], C["flee_when_later"]
    say(f"H15 inside a blast: moves away on the FIRST tick after launch {pct(f1, 'away')} (n={f1.get('n', 0)}) | "
        f"later ticks {pct(f2, 'away')} (n={f2.get('n', 0)})")
    say("H16 a lava-adjacent option exists: chosen tile next to lava vs chance:")
    for sp in SPECIES:
        c = C[f"lava_{sp}"]
        say(f"    {NAME[sp]:12} chosen {pct(c, 'chosen_near')} vs chance "
            f"{c.get('chance_near', 0) / max(c.get('n', 1), 1):.1%} | n={c.get('n', 0)}")
    st = C["stand"]
    say(f"    dinos standing ON lava: {st.get('on_lava', 0)} of {st.get('dinos', 0)}")
    say("H17 P(move away from / toward each edge) at distance 0-1 vs 5+ (safe):")
    for sp in SPECIES:
        parts = []
        for edge in ("top", "bottom", "left", "right"):
            c0, c5 = C[f"edge_{sp}_{edge}_0-1"], C[f"edge_{sp}_{edge}_5+"]
            parts.append(f"{edge}: away {pct(c0, 'away')}/{pct(c5, 'away')} toward {pct(c0, 'toward')}/{pct(c5, 'toward')}")
        say(f"    {NAME[sp]:12} " + " | ".join(parts) + "   (near/far)")
    c = C["collide"]
    o, f = C["ahead_occupied"], C["ahead_empty"]
    say(f"H18 tiles holding 2+ dinos: {c.get('shared_tiles', 0)} over {c.get('ticks', 0)} ticks | "
        f"P(stay) when the tile in its last direction holds a dino {pct(o, 'stay')} (n={o.get('n', 0)}) vs empty {pct(f, 'stay')}")
    say(f"H19 dinos standing on impassable / mountain tiles: {st.get('on_blocked', 0)} of {st.get('dinos', 0)} | "
        f"moves of more than 1 tile: {C['moves'].get('jump', 0)}")

    say("\n## Rolling window")
    w = C["window"]
    say(f"H20 origin y change per tick: " + ", ".join(f"{k[3:]}: {v}" for k, v in sorted(w.items()) if k.startswith("dy_")))
    say(f"H21 tick (mod 5) of each window shift: " + ", ".join(f"{k[-1]}: {v}" for k, v in sorted(w.items())
                                                          if k.startswith("shift_tick")))
    d = C["death"]
    say(f"H22 dinos outside the window: {st.get('outside_window', 0)} | deaths by scrolling: {d.get('scroll', 0)} "
        f"(was in the top row: {d.get('scroll_was_top_row', 0)})")

    say("\n## Corpses")
    cp = C["corpse"]
    say(f"H23 new corpses on the tile where a dino just died: {cp.get('new_on_death_tile', 0)} of {cp.get('new', 0)}")
    say(f"H24 dinos standing on a corpse tile: {st.get('on_corpse', 0)} | lava on a corpse tile (tile-ticks): "
        f"{cp.get('lava_on_corpse', 0)}")
    say(f"H25 corpses inside a landing blast: {cp.get('in_blast', 0)} | gone the next tick: {cp.get('in_blast_gone', 0)}")
    say(f"H26 corpses that disappeared: {cp.get('gone', 0)} | of which scrolled out of the window: "
        f"{cp.get('gone_scrolled_out', 0)}")

    say("\n## Waves")
    say("H27 most common species mix at each cycle position (share of waves):")
    for k in range(1, 11):
        r = C["recipe_pos" + str(k)]
        tot = sum(r.values())
        if tot:
            top = r.most_common(2)
            say(f"    pos {k:2}: " + " | ".join(f"{rec} {v / tot:.0%}" for rec, v in top) + f" (n={tot})")
    g = C["wave_gap"]
    say("H28 next wave start: after a board clear (+ticks) / by the timer (ticks since previous start):")
    say("    " + ", ".join(f"{k}: {v}" for k, v in sorted(g.items())))
    al = C["alive"]
    mx = max((int(k[4:]) for k in al if al[k]), default=0)
    say(f"H29 most dinos alive at once: {mx} | ticks with 20 alive: {al.get('max_20', 0)} | wave sizes: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(C['wave_size'].items(), key=lambda x: int(x[0]))))
    cy = C["cycle"]
    say(f"H30 wave k and wave k+10 have the same species mix: {pct(cy, 'same')} (n={cy.get('n', 0)})")
    say(f"\ndeath causes (helper): " + ", ".join(f"{k} {v}" for k, v in d.items() if k != "scroll_was_top_row"))

    txt = "\n".join(out)
    print(txt)
    path = os.path.join("results", "analysis", a.tag, "spec_tests.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(txt + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
