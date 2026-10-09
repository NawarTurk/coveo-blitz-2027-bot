"""
cage_cnn4_report = cage_cnn4 (Cage B, our best) with EXACTLY the same decisions + logging for server games.

The server only lets us download what the bot prints (the team log), so this bot prints:

1. TRACE (one line per tick, starts with "T|"): compact JSON with everything needed to analyse the game later
     t      tick                 s   score              o   [origin x, origin y]
     d      dinos [id, species letter S/V/T/R, x, y, age]
     m      falling meteors [x, y, turns left]          v   active volcanoes [x, y]
     lava   lava tiles [x, y] (only when it changes)     a   our action ["M", x, y] meteor / ["V", x, y] volcano / []
     ms     milliseconds the tick took
   Turn it off with TRACE = False.

2. REPORT at the end of the game (lines start with "[REPORT]"): score, timing, meteors, kills per meteor,
   multi-kills, how dinos died, waves (ticks per wave, 50% / 80% / all dead), last survivor species,
   per-100-tick table.

Use: ./scripts/ship.sh cage_cnn4_report
"""
from game_message import *
import json, time

import numpy as np

import cage_cnn4 as CB

CA, C, FC = CB.CA, CB.C, CB.FC

TRACE = True
C.BOT_NAME = "cage_cnn4_report"
SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}


def val(age):
    return 160.0 / (1 + age / 30.0)


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Bot(CB.Bot):
    def __init__(self):
        super().__init__()
        self.r = dict(meteors=0, kills=0, zero=0, m2=0, m3=0, m4=0, pts=0.0, clears=0, late=0, volc=0,
                      d_meteor=0, d_lava=0, d_scroll=0, d_eaten=0, d_other=0)
        self.pending, self.prev_s, self.biome = [], None, None
        self.born, self.info, self.dead = {}, {}, {}      # dino id -> spawn tick / species / death tick
        self.buckets = {}
        self.last_lava = None

    # ------------------------------------------------------------------ every tick (decisions untouched)
    def choose_actions(self, s):
        t0 = time.perf_counter()
        try:
            self.watch(s)
        except Exception as e:
            print("report error:", repr(e))
        actions = super().choose_actions(s)
        try:
            self.after(s, actions, 1000 * (time.perf_counter() - t0))
        except Exception as e:
            print("report error:", repr(e))
        return actions

    def watch(self, s):
        T = s["currentTick"]
        if self.biome is None:
            self.biome = s.get("biome") or s.get("map", {}).get("biome") or "?"
        if s.get("lastTickErrors"):
            self.r["late"] += 1
        for d in s["dinosaurs"]:
            if d["id"] not in self.born:
                self.born[d["id"]] = T
                self.info[d["id"]] = d["name"]
        p = self.prev_s
        b = self.buckets.setdefault(T // 100, [0, 0, 0])
        if p is not None and T == p["currentTick"] + 1:
            b[2] += s["score"] - p["score"]
            now = {d["id"] for d in s["dinosaurs"]}
            m = s["map"]
            ox, oy, h, w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
            ours = [tiles for L, tiles in self.pending if L == T - 1]
            other = [{(q["x"], q["y"]) for q in me["impactedTiles"]} for me in p["meteors"] if me["turnsUntilImpact"] == 1]
            trex = [(d["position"]["x"], d["position"]["y"]) for d in p["dinosaurs"] if d["name"] == "Tyrannosaurus"]
            per_shot = [[] for _ in ours]
            for d in p["dinosaurs"]:
                if d["id"] in now:
                    continue
                self.dead[d["id"]] = T
                q = (d["position"]["x"], d["position"]["y"])
                hit = [k for k, tl in enumerate(ours) if q in tl]
                if hit:
                    self.r["d_meteor"] += 1
                    per_shot[hit[0]].append(d["age"])
                elif any(q in L for L in other):
                    self.r["d_meteor"] += 1
                elif any(0 <= q[1] + bb - oy < h and 0 <= q[0] + aa - ox < w and m["tiles"][q[1] + bb - oy][q[0] + aa - ox]["hasLava"]
                         for aa, bb in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                    self.r["d_lava"] += 1
                elif q[1] < oy:
                    self.r["d_scroll"] += 1
                elif d["name"] != "Tyrannosaurus" and any(manh(q, x) <= 2 for x in trex):
                    self.r["d_eaten"] += 1
                else:
                    self.r["d_other"] += 1
            for ages in per_shot:
                k = len(ages)
                self.r["kills"] += k
                b[0] += k
                self.r["pts"] += sum(val(a) for a in ages) * (1 + 0.5 * max(k - 1, 0))
                if k == 0: self.r["zero"] += 1
                elif k == 2: self.r["m2"] += 1
                elif k == 3: self.r["m3"] += 1
                elif k >= 4: self.r["m4"] += 1
            self.pending = [(L, tl) for L, tl in self.pending if L >= T]
            if p["dinosaurs"] and not s["dinosaurs"]:
                self.r["clears"] += 1
        self.prev_s = s

    def after(self, s, actions, ms):
        T = s["currentTick"]
        a_out = []
        for a in actions:
            if isinstance(a, LaunchMeteorAction):
                c = (a.target.x, a.target.y)
                R, D = s["constants"]["meteorRadius"], s["constants"]["meteorDelay"]
                self.pending.append((T + D, {(c[0] + dx, c[1] + dy) for dx in range(-R, R + 1)
                                             for dy in range(-R, R + 1) if abs(dx) + abs(dy) <= R}))
                self.r["meteors"] += 1
                self.buckets.setdefault(T // 100, [0, 0, 0])[1] += 1
                a_out = ["M", c[0], c[1]]
            elif isinstance(a, TriggerVolcanoAction):
                self.r["volc"] += 1
                a_out = ["V", a.target.x, a.target.y]
        if not TRACE:
            return
        m = s["map"]
        ox, oy = m["origin"]["x"], m["origin"]["y"]
        lava = sorted([j + ox, i + oy] for i, row in enumerate(m["tiles"]) for j, tl in enumerate(row) if tl["hasLava"])
        rec = {"t": T, "s": s["score"], "o": [ox, oy],
               "d": [[d["id"], SP.get(d["name"], "?"), d["position"]["x"], d["position"]["y"], d["age"]] for d in s["dinosaurs"]],
               "m": [[me["target"]["x"], me["target"]["y"], me["turnsUntilImpact"]] for me in s["meteors"]],
               "v": [[v["position"]["x"], v["position"]["y"]] for v in s["volcanoes"]],
               "a": a_out, "ms": round(ms, 1)}
        if lava != self.last_lava:
            rec["lava"] = lava
            self.last_lava = lava
        print("T|" + json.dumps(rec, separators=(",", ":")))

    # ------------------------------------------------------------------ end of game
    def waves(self):
        """group dinos by spawn tick (<= 2 ticks apart = same wave), timing of each wave"""
        ids = sorted(self.born, key=lambda i: self.born[i])
        groups, cur = [], []
        for i in ids:
            if cur and self.born[i] - self.born[cur[-1]] > 2:
                groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            t0 = self.born[g[0]]
            dt = sorted(self.dead[i] - t0 for i in g if i in self.dead)
            n = len(g)
            q = lambda f: dt[int(np.ceil(f * n)) - 1] if len(dt) >= int(np.ceil(f * n)) else None
            last = max((i for i in g if i in self.dead), key=lambda i: self.dead[i], default=None)
            out.append(dict(n=n, p50=q(0.5), p80=q(0.8), all=dt[-1] if len(dt) == n else None,
                            last=self.info[last] if (last and len(dt) == n) else None))
        return out

    def finish(self, s):
        super().finish(s)
        r = self.r
        ms = sorted(1000 * t for t in self.times) or [0]
        nm = max(r["meteors"], 1)
        W = self.waves()
        cl = [w for w in W if w["all"] is not None]
        med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else float("nan")
        lasts = {}
        for w in cl:
            lasts[w["last"]] = lasts.get(w["last"], 0) + 1
        P = lambda *a: print("[REPORT]", *a)
        P(f"score {s['score']} | biome {self.biome} | ms/tick avg {sum(ms) / len(ms):.0f} p99 {ms[int(.99 * (len(ms) - 1))]:.0f} "
          f"max {ms[-1]:.0f} | >100ms {sum(1 for x in ms if x > 100)} | late (server) {r['late']}")
        P(f"meteors {r['meteors']} | kills {r['kills']} = {r['kills'] / nm:.3f}/meteor | zero-kill {r['zero']} ({100 * r['zero'] / nm:.0f}%) "
          f"| multi 2:{r['m2']} 3:{r['m3']} 4+:{r['m4']} | meteor points ~{r['pts']:.0f} | volcano {r['volc']}")
        P(f"deaths: meteor {r['d_meteor']} | lava {r['d_lava']} | scroll {r['d_scroll']} | eaten {r['d_eaten']} | other {r['d_other']}")
        P(f"waves {len(W)} | cleared {len(cl)} | board clears {r['clears']} | ticks per wave (median): 50% {med([w['p50'] for w in cl]):.0f} "
          f"/ 80% {med([w['p80'] for w in cl]):.0f} / all {med([w['all'] for w in cl]):.0f}")
        P("last survivor of each wave: " + ", ".join(f"{k} {v}" for k, v in sorted(lasts.items(), key=lambda kv: -kv[1])))
        P(f"decisions: {self.used}")
        P("per 100 ticks (kills/meteors/score gained): " +
          " ".join(f"{k * 100}:{v[0]}/{v[1]}/{v[2]}" for k, v in sorted(self.buckets.items())))
