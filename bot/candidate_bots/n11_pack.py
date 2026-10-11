"""
N11 pack = N7 safe timing (= N6 = N4 + follow-up cap + timing recovery; all frozen) + PACK mode only.

From N10 (3 server games): future / runner spots ~13,000 added, CNN4 chose 18, 0 kills -> removed.
Pack spots: 15 chosen, 5 kills (0.33 per shot vs ~0.2 for an average meteor) -> kept.
T-Rex mode: almost never triggered (1 shot) -> off here, to test separately later.

PACK: when >= 2 Triceratops are alive, the centre of every group of Triceratops within PACK_DIST of each other
is PROMOTED to a guaranteed CNN4 slot (after N6's priority follow-ups). CNN4 still makes the final choice.

Logs ([N11]): one line per cleared wave: number, cycle position 1-10, recipe, start / end tick, ticks to clear,
last survivor species, ticks with 2 / 1 dinos of that wave left; summary per wave type and pack stats.
Use: ./scripts/ship.sh n11_pack
"""
from collections import Counter, defaultdict

from game_message import *

import n4_survival_follow as N4
import n7_safe_timing as N7

CB, C = N4.CB, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
PACK_DIST = 4            # Triceratops this close belong to one pack
PACK_K = 4               # max pack spots per tick
# --------------------------------------------------------------------------

C.BOT_NAME = "n11_pack"
SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Bot(N7.Bot):
    def __init__(self):
        super().__init__()
        self.seen_ids, self.last_spawn, self.wave_no = set(), -99, 0
        self.open_waves = []
        self.by_type = defaultdict(list)          # cycle position -> [ticks to clear]
        self.last_sp = Counter()
        self.pack = set()
        self.p_pending = []
        self.st11 = dict(ticks=0, added=0, chosen=0, kills=0)
        print(f"[N11] loaded n11_pack | pack dist {PACK_DIST} | max {PACK_K} pack spots")

    # ---------------------------------------------------------------- wave tracking + logs
    def track_waves(self, s):
        T = s["currentTick"]
        new = [d for d in s["dinosaurs"] if d["id"] not in self.seen_ids]
        for d in new:
            self.seen_ids.add(d["id"])
        if new:
            if T - self.last_spawn > 2:
                self.wave_no += 1
                self.open_waves.append(dict(no=self.wave_no, pos=(self.wave_no - 1) % 10 + 1, t=T, ids={}, c=Counter(),
                                            left2=0, left1=0))
            self.last_spawn = T
            for d in new:
                self.open_waves[-1]["ids"][d["id"]] = SP.get(d["name"], "?")
                self.open_waves[-1]["c"][SP.get(d["name"], "?")] += 1
        alive = {d["id"] for d in s["dinosaurs"]}
        still = []
        for w in self.open_waves:
            left = [i for i in w["ids"] if i in alive]
            if len(left) == 2:
                w["left2"] += 1
            elif len(left) == 1:
                w["left1"] += 1
                w["last"] = w["ids"][left[0]]
            if left:
                still.append(w)
                continue
            dur = T - w["t"]
            self.by_type[w["pos"]].append(dur)
            self.last_sp[w.get("last", "?")] += 1
            rec = " ".join(f"{w['c'][k]}{k}" for k in "SVTR" if w["c"][k])
            print(f"[N11] wave {w['no']} type {w['pos']} ({rec}) start {w['t']} end {T} | {dur} ticks | "
                  f"last {w.get('last', '?')} | 2 left {w['left2']} ticks, 1 left {w['left1']} ticks")
        self.open_waves = still

    def choose_actions(self, s):
        try:
            self.track_waves(s)
        except Exception as e:
            print("[N11] wave error:", repr(e))
        self.names = {d["id"]: d["name"] for d in s["dinosaurs"]}
        return super().choose_actions(s)

    # ---------------------------------------------------------------- pack spots
    def pack_spots(self):
        tri = [d["pos"] for d in self.dinos if self.names.get(d["id"]) == "Triceratops"]
        out, used = [], set()
        for i, p in enumerate(tri):
            if i in used:
                continue
            grp = [j for j, q in enumerate(tri) if manh(p, q) <= PACK_DIST]
            if len(grp) < 2:
                continue
            used |= set(grp)
            c = (round(sum(tri[j][0] for j in grp) / len(grp)), round(sum(tri[j][1] for j in grp) / len(grp)))
            out.append(c)
        return out[:PACK_K]

    def candidates(self, s):
        lst = super().candidates(s)                         # N6 order
        self.pack = set()
        if not lst or self.cool > 0 or not hasattr(self, "names"):
            return lst
        extra = [p for p in self.pack_spots() if p not in self.follow and self.in_window(*p)]
        if not extra:
            return lst
        self.st11["ticks"] += 1
        self.st11["added"] += len(extra)
        self.pack = set(extra)
        for p in extra:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        k = 0
        while k < len(lst) and lst[k] in self.follow:
            k += 1
        return lst[:k] + extra + [p for p in lst[k:] if p not in self.pack]

    # ---------------------------------------------------------------- stats
    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, LaunchMeteorAction) and (a.target.x, a.target.y) in self.pack:
                p = (a.target.x, a.target.y)
                self.st11["chosen"] += 1
                self.p_pending.append((s["currentTick"] + s["constants"]["meteorDelay"],
                                       N4.blast(p, s["constants"]["meteorRadius"])))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles in self.p_pending:
                if L == T - 1:
                    self.st11["kills"] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                              and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.p_pending = [x for x in self.p_pending if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        parts = []
        for k in range(1, 11):
            v = sorted(self.by_type.get(k, []))
            if v:
                parts.append(f"{k}:{sum(v) / len(v):.0f}")
        print("[N11] avg ticks per wave type: " + " ".join(parts))
        print("[N11] last survivor: " + ", ".join(f"{k} {v}" for k, v in self.last_sp.most_common()))
        x = self.st11
        print(f"[N11] pack | ticks with pack spots {x['ticks']} | added {x['added']} | chosen {x['chosen']} | "
              f"kills {x['kills']} = {x['kills'] / max(x['chosen'], 1):.2f}/shot (all meteors "
              f"{self.r['kills'] / max(self.r['meteors'], 1):.2f}/shot)")
