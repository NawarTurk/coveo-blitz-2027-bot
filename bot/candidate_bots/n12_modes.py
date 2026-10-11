"""
N12 modes = N7 safe timing (frozen baseline) + one STRATEGY PER BOARD MODE, each with an on/off switch.

Every tick the bot reads the board and picks one mode (first match wins, same rules as analysis/mode_report.py):
  mode        when                                   strategy (CNN4 still makes the final pick)
  straggler   1-2 dinos on the board                 1st shot: best kill. Once a meteor falls on a straggler, the next
                                                     shots may only go to its escape spots, the other straggler,
                                                     escape tiles no meteor covers, or the 3 best direct spots
  rex         a T-Rex is adjacent to another dino    spots that hit the T-Rex (most dinos covered first) jump the queue
  pack        2+ Triceratops within 4 tiles          centre of each Triceratops group jumps the queue (from N11)
  crowd       6+ dinos                               spots whose blast covers 2+ dinos jump the queue (multi-kills)
  normal      3-5 dinos, nothing special             spots on the most trapped dino (fewest escape tiles) jump the queue
  empty       no dinos                               nothing to shoot
Flags logged in every mode: trapped (a dino with <= 3 escape tiles), rex_risk (T-Rex 1-2 tiles from prey), pack.
"Jump the queue" = placed right after N6's priority follow-ups, so CNN4 always scores them.

Switch a mode off in ON (it then plays exactly like N7).

Logs ([N12]): straggler stay lines; per mode at the end: ticks, shots, kills, kills/shot, and how many shots
went to that mode's own spots and what they killed; straggler shots by kind.
Use: ./scripts/ship.sh n12_modes
"""
from collections import Counter, defaultdict

from game_message import *

import n4_survival_follow as N4
import n7_safe_timing as N7

CB, C = N4.CB, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
ON = dict(straggler=True, rex=True, pack=True, crowd=True, normal=True)   # strategy per mode
STRAGGLER = True         # kept for older notes; ON["straggler"] decides
STRAG_MAX = 2            # straggler mode = 1..STRAG_MAX dinos on the board
TOP_DIRECT = 3           # best direct spots always kept in straggler 2nd/3rd shots
TRAP_MAX, PACK_DIST, CROWD_MIN = 3, 4, 6
REX_K, PACK_K, CROWD_K, NORMAL_K = 4, 4, 6, 4   # spots promoted per tick in each mode
# --------------------------------------------------------------------------

C.BOT_NAME = "n12_modes"
MODES = ["straggler", "rex", "pack", "crowd", "normal", "empty"]
FLAGS = ["trapped", "rex_risk", "pack"]
R, MOVES = 2, 3
FOOT = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if abs(a) + abs(b) <= R]
REACH = [(a, b) for a in range(-MOVES, MOVES + 1) for b in range(-MOVES, MOVES + 1) if abs(a) + abs(b) <= MOVES]


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def board_mode(s):
    """primary mode + flags from the raw game state (same rules as analysis/mode_report.py)"""
    m = s["map"]
    ox, oy = m["origin"]["x"], m["origin"]["y"]
    free = {(j + ox, i + oy) for i, row in enumerate(m["tiles"]) for j, t in enumerate(row)
            if not (t["isImpassable"] or t["isMountain"] or t["hasLava"])}
    ds = [(d["name"], (d["position"]["x"], d["position"]["y"])) for d in s["dinosaurs"]]
    n = len(ds)
    trapped = False
    for _, p in ds:
        reach = [(p[0] + a, p[1] + b) for a, b in REACH]
        reach = [q for q in reach if q in free]
        if min(sum(1 for q in reach if manh(q, (p[0] + a, p[1] + b)) > R) for a, b in FOOT) <= TRAP_MAX:
            trapped = True
            break
    rex = [p for k, p in ds if k == "Tyrannosaurus"]
    allp = [p for _, p in ds]
    rex_adj = any(manh(x, o) == 1 for x in rex for o in allp)
    rex_risk = rex_adj or any(manh(x, o) == 2 for x in rex for o in allp)
    tri = [p for k, p in ds if k == "Triceratops"]
    pack = any(manh(a, b) <= PACK_DIST for i, a in enumerate(tri) for b in tri[i + 1:])
    if n == 0:
        mode = "empty"
    elif n <= STRAG_MAX:
        mode = "straggler"
    elif rex_adj:
        mode = "rex"
    elif pack:
        mode = "pack"
    elif n >= CROWD_MIN:
        mode = "crowd"
    else:
        mode = "normal"
    return mode, {"trapped": trapped, "rex_risk": rex_risk, "pack": pack}


class Bot(N7.Bot):
    OWN_KINDS = ("follow", "other", "route", "own")      # shot kinds that count as the mode's own spots

    def __init__(self):
        super().__init__()
        self.mode, self.flags = "empty", {}
        self.ms = {m: Counter() for m in MODES}
        self.fs = {f: Counter() for f in FLAGS}
        self.kind_n, self.kind_k = Counter(), Counter()
        self.kinds = {}                       # spot -> straggler shot kind this tick
        self.shot_kind = None
        self.k_pending = []                   # (landing tick, tiles, mode, kind)
        self.stay = None                      # current straggler stay
        self.stays = []
        self.promo = set()
        print(f"[N12] loaded n12_modes | strategies on: {', '.join(k for k, v in ON.items() if v) or 'none'} "
              f"| straggler 1..{STRAG_MAX} dinos, keep top {TOP_DIRECT} direct | promoted per tick rex {REX_K} "
              f"pack {PACK_K} crowd {CROWD_K} normal {NORMAL_K}")

    # ---------------------------------------------------------------- mode per tick
    def choose_actions(self, s):
        try:
            self.mode, self.flags = board_mode(s)
        except Exception as e:
            print("[N12] mode error:", repr(e))
            self.mode, self.flags = "normal", {}
        self.track_stay(s)
        self.kinds, self.shot_kind, self.promo = {}, None, set()
        self.names = {d["id"]: d["name"] for d in s["dinosaurs"]}
        return super().choose_actions(s)

    def track_stay(self, s):
        T = s["currentTick"]
        if self.mode == "straggler":
            if self.stay is None:
                self.stay = dict(start=T, shots=0, kills=0)
        elif self.stay is not None:
            st = self.stay
            st["len"] = T - st["start"]
            st["end"] = "cleared" if self.mode == "empty" else "next wave"
            self.stays.append(st)
            print(f"[N12] straggler stay t={st['start']}-{T} | {st['len']} ticks | shots {st['shots']} | "
                  f"kills {st['kills']} | ended: {st['end']}")
            self.stay = None

    # ---------------------------------------------------------------- straggler candidates
    def candidates(self, s):
        lst = super().candidates(s)                         # N7 = N6 order (sets dinos, falling, follow, prior)
        if not lst or self.cool > 0 or not ON.get(self.mode):
            return lst
        if self.mode == "straggler":
            return self.straggler(lst)
        extra = {"rex": self.rex_spots, "pack": self.pack_spots, "crowd": self.crowd_spots,
                 "normal": self.normal_spots}[self.mode]()
        extra = [p for p in dict.fromkeys(extra) if p not in self.follow and self.in_window(*p)]
        if not extra:
            return lst
        self.promo = set(extra)
        k = 0
        while k < len(lst) and lst[k] in self.follow:
            k += 1
        return lst[:k] + extra + [p for p in lst[k:] if p not in self.promo]

    # ---------------------------------------------------------------- mode strategies
    def covers(self, c):
        return [d for d in self.dinos if manh(d["pos"], c) <= self.R]

    def ranked(self, spots, k):
        for p in spots:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        return sorted(spots, key=lambda p: (-len(self.covers(p)), -self.prior[p]))[:k]

    def rex_spots(self):
        """blast centres on every T-Rex that is next to prey; ones also hitting the prey first"""
        rex = [d for d in self.dinos if self.names.get(d["id"]) == "Tyrannosaurus"]
        rex = [r for r in rex if any(manh(r["pos"], o["pos"]) == 1 for o in self.dinos if o is not r)]
        spots = {(r["pos"][0] + a, r["pos"][1] + b) for r in rex for a, b in FOOT}
        return self.ranked(list(spots), REX_K)

    def pack_spots(self):
        """centre of every group of Triceratops within PACK_DIST of each other (N11)"""
        tri = [d["pos"] for d in self.dinos if self.names.get(d["id"]) == "Triceratops"]
        out, used = [], set()
        for i, p in enumerate(tri):
            if i in used:
                continue
            grp = [j for j, q in enumerate(tri) if manh(p, q) <= PACK_DIST]
            if len(grp) < 2:
                continue
            used |= set(grp)
            out.append((round(sum(tri[j][0] for j in grp) / len(grp)), round(sum(tri[j][1] for j in grp) / len(grp))))
        return out[:PACK_K]

    def crowd_spots(self):
        """spots whose blast covers 2+ dinos, most dinos / best prior first"""
        spots = {(d["pos"][0] + a, d["pos"][1] + b) for d in self.dinos for a, b in FOOT}
        multi = [p for p in spots if len(self.covers(p)) >= 2]
        return self.ranked(multi, CROWD_K)

    def normal_spots(self):
        """blast centres on the most trapped dino (fewest escape tiles for its best centre)"""
        best, tgt = 10 ** 9, None
        for d in self.dinos:
            e = min(sum(1 for q in d["reach"] if manh(q, (d["pos"][0] + a, d["pos"][1] + b)) > self.R) for a, b in FOOT)
            if e < best:
                best, tgt = e, d
        if tgt is None:
            return []
        spots = [(tgt["pos"][0] + a, tgt["pos"][1] + b) for a, b in FOOT]
        for p in spots:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        return sorted(spots, key=lambda p: -self.prior[p])[:NORMAL_K]

    def straggler(self, lst):
        near = lambda d: any(manh(d["pos"], cen) <= self.R + 1 for _, cen, _ in self.falling)
        hit = [d for d in self.dinos if near(d)]
        if not hit:                                         # 1st shot: unchanged
            self.kinds = {p: "first" for p in lst}
            return lst
        free_d = [d for d in self.dinos if not near(d)]
        deadly = set().union(*(t for _, _, t in self.falling)) if self.falling else set()
        pool, kinds = [], {}

        def add(p, k):
            if p not in kinds and self.in_window(*p):
                kinds[p] = k
                pool.append(p)
        for p in lst:                                       # escape follow-ups, in N6's order
            if p in self.follow:
                add(p, "follow")
        for d in free_d:                                    # the other straggler, still untouched
            for a, b in FOOT:
                add((d["pos"][0] + a, d["pos"][1] + b), "other")
        for d in hit:                                       # escape tiles no falling meteor covers
            for q in d["reach"]:
                if q not in deadly:
                    add(q, "route")
        for p in lst[:TOP_DIRECT]:
            add(p, "direct")
        for p in pool:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        fol = [p for p in pool if kinds[p] == "follow"]                      # follow-ups first (N6 order)
        pool = fol + sorted((p for p in pool if kinds[p] != "follow"), key=lambda p: -self.prior[p])
        self.kinds = kinds
        return pool

    # ---------------------------------------------------------------- stats
    def after(self, s, actions, ms):
        shot = None
        for a in actions:
            if isinstance(a, LaunchMeteorAction):
                shot = (a.target.x, a.target.y)
        m = self.ms[self.mode]
        m["ticks"] += 1
        for f in FLAGS:
            if self.flags.get(f):
                self.fs[f]["ticks"] += 1
                self.fs[f]["shots"] += shot is not None
        if shot is not None:
            m["shots"] += 1
            kind = self.kinds.get(shot) if self.mode == "straggler" else ("own" if shot in self.promo else None)
            if kind in self.OWN_KINDS:
                m["own"] += 1
            if self.mode == "straggler":
                kind = kind or "first"
                self.kind_n[kind] += 1
                if self.stay is not None:
                    self.stay["shots"] += 1
            flags = tuple(f for f in FLAGS if self.flags.get(f))
            self.k_pending.append((s["currentTick"] + s["constants"]["meteorDelay"],
                                   N4.blast(shot, s["constants"]["meteorRadius"]), self.mode, kind, flags))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles, mode, kind, flags in self.k_pending:
                if L == T - 1:
                    k = sum(1 for d in p["dinosaurs"] if d["id"] not in now
                            and (d["position"]["x"], d["position"]["y"]) in tiles)
                    self.ms[mode]["kills"] += k
                    if kind in self.OWN_KINDS:
                        self.ms[mode]["own_k"] += k
                    for f in flags:
                        self.fs[f]["kills"] += k
                    if kind and mode == "straggler":
                        self.kind_k[kind] += k
                        if self.stay is not None:
                            self.stay["kills"] += k
            self.k_pending = [x for x in self.k_pending if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        tot = max(sum(m["ticks"] for m in self.ms.values()), 1)
        for k, m in self.ms.items():
            if not m["ticks"]:
                continue
            own = "" if k == "empty" else (f" | strategy {'ON' if ON.get(k) else 'off'}: own spots chosen {m['own']} "
                                           f"-> {m['own_k']} kills {m['own_k'] / max(m['own'], 1):.2f}/shot")
            print(f"[N12] mode {k:9} | {m['ticks']} ticks ({m['ticks'] / tot:.0%}) | {m['shots']} shots | "
                  f"{m['kills']} kills {m['kills'] / max(m['shots'], 1):.2f}/shot{own}")
        print("[N12] flags: " + " | ".join(
            f"{f} {c['ticks']}t {c['shots']} shots {c['kills']} kills {c['kills'] / max(c['shots'], 1):.2f}/shot"
            for f, c in self.fs.items()))
        print("[N12] straggler shots by kind: " + " | ".join(
            f"{k} {self.kind_n[k]} shots {self.kind_k[k]} kills {self.kind_k[k] / max(self.kind_n[k], 1):.2f}/shot"
            for k in ["first", "follow", "other", "route", "direct", "normal", "wide"] if self.kind_n[k]))
        v = sorted(x["len"] for x in self.stays)
        if v:
            ends = Counter(x["end"] for x in self.stays)
            print(f"[N12] straggler stays {len(v)} | median {v[len(v) // 2]} ticks | 80% {v[int(len(v) * .8)]} | "
                  f"max {v[-1]} | total {sum(v)} ticks | ended: " + ", ".join(f"{k} {n}" for k, n in ends.items()))
