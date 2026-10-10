"""
N10 wave modes = N8 future spots (N7 timing, N6 follow-up cap, N4 survival rule: all frozen)
+ a candidate strategy per wave group. CNN4 still judges every spot; only WHICH spots it gets changes.

Wave tracking: a new batch of dino IDs (spawned > 2 ticks after the previous batch) = a new wave;
its species recipe (10-wave cycle) sets the group, the dinos actually alive confirm it:

  RUNNER  (waves 1,2,3,5,7: Stegosaurus / raptors)  Stegosaurus never stand still (6%), raptors park.
          -> N8 forecast spots: RUNNER_FUTURE_K guaranteed slots instead of 6.
  PACK    (waves 4,6,8: 7-8 Triceratops)            packs are loose (~5 tiles) but multi-kills pay x1.5 / x2.
          -> pack spots: centre of every group of >= 2 Triceratops within PACK_DIST of each other.
  REX     (waves 6,8,9,10: T-Rex present)          a dino next to a T-Rex is eaten next tick 58% (0 points).
          -> rex spots: on the T-Rex, and between the T-Rex and each prey within REX_DIST (both in one blast).
  Modes can overlap (wave 6 / 8 = PACK + REX). Pack / rex spots are PROMOTED to guaranteed CNN4 slots
  (after N6's follow-ups and N8's future spots), whether or not the cage list already had them.

Logs ([N10]): a line per finished wave (number, recipe, modes, ticks to clear, kills),
summary per mode: waves, median ticks, spots added / chosen / kills per shot. + all earlier logs and the trace.
Use: ./scripts/ship.sh n10_wave_modes
"""
from collections import Counter

from game_message import *

import n4_survival_follow as N4
import n8_future_spots as N8

CB, CA, C = N4.CB, N4.CB.CA, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
RUNNER_FUTURE_K = 10     # N8 forecast spots in runner mode (N8: 6)
PACK_DIST = 4            # Triceratops this close belong to one pack
PACK_K = 4               # max pack spots
REX_DIST = 3             # prey this close to a T-Rex
REX_K = 4                # max rex spots
NEW_TRAP = False         # True = updated trap table (94/63/37/14/1%) for aiming; off = one change at a time
# --------------------------------------------------------------------------

if NEW_TRAP:
    CA.TRAP = [(0, 0.94), (3, 0.63), (7, 0.37), (14, 0.14), (999, 0.011)]
C.BOT_NAME = "n10_wave_modes"
SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
CYCLE_MODES = {1: {"runner"}, 2: {"runner"}, 3: {"runner"}, 4: {"pack"}, 5: {"runner"},
               6: {"pack", "rex"}, 7: {"runner"}, 8: {"pack", "rex"}, 9: {"rex"}, 10: {"rex"}}


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Bot(N8.Bot):
    def __init__(self):
        super().__init__()
        self.seen_ids, self.last_spawn, self.wave_no = set(), -99, 0
        self.open_waves = []             # dict(no, pos, recipe, t, ids)
        self.modes = set()
        self.mode_spots = {}             # spot -> mode
        self.m_pending = []              # (landing tick, tiles, mode)
        self.st10 = {m: dict(waves=0, ticks=[], added=0, chosen=0, kills=0) for m in ("runner", "pack", "rex", "none")}
        self.mode_ticks = Counter()
        print(f"[N10] loaded n10_wave_modes | runner future {RUNNER_FUTURE_K} | pack {PACK_K} spots (dist {PACK_DIST}) "
              f"| rex {REX_K} spots (dist {REX_DIST}) | new trap table {NEW_TRAP}")

    # ---------------------------------------------------------------- wave tracking
    def track_waves(self, s):
        T = s["currentTick"]
        new = [d for d in s["dinosaurs"] if d["id"] not in self.seen_ids]
        for d in new:
            self.seen_ids.add(d["id"])
        if new:
            if T - self.last_spawn > 2:
                self.wave_no += 1
                pos = (self.wave_no - 1) % 10 + 1
                self.open_waves.append(dict(no=self.wave_no, pos=pos, t=T, ids=set(), c=Counter()))
            self.last_spawn = T
            for d in new:
                self.open_waves[-1]["ids"].add(d["id"])
                self.open_waves[-1]["c"][SP.get(d["name"], "?")] += 1
        alive = {d["id"] for d in s["dinosaurs"]}
        still = []
        for w in self.open_waves:
            if w["ids"] & alive:
                still.append(w)
                continue
            dur = T - w["t"]
            ms = self.wave_modes(w)
            rec = " ".join(f"{w['c'][k]}{k}" for k in "SVTR" if w["c"][k])
            for m in (ms or {"none"}):
                self.st10[m]["waves"] += 1
                self.st10[m]["ticks"].append(dur)
            print(f"[N10] t={T} wave {w['no']} (cycle {w['pos']}, {rec}) cleared in {dur} ticks | modes {'+'.join(sorted(ms)) or 'none'}")
        self.open_waves = still

    def wave_modes(self, w):
        c = w["c"]
        ms = set()
        if c["S"] + c["V"] >= 6 and not c["T"] and not c["R"]:
            ms.add("runner")
        if c["T"] >= 4:
            ms.add("pack")
        if c["R"] >= 1:
            ms.add("rex")
        return ms or CYCLE_MODES.get(w["pos"], set())

    def board_modes(self, s):
        """wave groups give the prior; dinos actually alive decide"""
        prior = set()
        for w in self.open_waves:
            prior |= self.wave_modes(w)
        names = Counter(d["name"] for d in s["dinosaurs"])
        ms = set()
        if "runner" in prior and (names["Stegosaurus"] or names["Velociraptor"]):
            ms.add("runner")
        if "pack" in prior and names["Triceratops"] >= 2:
            ms.add("pack")
        if "rex" in prior and names["Tyrannosaurus"]:
            ms.add("rex")
        return ms

    def choose_actions(self, s):
        try:
            self.track_waves(s)
            self.modes = self.board_modes(s)
        except Exception as e:
            print("[N10] wave error:", repr(e))
            self.modes = set()
        for m in (self.modes or {"none"}):
            self.mode_ticks[m] += 1
        N8.FUTURE_K = RUNNER_FUTURE_K if "runner" in self.modes else 6
        return super().choose_actions(s)

    # ---------------------------------------------------------------- mode spots
    def pack_spots(self):
        tri = [d["pos"] for d in self.dinos if self.names.get(d["id"]) == "Triceratops"] if hasattr(self, "names") else []
        out, used = [], set()
        for i, p in enumerate(tri):
            if i in used:
                continue
            grp = [q for j, q in enumerate(tri) if manh(p, q) <= PACK_DIST]
            if len(grp) < 2:
                continue
            used |= {j for j, q in enumerate(tri) if manh(p, q) <= PACK_DIST}
            c = (round(sum(q[0] for q in grp) / len(grp)), round(sum(q[1] for q in grp) / len(grp)))
            out.append(c)
        return out[:PACK_K]

    def rex_spots(self):
        if not hasattr(self, "names"):
            return []
        rex = [d["pos"] for d in self.dinos if self.names.get(d["id"]) == "Tyrannosaurus"]
        prey = [d["pos"] for d in self.dinos if self.names.get(d["id"]) != "Tyrannosaurus"]
        out = []
        for r in rex:
            near = sorted((q for q in prey if manh(q, r) <= REX_DIST), key=lambda q: manh(q, r))
            for q in near[:2]:
                out.append(((r[0] + q[0]) // 2, (r[1] + q[1]) // 2))      # both in one blast
            if near:
                out.append(r)
        return out[:REX_K]

    def candidates(self, s):
        self.names = {d["id"]: d["name"] for d in s["dinosaurs"]}
        lst = super().candidates(s)
        self.mode_spots = {}
        if not lst or self.cool > 0:
            return lst
        head = set(self.follow) | set(self.future)
        extra = []
        for mode, fn in (("pack", self.pack_spots), ("rex", self.rex_spots)):
            if mode not in self.modes:
                continue
            for p in fn():                                  # new spots AND existing ones are promoted
                if p not in head and p not in extra and self.in_window(*p):
                    extra.append(p)
                    self.mode_spots[p] = mode
                    self.st10[mode]["added"] += 1
                    if p not in self.prior:
                        self.prior[p] = self.cage_prior(p)
        if "runner" in self.modes:
            for p in self.future:
                self.mode_spots.setdefault(p, "runner")
            self.st10["runner"]["added"] += len(self.future)
        if not extra:
            return lst
        k = 0                                               # after N6's priority follow-ups and N8's future spots
        while k < len(lst) and (lst[k] in self.follow or lst[k] in self.future):
            k += 1
        ex = set(extra)
        return lst[:k] + extra + [p for p in lst[k:] if p not in ex]

    # ---------------------------------------------------------------- stats
    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, LaunchMeteorAction):
                p = (a.target.x, a.target.y)
                m = self.mode_spots.get(p)
                if m:
                    self.st10[m]["chosen"] += 1
                    self.m_pending.append((s["currentTick"] + s["constants"]["meteorDelay"],
                                           N4.blast(p, s["constants"]["meteorRadius"]), m))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles, m in self.m_pending:
                if L == T - 1:
                    self.st10[m]["kills"] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                                 and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.m_pending = [x for x in self.m_pending if x[0] >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        for m in ("runner", "pack", "rex", "none"):
            x = self.st10[m]
            t = sorted(x["ticks"])
            med = t[len(t) // 2] if t else None
            print(f"[N10] {m:6s} | ticks on {self.mode_ticks[m]} | waves {x['waves']} median clear {med} ticks "
                  f"| spots added {x['added']} chosen {x['chosen']} kills {x['kills']} "
                  f"= {x['kills'] / max(x['chosen'], 1):.2f}/shot")
