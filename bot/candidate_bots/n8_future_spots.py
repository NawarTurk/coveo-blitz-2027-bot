"""
N8 future spots = N7 safe timing (= N6 = N4 + follow-up cap + timing recovery; all frozen) + ONE change:
candidate spots where the dinos WILL BE, not only where they are.

Every tick the bot plans a meteor (normal timing level only, so N7's protection is untouched):
  1. one CNN4 run with NO new meteor -> P(dino on each tile) at our impact tick
     (meteors already falling stay in the input, so fleeing dinos are predicted fleeing)
  2. every tile as a blast centre: sum of P x kill value over its 13 blast tiles (plain numpy, < 1 ms)
  3. the best FUTURE_K centres that are NOT already candidates ("novel") ...
  4. ... get a GUARANTEED CNN4 slot: placed right after N6's priority follow-ups, before the cage spots
  5. normal CNN4 scoring (with the dinos' reaction to that shot) picks the winner, as always

Logs:
  [N8] summary: forecast ticks, novel spots added, future spots CNN4 chose, their kills/shot vs all meteors,
                forecast mass vs live dinos (mass can be lower: dinos die to falling meteors / lava / scroll)
  + all N4 / N6 / N7 logs and the full trace.
Use: ./scripts/ship.sh n8_future_spots
"""
import time

import numpy as np
from game_message import *

import n4_survival_follow as N4
import n7_safe_timing as N7

CB, C = N4.CB, N4.C
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
FUTURE_K = 6             # novel future-map centres guaranteed a CNN4 slot
MIN_FUTURE = 0.05        # skip centres worth less than this (expected value, newborn kill = 1)
# --------------------------------------------------------------------------

C.BOT_NAME = "n8_future_spots"


class Bot(N7.Bot):
    def __init__(self):
        super().__init__()
        self.future = set()
        self.fut_pending = []            # (landing tick, blast tiles)
        self.n8 = dict(ticks=0, added=0, chosen=0, kills=0, mass=0.0, live=0, skipped=0)
        print(f"[N8] loaded n8_future_spots | FUTURE_K {FUTURE_K} | MIN_FUTURE {MIN_FUTURE}")

    # ---------------------------------------------------------------- the change
    def future_spots(self, s, taken):
        """top novel blast centres on the no-new-meteor forecast"""
        X0, sc = self.base_input(s)
        P = self.predict4([X0], sc)[0]                       # (H, W) P(dino) at our impact tick
        live = len(self.dinos)
        self.n8["mass"] += float(P.sum()); self.n8["live"] += live
        G = P * self.value_grid()                            # expected value per tile
        H, W, R = self.H, self.W, self.R
        pad = np.pad(G, R)
        S = np.zeros((H, W), np.float32)
        for dx, dy in self.foot:                             # sum over the 13 blast tiles of each centre
            S += pad[R + dy:R + dy + H, R + dx:R + dx + W]
        out = []
        for k in np.argsort(-S, axis=None):
            if S.flat[k] < MIN_FUTURE or len(out) >= FUTURE_K:
                break
            i, j = divmod(int(k), W)
            p = (j + self.ox, i + self.oy)
            if p in taken or not self.in_window(*p):
                continue
            out.append(p)
        return out

    def candidates(self, s):
        lst = super().candidates(s)                          # N6 order: priority follow-ups, cage spots, rest
        self.future = set()
        if self.cool > 0 or self.level < len(N7.LEVELS) - 1 or not lst:
            self.n8["skipped"] += 1                          # N7 cooldown / probe: no extra CNN4 work
            return lst
        try:
            fut = self.future_spots(s, set(lst))
        except Exception as e:
            print("[N8] forecast error:", repr(e))
            return lst
        if not fut:
            return lst
        self.n8["ticks"] += 1; self.n8["added"] += len(fut)
        self.future = set(fut)
        for p in fut:
            if p not in self.prior:
                self.prior[p] = self.cage_prior(p)
        k = 0                                                # keep N6's priority follow-ups in front
        while k < len(lst) and lst[k] in self.follow:
            k += 1
        return lst[:k] + fut + lst[k:]

    # ---------------------------------------------------------------- [N8] stats
    def after(self, s, actions, ms):
        for a in actions:
            if isinstance(a, LaunchMeteorAction) and (a.target.x, a.target.y) in self.future:
                c, T = (a.target.x, a.target.y), s["currentTick"]
                self.n8["chosen"] += 1
                self.fut_pending.append((T + s["constants"]["meteorDelay"], N4.blast(c, s["constants"]["meteorRadius"])))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            for L, tiles in self.fut_pending:
                if L == T - 1:
                    self.n8["kills"] += sum(1 for d in p["dinosaurs"] if d["id"] not in now
                                            and (d["position"]["x"], d["position"]["y"]) in tiles)
            self.fut_pending = [(L, tl) for L, tl in self.fut_pending if L >= T]
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        n = self.n8
        t = max(n["ticks"], 1)
        allk = self.r["kills"] / max(self.r["meteors"], 1)
        print(f"[N8] summary | forecast ticks {n['ticks']} (skipped {n['skipped']}) | novel future spots added {n['added']} "
              f"({n['added'] / t:.1f}/tick) | chosen by CNN4 {n['chosen']} | their kills {n['kills']} = "
              f"{n['kills'] / max(n['chosen'], 1):.2f}/shot (all meteors {allk:.2f}/shot) | forecast mass "
              f"{n['mass'] / t:.1f} vs live dinos {n['live'] / t:.1f} per tick")
