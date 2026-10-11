"""
N13 mode scoring = N12b (N7 targeting + N12 mode framework + wide straggler spots) + MODE-SPECIFIC SCORING.

Lesson from N12 / N12b: forcing or pushing spots loses, and new spots never win while 30% of every score is the
trap table (it only rewards hitting where a dino is NOW). N13 changes how much CNN4 is trusted, per mode.
Candidates are unchanged: N7 cage spots + follow-ups everywhere, N12b wide spots in straggler only. Nothing forced.

final score = (1 - W) * CNN4 + W * trap table, then x (1 + bonus)          (N7: W = 0.3 everywhere, no bonus)
  mode        W (trap share)   soft bonus
  straggler   0.00             -
  crowd       0.00             -
  pack        0.00             +PACK_BONUS if the blast covers 2+ Triceratops (now, +1 tile for movement)
  normal      0.15             -
  rex         0.00             +REX_BONUS if the blast covers a T-Rex that is adjacent to prey
  empty       nothing to shoot
Exploration build: all rules at once; after 3-5 games keep the modes that improve (-> N14), revert the rest.
Each mode's W / bonus is a knob below (set W to 0.3 and bonus to 0 = N7 for that mode).

Logs ([N13]):
  one line per shot when its result is known: tick, mode, source (normal / follow / wide), CNN4 part, trap part,
  bonus, final score, kills
  per mode at the end: ticks, shots, kills, kills/shot, zero-kill %, points from kills, avg episode length,
  avg CNN4 / trap part of chosen shots
  + all N12 / N12b logs (straggler stays, wide spots generated / chosen / kills) and the full trace
Use: ./scripts/ship.sh n13_mode_scoring
"""
import time
from collections import Counter, defaultdict

import numpy as np

import n12b_straggler_wide as N12b

N12, N4 = N12b.N12, N12b.N4
CB, C, manh = N12.CB, N12.C, N12.manh
FC = CB.FC
print = N12.print

# ---- knobs ---------------------------------------------------------------
TRAP_W = dict(straggler=0.0, crowd=0.0, pack=0.0, normal=0.15, rex=0.0, empty=0.3)   # N7 = 0.3 everywhere
PACK_BONUS = 0.15        # x1.15 for blasts covering 2+ Triceratops (pack mode only)
REX_BONUS = 0.15         # x1.15 for blasts covering a T-Rex adjacent to prey (rex mode only)
SHOT_LOG = True          # one [N13] line per shot
# --------------------------------------------------------------------------

C.BOT_NAME = "n13_mode_scoring"


def plan_meteor_scored(self, s):
    """cage_cnn4.plan_meteor with per-mode trap weight + soft bonus; keeps the parts of the chosen spot"""
    ranked = self.candidates(s)
    if not ranked:
        return None
    self.static = None
    if self.unit4 > CB.UNIT_CAP:
        self.unit4 = CB.UNIT_CAP
        self.capped += 1
    left = C.TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 3
    n = min(CB.MAX4, len(ranked), int(left / self.unit4) if self.unit4 > 0 else CB.MAX4)
    if n < 1:
        self.fallbacks += 1
        return ranked[0]
    cands = ranked[:n]
    X0, sc = self.base_input(s)
    vg = self.value_grid()
    land = self.t + self.D
    n_left = sum(1 for d in self.dinos if not any(d["pos"] in tiles and L < land for L, _, tiles in self.falling))
    W = TRAP_W.get(self.mode, CB.PRIOR_W)
    best_t, best_v, best_parts, done = None, -1.0, None, 0
    t_start = time.perf_counter()
    for a in range(0, len(cands), CB.CHUNK):
        if 1000 * (time.perf_counter() - self.tick_t0) + self.unit4 * CB.CHUNK > C.HARD_STOP_MS:
            if done == 0:
                self.fallbacks += 1
                return ranked[0]
            break
        chunk = cands[a:a + CB.CHUNK]
        Xs, masks = [], []
        for cx, cy in chunk:
            X = X0.copy()
            mk = np.zeros((self.H, self.W), bool)
            for dx, dy in self.foot:
                i, j = cy + dy - self.oy, cx + dx - self.ox
                if 0 <= i < self.h and 0 <= j < self.w:
                    X[self.i_new, i, j] = 1
                    mk[i, j] = True
            Xs.append(X); masks.append(mk)
        pd = self.predict4(Xs, sc)
        for k, c in enumerate(chunk):
            P = pd[k][masks[k]]
            E = float(P.sum())
            v = float((P * vg[masks[k]]).sum()) * (1 + 0.5 * max(E - 1, 0))
            if n_left > 0:
                v += FC.CLEAR_BONUS * min(1.0, E / n_left) ** n_left
            cnn, trap = (1 - W) * v, W * self.prior.get(c, 0.0)
            bonus = self.mode_bonus(c)
            f = (cnn + trap) * (1 + bonus)
            if f > best_v:
                best_t, best_v, best_parts = c, f, (cnn, trap, bonus, f)
        done += len(chunk)
    unit = 1000 * (time.perf_counter() - t_start) / max(done, 1)
    self.unit4 = 0.7 * self.unit4 + 0.3 * unit
    self.n4 = done
    self.n_cand = done
    if best_t is None:
        return ranked[0]
    self.prior[best_t] = max(self.prior.get(best_t, 0.0), best_v)   # the fire/no-fire check uses the final score
    self.parts = (best_t, best_parts)
    return best_t


CB.Bot.plan_meteor = plan_meteor_scored        # the N7 / N6 / N4 wrappers call this through super()


class Bot(N12b.Bot):
    def __init__(self):
        super().__init__()
        self.parts = None
        self.m13 = defaultdict(Counter)
        self.ep = defaultdict(list)            # mode -> episode lengths
        self.cur_ep = [None, 0]
        self.p13 = []                          # pending shots: dict
        print(f"[N13] loaded n13_mode_scoring | trap share per mode {TRAP_W} | pack bonus {PACK_BONUS} "
              f"| rex bonus {REX_BONUS} | wide straggler spots {N12b.WIDE}")

    # ---------------------------------------------------------------- soft bonuses
    def mode_bonus(self, c):
        if self.mode == "pack" and PACK_BONUS:
            tri = sum(1 for d in self.dinos if self.names.get(d["id"]) == "Triceratops"
                      and manh(d["pos"], c) <= self.R + 1)
            return PACK_BONUS if tri >= 2 else 0.0
        if self.mode == "rex" and REX_BONUS:
            for r in self.dinos:
                if self.names.get(r["id"]) == "Tyrannosaurus" and manh(r["pos"], c) <= self.R \
                        and any(manh(r["pos"], o["pos"]) == 1 for o in self.dinos if o is not r):
                    return REX_BONUS
        return 0.0

    # ---------------------------------------------------------------- logs
    def choose_actions(self, s):
        self.parts = None
        return super().choose_actions(s)

    def after(self, s, actions, ms):
        mode = self.mode
        if self.cur_ep[0] != mode:
            if self.cur_ep[0] is not None:
                self.ep[self.cur_ep[0]].append(self.cur_ep[1])
            self.cur_ep = [mode, 0]
        self.cur_ep[1] += 1
        st = self.m13[mode]
        st["ticks"] += 1
        for a in actions:
            if isinstance(a, N12.LaunchMeteorAction):
                c = (a.target.x, a.target.y)
                st["shots"] += 1
                src = self.kinds.get(c) if mode == "straggler" else None
                src = src if src in ("wide", "follow") else ("follow" if c in getattr(self, "follow", ()) else "normal")
                cnn, trap, bonus, f = self.parts[1] if self.parts and self.parts[0] == c else (None, None, None, None)
                if cnn is not None:
                    st["scored"] += 1
                    st["cnn"] += cnn
                    st["trap"] += trap
                    st["bonused"] += bonus > 0
                self.p13.append(dict(t=s["currentTick"], L=s["currentTick"] + s["constants"]["meteorDelay"],
                                     tiles=N4.blast(c, s["constants"]["meteorRadius"]), c=c, mode=mode, src=src,
                                     cnn=cnn, trap=trap, bonus=bonus, f=f))
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            keep = []
            for x in self.p13:
                if x["L"] == T - 1:
                    dead = [d for d in p["dinosaurs"] if d["id"] not in now
                            and (d["position"]["x"], d["position"]["y"]) in x["tiles"]]
                    pts = sum(160 / (1 + (d["age"] + 1) / 30) for d in dead)
                    st = self.m13[x["mode"]]
                    st["kills"] += len(dead)
                    st["zero"] += not dead
                    st["pts"] += pts
                    st[f"src_{x['src']}"] += 1
                    st[f"srck_{x['src']}"] += len(dead)
                    if SHOT_LOG:
                        parts = ("cnn - trap - bonus - final - (fallback, no CNN4)" if x["cnn"] is None else
                                 f"cnn {x['cnn']:.3f} trap {x['trap']:.3f} bonus {x['bonus']:.2f} final {x['f']:.3f}")
                        print(f"[N13] shot t={x['t']} {x['mode']} {x['src']} at {x['c']} | {parts} | kills {len(dead)}")
                elif x["L"] >= T:
                    keep.append(x)
            self.p13 = keep
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        if self.cur_ep[0] is not None:
            self.ep[self.cur_ep[0]].append(self.cur_ep[1])
        tot = max(sum(m["ticks"] for m in self.m13.values()), 1)
        for mode in N12.MODES:
            m = self.m13.get(mode)
            if not m or not m["ticks"]:
                continue
            ep = self.ep.get(mode, [])
            sh = max(m["shots"], 1)
            sc = max(m["scored"], 1)
            srcs = " ".join(f"{k} {m['src_' + k]}->{m['srck_' + k]}" for k in ("normal", "follow", "wide") if m["src_" + k])
            print(f"[N13] mode {mode:9} | trap share {TRAP_W.get(mode, 0.3):.2f} | {m['ticks']} ticks ({m['ticks'] / tot:.0%}) "
                  f"| {m['shots']} shots | {m['kills']} kills {m['kills'] / sh:.2f}/shot | zero-kill {m['zero'] / sh:.0%} "
                  f"| points {m['pts']:.0f} | avg episode {sum(ep) / max(len(ep), 1):.1f} ticks ({len(ep)}) "
                  f"| chosen shots avg cnn {m['cnn'] / sc:.3f} trap {m['trap'] / sc:.3f} bonus used {m['bonused']} "
                  f"| shots->kills by source: {srcs}")
