"""
N3 chain follow = cage_cnn4 (Cage B, our best) + an explicit "will it survive?" check + follow-up spots.

Why: in a server game (v32, 14,741) cage_cnn4 held fire for 182 ticks because every live dino stood under a
falling meteor ("doomed"), but 71% of those dinos were still alive 5 ticks later.

The change (CNN4 scoring untouched):
  IF every live dino is inside a falling meteor's blast and a meteor slot is free:
      P(survive) of each dino = CNN4 with no new meteor -> sum of P(dino) on the tiles it can reach at our
                                impact tick (CNN4 already knows the falling meteors)
      IF any dino has P(survive) >= SURVIVE_MIN  -> FIRE: run the normal planner (with follow-up spots)
      ELSE                                       -> HOLD (like cage_cnn4)
  Follow-up spots: while a meteor is falling, spots where a fleeing dino should stop (3 tiles from the blast
  centre straight away from it, +-1 sideways, and 4 tiles out), scored FIRST by CNN4.

Logs (team log, no per-tick trace):
  [N3] t=..  doomed [id species (x,y) P(survive)] -> HOLD / FIRE        every all-doomed tick
  [N3] t=..  fired (x,y) follow-up=yes/no                               the shot after a FIRE
  [N3] t=..  outcome of t=..: HOLD/FIRE, P .. -> survived / died, shot killed k
  [N3] summary: hold right/wrong, fire right/wrong, calibration of P(survive), follow-up kills
  + the [REPORT] lines of cage_cnn4_report (kills/meteor, wave timing, score)

Use: ./scripts/ship.sh n3_chain_follow
"""
from game_message import *

import numpy as np

import cage_cnn4_report as RP

CB, CA, C, FC = RP.CB, RP.CA, RP.C, RP.FC

# ---- knobs ---------------------------------------------------------------
SURVIVE_MIN = 0.3        # fire when some "doomed" dino has at least this chance to survive
STOP_DIST = (3, 4)       # follow-up spot centres this far (Manhattan) from the falling meteor's centre
SIDE = True              # also +-1 tile sideways at distance 3
MAX_GAP = 3              # our meteor must land 0..MAX_GAP ticks after the falling one
LOG_EACH = True          # one [N3] line per all-doomed tick + its outcome
# --------------------------------------------------------------------------

FC.HOLD_WHEN_ALL_DOOMED = True          # kept on; all_doomed() below decides per tick
RP.TRACE = False                        # keep the [REPORT] lines, drop the per-tick trace
C.BOT_NAME = "n3_chain_follow"
BINS = (0.1, 0.3, 0.5, 0.7, 1.01)


def sgn(v):
    return (v > 0) - (v < 0)


def blast(c, R):
    return {(c[0] + dx, c[1] + dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1) if abs(dx) + abs(dy) <= R}


class Bot(RP.Bot):
    def __init__(self):
        super().__init__()
        self.n3 = dict(spots=0, cut=0, fired=0, kills=0, hold_ok=0, hold_bad=0, fire_ok=0, fire_bad=0,
                       fire_kills=0)
        self.calib = {b: [0, 0] for b in BINS}       # P bin -> [cases, survived]
        self.follow = set()
        self.fu_pending = []                          # (landing tick, blast tiles) of follow-up shots
        self.decisions = []                           # open "all doomed" decisions waiting for their outcome
        self.dec_now = None

    # ---------------------------------------------------------------- 1. explicit survival check
    def survival(self, s):
        """P(alive at our impact tick) per dino, from ONE CNN4 pass without a new meteor"""
        self.setup(s)
        X0, sc = self.base_input(s)
        P = self.predict4([X0], sc)[0]
        ii, jj = np.mgrid[0:self.H, 0:self.W]
        X, Y = jj + self.ox, ii + self.oy
        out = {}
        for d in self.dinos:
            near = (np.abs(X - d["pos"][0]) + np.abs(Y - d["pos"][1])) <= self.D + 1
            out[d["id"]] = float(min(1.0, P[near].sum()))
        return out

    def all_doomed(self, s):
        """called by the cage planner: True = hold fire this tick"""
        self.dec_now = None
        if not super().all_doomed(s):
            return False
        if len(s["meteors"]) >= s["constants"]["maxMeteors"]:
            return True                                # no free slot anyway
        try:
            ps = self.survival(s)
        except Exception as e:
            print("[N3] survival error:", repr(e))
            return True                                # same as cage_cnn4
        if not ps:
            return True
        fire = max(ps.values()) >= SURVIVE_MIN
        T = s["currentTick"]
        names = {d["id"]: RP.SP.get(d["name"], "?") for d in s["dinosaurs"]}
        pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        dec = dict(t=T, ps=ps, fire=fire, target=None, follow=False)
        self.decisions.append(dec)
        self.dec_now = dec
        if LOG_EACH:
            txt = " ".join(f"[{i} {names.get(i, '?')} {pos.get(i)} P={p:.2f}]" for i, p in ps.items())
            print(f"[N3] t={T} doomed {txt} -> {'FIRE' if fire else 'HOLD'}")
        return not fire

    # ---------------------------------------------------------------- 2. follow-up spots
    def follow_spots(self):
        land = self.t + self.D
        out = []
        for L, cen, tiles in self.falling:
            if not (0 <= land - L <= MAX_GAP):
                continue
            for d in self.dinos:
                x, y = d["pos"]
                if RP.manh(d["pos"], cen) > self.R + 1:
                    continue
                vx, vy = x - cen[0], y - cen[1]
                if (vx, vy) == (0, 0):                            # on the centre: use its last move
                    h = list(self.hist4.get(d["id"], ()))
                    if len(h) >= 2:
                        vx, vy = h[-1][0] - h[-2][0], h[-1][1] - h[-2][1]
                if (vx, vy) == (0, 0):
                    continue
                ax, ay = (sgn(vx), 0) if abs(vx) >= abs(vy) else (0, sgn(vy))
                for k in STOP_DIST:
                    px, py = x, y
                    while RP.manh((px, py), cen) < k:
                        px, py = px + ax, py + ay
                    pts = [(px, py)]
                    if SIDE and k == STOP_DIST[0]:
                        pts += [(px + ay, py + ax), (px - ay, py - ax)]
                    for p in pts:
                        if p not in tiles and self.in_window(*p) and p not in out:
                            out.append(p)
        return out

    def candidates(self, s):
        ranked = super().candidates(s)
        self.plain_first = ranked[0] if ranked else None
        fs = self.follow_spots()
        self.follow = set(fs)
        if fs:
            pos = {p: i for i, p in enumerate(ranked)}
            self.n3["spots"] += len(fs)
            self.n3["cut"] += sum(1 for p in fs if pos.get(p, 10 ** 9) >= CB.MAX4)
            for p in fs:
                if p not in self.prior:
                    self.prior[p] = self.cage_prior(p)
        return fs + [p for p in ranked if p not in self.follow]

    def plan_meteor(self, s):
        fb = self.fallbacks
        t = super().plan_meteor(s)
        if self.fallbacks > fb:                  # out of time: same fallback as cage_cnn4
            t = self.plain_first
        return t

    # ---------------------------------------------------------------- 3. logs
    def after(self, s, actions, ms):
        T = s["currentTick"]
        R, D = s["constants"]["meteorRadius"], s["constants"]["meteorDelay"]
        for a in actions:
            if not isinstance(a, LaunchMeteorAction):
                continue
            c = (a.target.x, a.target.y)
            fu = c in self.follow
            if fu:
                self.n3["fired"] += 1
                self.fu_pending.append((T + D, blast(c, R)))
            if self.dec_now is not None and self.dec_now["t"] == T:
                self.dec_now["target"], self.dec_now["follow"] = c, fu
                self.dec_now["tiles"] = blast(c, R)
                if LOG_EACH:
                    print(f"[N3] t={T} fired {c} follow-up={'yes' if fu else 'no'}")
        super().after(s, actions, ms)

    def watch(self, s):
        p, T = self.prev_s, s["currentTick"]
        if p is not None and T == p["currentTick"] + 1:
            now = {d["id"] for d in s["dinosaurs"]}
            dead_here = [d for d in p["dinosaurs"] if d["id"] not in now]
            for L, tiles in self.fu_pending:
                if L == T - 1:
                    self.n3["kills"] += sum(1 for d in dead_here if (d["position"]["x"], d["position"]["y"]) in tiles)
            self.fu_pending = [(L, tl) for L, tl in self.fu_pending if L >= T]
            # outcomes: a decision at tick t is judged at t + D + 1 (our meteor would have landed at t + D)
            D = s["constants"]["meteorDelay"]
            keep = []
            for dec in self.decisions:
                if dec.get("target") and T - 1 == dec["t"] + D:
                    dec["k"] = sum(1 for d in dead_here if (d["position"]["x"], d["position"]["y"]) in dec["tiles"])
                if T < dec["t"] + D + 1:
                    keep.append(dec)
                    continue
                if T > dec["t"] + D + 1:                   # a tick was skipped: judge with what we have
                    pass
                self.judge(dec, now, T)
            self.decisions = keep
        super().watch(s)

    def judge(self, dec, now, T):
        """survived = still alive after our meteor would have landed (killed by OUR follow-up counts as survived
        the doom, i.e. firing was right)"""
        k = dec.get("k", 0)
        surv = {i: (i in now) for i in dec["ps"]}
        any_surv = any(surv.values()) or k > 0
        for i, pr in dec["ps"].items():
            b = next(b for b in BINS if pr < b)
            self.calib[b][0] += 1
            self.calib[b][1] += int(surv[i] or k > 0)
        n = self.n3
        if dec["fire"]:
            n["fire_ok" if any_surv else "fire_bad"] += 1
            n["fire_kills"] += k
        else:
            n["hold_bad" if any_surv else "hold_ok"] += 1
        if LOG_EACH:
            shot = f", shot {dec['target']} killed {k}" if dec.get("target") else ""
            print(f"[N3] t={T} outcome of t={dec['t']}: {'FIRE' if dec['fire'] else 'HOLD'} "
                  f"maxP={max(dec['ps'].values()):.2f} -> {'survived' if any_surv else 'died'}{shot}")

    def finish(self, s):
        super().finish(s)
        n = self.n3
        nh, nf = n["hold_ok"] + n["hold_bad"], n["fire_ok"] + n["fire_bad"]
        print(f"[N3] summary | all-doomed ticks judged {nh + nf} | HOLD {nh}: right (died) {n['hold_ok']}, "
              f"wrong (survived) {n['hold_bad']} | FIRE {nf}: right (survived) {n['fire_ok']}, "
              f"wasted (died anyway) {n['fire_bad']} | kills by those shots {n['fire_kills']}")
        lo = 0.0
        parts = []
        for b in BINS:
            c, sv = self.calib[b]
            parts.append(f"P {lo:.1f}-{min(b, 1):.1f}: {sv}/{c} survived" + (f" ({100 * sv / c:.0f}%)" if c else ""))
            lo = b
        print("[N3] calibration of P(survive) | " + " | ".join(parts))
        print(f"[N3] follow-up spots {n['spots']} | would be cut in cage_cnn4 {n['cut']} "
              f"({100 * n['cut'] / max(n['spots'], 1):.0f}%) | follow-up shots {n['fired']} | kills {n['kills']} "
              f"= {n['kills'] / max(n['fired'], 1):.2f}/shot")
