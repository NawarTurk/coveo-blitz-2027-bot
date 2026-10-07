"""
ZH1 – pure heuristic bot (NO ML: no torch, no CNN/CNN4, no XGBoost, no model files).

For every candidate meteor it runs a SPECIES-CONDITIONED PROBABILISTIC 4-TICK SIMULATION of how each nearby
dinosaur reacts to that meteor (and to the meteors already falling), and scores the shot from the
probability that each dino is still inside the blast at impact.

Per dino, per simulated tick:  legal moves {stay, up, down, left, right}  ->  species-specific logits  ->
softmax -> propagate probability mass (states = position + last move, pruned to top K).
Logit terms (weights per species in SPECIES below, all tunable):
  stay bias (calm), stay penalty when threatened, momentum (repeat last move), move away from threat centres,
  big penalty for stepping into a blast that lands next tick, penalty for staying in any active blast,
  Stegosaurus: avoid trailing edge; Triceratops: herd centroid + higher ground (secondary);
  Velociraptor: stop as soon as it is just outside the blast ("minimum movement"), weak corpse pull;
  T-Rex: prefer tiles with more free neighbours, avoid crowds.
Measured facts used: calm stay rates (Steg 17%, Tri 30%, T-Rex 53%, Raptor 55%), 80% step away at once,
62% run straight, dinos end 3-5 tiles out, walls trap (flee_analysis.py on 550 games).
UNVERIFIED (kept small, tunable): raptor corpse pull (logs showed none), Triceratops herd/high ground.

Simulation timing: meteor launched now lands D ticks later and hits before dinos move; dinos see the new
meteor from the 2nd simulated move (CANDIDATE_SEEN_FROM_STEP). Existing meteors land on their own ticks and
remove mass (killed by them). Rows scrolling away remove mass (scroll death).

Shot value (points):  sum_i v_i p_i (1 + 0.5 sum_{j!=i} p_j)       [exact expected multi-kill under independence]
                    + CLEAR_BONUS * P(every dino dead by impact)
                    + small cage/herding value: low escape entropy of the surviving mass
  v_i = 160 / (1 + age_at_impact/30), x (1 + TREX_RESCUE) if a T-Rex is close (eaten = 0 points)
Candidates: all centres covering a dino, around predicted positions, chain spots on escape paths of dinos
fleeing existing meteors, pair midpoints. Cheap pre-rank -> full simulation of the best within PLAN_MS.
Hold only if the dinos are very likely to die from existing meteors anyway. Volcano (heuristic logic)
when meteor slots are full or no meteor is worth firing.

Use it:  cp bot_zh1.py bot.py
Needs:   heuristic.py (pure python), game_message.py.  requirements: msgspec, websockets
Scores:  logs/scores_zh1.csv ; end-of-game diagnostics printed
"""
from game_message import *
import gc, math, os, time
from collections import deque
from datetime import datetime

import msgspec

import heuristic as H

# ---------------------------------------------------------------- knobs
LOG_GAMES = True
BOT_NAME = "species_heuristic"
PLAN_MS = 45            # stop full simulations after this many ms into the tick
HARD_STOP_MS = 85       # never start more work past this
VOLC_MS = 55            # only run the (slower) volcano scorer if we are below this
K_STATES = 6            # states kept per dino per simulated tick
MIN_STATE_P = 0.01
MAX_FULL = 250          # max candidates fully simulated per tick
CANDIDATE_SEEN_FROM_STEP = 1
PTS, AGE_SCALE = 160.0, 30.0
CLEAR_BONUS = 600.0     # points, x P(board empty at impact)
CLEAR_MAX_DINOS = 12
TREX_EAT_DIST, TREX_RESCUE = 2, 0.3
ENTROPY_W = 0.10        # cage/herding value weight (small)
HOLD_SURVIVORS = 0.5    # hold a free slot only if expected survivors of existing meteors < this
MIN_EV = 5.0            # points: below this a meteor isn't worth a slot
DEADLY = -8.0           # logit for stepping into a blast landing next tick
EDGE_ROWS = 3
SCROLL_SKIP, SCROLL_KEEP = 0.6, 0.25   # P(scroll death before impact) above this -> value x SCROLL_KEEP (still counts for multi/clear)
HERD_RADIUS = 6
PRE_TRAP = [(0, 0.79), (3, 0.47), (7, 0.26), (14, 0.10), (999, 0.036)]   # measured caught rate vs free escapes

SPECIES = {   # logit weights
    "Stegosaurus":   dict(stay=-0.2, stay_threat=-1.5, mom=1.5, away=2.0, inside=-1.0, edge=1.0, herd=0.0, high=0.0,
                          corpse=0.0, stop_safe=0.0, space=0.0, crowd=0.0),
    "Velociraptor":  dict(stay=1.6, stay_threat=-1.5, mom=0.5, away=2.0, inside=-1.5, edge=0.3, herd=0.0, high=0.0,
                          corpse=0.1, stop_safe=1.5, space=0.0, crowd=0.0),
    "Triceratops":   dict(stay=0.5, stay_threat=-1.5, mom=0.7, away=2.0, inside=-1.0, edge=0.3, herd=0.3, high=0.1,
                          corpse=0.0, stop_safe=0.0, space=0.0, crowd=0.0),
    "Tyrannosaurus": dict(stay=1.5, stay_threat=-2.0, mom=0.5, away=2.5, inside=-2.0, edge=0.3, herd=0.0, high=0.0,
                          corpse=0.0, stop_safe=0.0, space=0.4, crowd=0.2),
}
DEFAULT_SP = SPECIES["Triceratops"]
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

# volcano helper (pure python, from heuristic) with the verified reward and no unverified claims
H.LOG_GAMES = False
H.AGE_HALF = AGE_SCALE
H.NE_METEOR_BONUS = H.NE_VOLCANO_BONUS = 0.0
H.TRI_CCW_PROB = 0.5
H.TRI_OFFSET_BONUS = H.RAPTOR_BAIT_BONUS = H.FUTURE_RAPTOR_W = 0.0
H.DOUBLE_BOUNCE_WEIGHT = H.FUNNEL_BONUS = H.REPEAT_IMPACT_BONUS = 0.0


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def p_trap(exits):
    for hi, p in PRE_TRAP:
        if exits <= hi:
            return p
    return PRE_TRAP[-1][1]


class Bot:
    def __init__(self):
        self.hist = {}
        self.prev_pos, self.prev_age = {}, {}
        self.pending = []                      # fired shots waiting for their impact
        self.last_volcano = None
        self.times = []
        self.n_full = []
        self.d = dict(meteors=0, volcano=0, held=0, nothing=0, low_ev=0, chain=0, cage=0, independent=0,
                      kills=0, zero_kill=0, multi2=0, multi3=0, multi4p=0, chain_kills=0, cage_kills=0,
                      pred_kills=0.0, clears=0, slow100=0, slow150=0, scroll_skip=0)
        self.kill_ages, self.wave_ticks = [], []
        self.calib = {b: [0, 0] for b in ("0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75+")}
        self.wave_start = None
        self.biome = None
        if LOG_GAMES:
            os.makedirs("logs", exist_ok=True)
            self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"logs/game_{self.stamp}.jsonl"
            self.log = open(self.log_path, "ab")
        gc.collect(); gc.freeze(); gc.disable()

    # ================================================================ main loop
    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        self.t0 = time.perf_counter()
        s = msgspec.to_builtins(game_message)
        try:
            self.bookkeeping(s)
            actions = self.choose(s)
        except Exception as e:                 # never crash mid-game
            print("species_heuristic error:", repr(e))
            actions = []
        dt = time.perf_counter() - self.t0
        self.times.append(dt)
        if dt > 0.10: self.d["slow100"] += 1
        if dt > 0.15: self.d["slow150"] += 1
        self.prev_pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        self.prev_age = {d["id"]: d["age"] for d in s["dinosaurs"]}
        if LOG_GAMES:
            self.log.write(msgspec.json.encode({"tick": s["currentTick"], "state": game_message, "actions": actions}) + b"\n")
            self.log.flush()
        if s["currentTick"] >= s["constants"]["maxTicks"]:
            self.finish(s)
        return actions

    # ================================================================ diagnostics: resolve fired shots
    def bookkeeping(self, s):
        T = s["currentTick"]
        if self.biome is None:
            self.biome = s.get("biome") or s["map"].get("biome") or "?"
        ids = {d["id"] for d in s["dinosaurs"]}
        for d in s["dinosaurs"]:
            self.hist.setdefault(d["id"], deque(maxlen=3)).append((d["position"]["x"], d["position"]["y"]))
        for k in [k for k in self.hist if k not in ids]:
            del self.hist[k]
        keep = []
        for sh in self.pending:
            if sh["land"] == T - 1:            # landed on the previous state (hits before moves)
                killed = [i for i, p in self.prev_pos.items() if p in sh["tiles"] and i not in ids]
                n = len(killed)
                self.d["kills"] += n
                if n == 0: self.d["zero_kill"] += 1
                if n == 2: self.d["multi2"] += 1
                if n == 3: self.d["multi3"] += 1
                if n >= 4: self.d["multi4p"] += 1
                if sh["type"] == "chain": self.d["chain_kills"] += n
                if sh["type"] == "cage": self.d["cage_kills"] += n
                self.kill_ages += [self.prev_age.get(i, 0) for i in killed]
                self.on_resolve(sh, killed)
                for i, p in sh["pred"].items():
                    b = "0-0.1" if p < .1 else "0.1-0.25" if p < .25 else "0.25-0.5" if p < .5 else "0.5-0.75" if p < .75 else "0.75+"
                    self.calib[b][0] += 1
                    self.calib[b][1] += int(i in killed)
            elif sh["land"] >= T - 1:
                keep.append(sh)
        self.pending = keep
        if self.prev_pos and not ids:
            self.d["clears"] += 1
            if self.wave_start is not None:
                self.wave_ticks.append(T - self.wave_start)
            self.wave_start = None
        if ids and (not self.prev_pos or self.wave_start is None):
            self.wave_start = T

    # ================================================================ per-tick geometry
    def setup(self, s):
        c, m = s["constants"], s["map"]
        self.t = s["currentTick"]
        self.D, self.R, self.shift = c["meteorDelay"], c["meteorRadius"], c["mapShiftInterval"]
        self.ox, self.oy, self.H, self.W = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        self.foot = [(a, b) for a in range(-self.R, self.R + 1) for b in range(-self.R, self.R + 1) if abs(a) + abs(b) <= self.R]
        self.blocked, self.elev = set(), {}
        for i, row in enumerate(m["tiles"]):
            for j, tl in enumerate(row):
                q = (j + self.ox, i + self.oy)
                self.elev[q] = tl["elevation"]
                if tl["isImpassable"] or tl["isMountain"] or tl["hasLava"]:
                    self.blocked.add(q)
        self.corpses = [(p["x"], p["y"]) for p in s["corpses"]]
        self.threats = []                      # (centre, tiles, land tick, is_candidate, visible_from_step)
        for me in s["meteors"]:
            self.threats.append(((me["target"]["x"], me["target"]["y"]),
                                 frozenset((p["x"], p["y"]) for p in me["impactedTiles"]),
                                 self.t + me["turnsUntilImpact"] - 1, False, 0))
        self.dinos = []
        pos_all = []
        for d in s["dinosaurs"]:
            p = (d["position"]["x"], d["position"]["y"])
            h = list(self.hist.get(d["id"], ()))
            last = (h[-1][0] - h[-2][0], h[-1][1] - h[-2][1]) if len(h) >= 2 else (0, 0)
            if last not in MOVES: last = (0, 0)
            age = d["age"] + self.D
            self.dinos.append({"id": d["id"], "name": d["name"], "pos": p, "last": last, "age": d["age"],
                               "v": PTS / (1 + age / AGE_SCALE), "sp": SPECIES.get(d["name"], DEFAULT_SP)})
            pos_all.append(p)
        self.occupied = set(pos_all)
        trex = [d["pos"] for d in self.dinos if d["name"] == "Tyrannosaurus"]
        tris = [d["pos"] for d in self.dinos if d["name"] == "Triceratops"]
        for d in self.dinos:
            if d["name"] != "Tyrannosaurus" and any(manh(d["pos"], q) <= TREX_EAT_DIST + 1 for q in trex):
                d["v"] *= 1 + TREX_RESCUE
            d["herd"] = None
            if d["name"] == "Triceratops":
                near = [q for q in tris if q != d["pos"] and manh(q, d["pos"]) <= HERD_RADIUS]
                if near:
                    d["herd"] = (sum(q[0] for q in near) / len(near), sum(q[1] for q in near) / len(near))
            d["corpse"] = min(self.corpses, key=lambda q: manh(q, d["pos"])) if (self.corpses and d["name"] == "Velociraptor") else None
        self.free_nb = {}

    def in_window(self, x, y):
        return 0 <= y - self.oy < self.H and 0 <= x - self.ox < self.W

    def nfree(self, q):
        v = self.free_nb.get(q)
        if v is None:
            v = sum(1 for a, b in MOVES[1:] if (q[0] + a, q[1] + b) not in self.blocked and self.in_window(q[0] + a, q[1] + b))
            self.free_nb[q] = v
        return v

    # ================================================================ species move model
    def move_probs(self, d, p, lm, active, tau, oy_k):
        sp = d["sp"]
        R = self.R
        near = [th for th in active if manh(p, th[0]) <= R + 1]
        threatened = bool(near)
        in_blast = any(p in th[2 - 1] for th in active)
        d0 = min((manh(p, th[0]) for th in near), default=0)
        out, logits = [], []
        for m in MOVES:
            q = (p[0] + m[0], p[1] + m[1])
            if m != (0, 0):
                if q in self.blocked or not (self.ox <= q[0] < self.ox + self.W) or not (oy_k <= q[1] < oy_k + self.H):
                    continue
            L = 0.0
            if m == (0, 0):
                L += sp["stay"] + (sp["stay_threat"] if in_blast else 0.0)
                if not in_blast and threatened:
                    L += sp["stop_safe"]
            elif m == lm:
                L += sp["mom"]
            for th in active:
                if q in th[1]:
                    L += DEADLY if th[2] == tau + 1 else sp["inside"]
            if near:
                L += sp["away"] * (min(manh(q, th[0]) for th in near) - d0)
            row = q[1] - oy_k
            if row < EDGE_ROWS:
                L -= sp["edge"] * (EDGE_ROWS - row) / EDGE_ROWS
            if d["herd"] is not None and sp["herd"]:
                hx, hy = d["herd"]
                L += sp["herd"] * ((abs(p[0] - hx) + abs(p[1] - hy)) - (abs(q[0] - hx) + abs(q[1] - hy)))
            if sp["high"]:
                L += sp["high"] * (self.elev.get(q, 0) - self.elev.get(p, 0)) / 5.0
            if d["corpse"] is not None and sp["corpse"]:
                L += sp["corpse"] * (manh(p, d["corpse"]) - manh(q, d["corpse"]))
            if sp["space"]:
                L += sp["space"] * (self.nfree(q) - self.nfree(p)) / 4.0
            if sp["crowd"] and m != (0, 0):
                L -= sp["crowd"] * sum(1 for a, b in MOVES[1:] if (q[0] + a, q[1] + b) in self.occupied and (q[0] + a, q[1] + b) != p)
            out.append(m); logits.append(L)
        mx = max(logits)
        ex = [math.exp(l - mx) for l in logits]
        z = sum(ex)
        return [(m, e / z) for m, e in zip(out, ex)]

    def simulate(self, d, threats, start=None, k0=0, oy0=None, acc=(0.0, 0.0, 0.0), snap_at=None):
        """returns kill_by_candidate, dead_by_other, scrolled, final states, snapshot(after step snap_at)"""
        states = dict(start) if start is not None else {(d["pos"], d["last"]): 1.0}
        oy_k = self.oy if oy0 is None else oy0
        kill, other, scroll = acc
        snap = None
        for k in range(k0, self.D + 1):
            tau = self.t + k
            for th in threats:
                if th[2] == tau:
                    rm, ns = 0.0, {}
                    for key, pr in states.items():
                        if key[0] in th[1]:
                            rm += pr
                        else:
                            ns[key] = pr
                    states = ns
                    if th[3]: kill += rm
                    else: other += rm
            if k == snap_at:
                snap = (dict(states), oy_k, (kill, other, scroll))
            if k == self.D:
                break
            active = [th for th in threats if th[2] > tau and k >= th[4]]
            new = {}
            for (p, lm), pr in states.items():
                for m, q in self.move_probs(d, p, lm, active, tau, oy_k):
                    np_ = (p[0] + m[0], p[1] + m[1])
                    key = (np_, m if m != (0, 0) else lm)
                    new[key] = new.get(key, 0.0) + pr * q
            if (tau + 1) % self.shift == 0:
                oy_k += 1
                ns = {}
                for key, pr in new.items():
                    if key[0][1] < oy_k: scroll += pr
                    else: ns[key] = pr
                new = ns
            if len(new) > K_STATES:
                tot = sum(new.values())
                top = sorted(new.items(), key=lambda kv: -kv[1])[:K_STATES]
                kept = sum(pr for _, pr in top)
                new = {k_: pr * tot / kept for k_, pr in top if pr * tot / kept >= MIN_STATE_P} or dict(top[:1])
            states = new
        return kill, other, scroll, states, snap

    # ================================================================ decision
    def choose(self, s):
        c = s["constants"]
        if not s["dinosaurs"]:
            self.d["nothing"] += 1
            return []
        self.setup(s)
        # baseline: every dino with the existing meteors only (+ snapshot where the new meteor becomes visible)
        self.base = {}
        for d in self.dinos:
            kill, other, scroll, st, snap = self.simulate(d, self.threats, snap_at=CANDIDATE_SEEN_FROM_STEP)
            self.base[d["id"]] = {"other": other, "scroll": scroll, "alive": sum(st.values()), "final": st, "snap": snap}
        for d in self.dinos:                   # dinos that will scroll off anyway: little meteor value
            if self.base[d["id"]]["scroll"] > SCROLL_SKIP:
                d["v"] *= SCROLL_KEEP
                self.d["scroll_skip"] += 1
        free_slots = c["maxMeteors"] - len(s["meteors"])
        if free_slots > 0:
            survivors = sum(b["alive"] for b in self.base.values())
            if self.should_hold(s, survivors):
                self.d["held"] += 1
                return self.try_volcano(s, idle=True)
            best = self.plan_meteor(s)
            if best is not None and best["ev"] >= MIN_EV:
                self.fire(best)
                return [LaunchMeteorAction(target=WorldPosition(*best["c"]))]
            self.d["low_ev"] += 1
            return self.try_volcano(s, idle=True)
        return self.try_volcano(s, idle=False)

    def should_hold(self, s, survivors):
        return bool(s["meteors"]) and survivors < HOLD_SURVIVORS

    def on_resolve(self, sh, killed):       # hook for subclasses (ZH2)
        pass

    def candidates(self):
        R, D = self.R, self.D
        spots = set()
        for d in self.dinos:
            x, y = d["pos"]
            for a, b in self.foot:
                spots.add((x + a, y + b))
            for (q, _), pr in self.base[d["id"]]["final"].items():       # around predicted positions
                if pr >= 0.15:
                    for a, b in MOVES:
                        spots.add((q[0] + a, q[1] + b))
        ds = self.dinos
        for i in range(len(ds)):                                         # pair midpoints
            for j in range(i + 1, len(ds)):
                a, b = ds[i]["pos"], ds[j]["pos"]
                if manh(a, b) <= 2 * R + 1:
                    spots.add(((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
        spots = [p for p in spots if self.in_window(*p)]
        # cheap pre-rank: baseline mass in the footprint (no reaction) + measured trap rate for dinos covered now
        pre = {}
        for p in spots:
            v = 0.0
            for d in ds:
                if manh(d["pos"], p) > R + D + 1:
                    continue
                fin = self.base[d["id"]]["final"]
                mass = sum(pr for (q, _), pr in fin.items() if manh(q, p) <= R)
                trap = 0.0
                if manh(d["pos"], p) <= R:
                    x, y = d["pos"]
                    ex = sum(1 for a in range(-3, 4) for b in range(-3 + abs(a), 4 - abs(a))
                             if (x + a, y + b) not in self.blocked and self.in_window(x + a, y + b) and manh((x + a, y + b), p) > R)
                    trap = p_trap(ex)
                v += d["v"] * max(mass, trap)
            pre[p] = v
        return sorted(spots, key=lambda p: -pre[p]), pre

    def plan_meteor(self, s):
        self.evaluated = []                    # every fully simulated candidate (ZH2 rescoring uses it)
        ranked, pre = self.candidates()
        ranked = [p for p in ranked if pre[p] > 0][:MAX_FULL]
        if not ranked:
            return None
        R, D = self.R, self.D
        land = self.t + D
        best, n = None, 0
        for cpos in ranked:
            el = 1000 * (time.perf_counter() - self.t0)
            if el > PLAN_MS or el > HARD_STOP_MS:
                break
            tiles = frozenset((cpos[0] + a, cpos[1] + b) for a, b in self.foot)
            cand = (cpos, tiles, land, True, CANDIDATE_SEEN_FROM_STEP)
            threats = self.threats + [cand]
            ps, vs, dead_all, ent_val = {}, {}, [], 0.0
            chain = False
            for d in self.dinos:
                b = self.base[d["id"]]
                if manh(d["pos"], cpos) > R + D + 1:
                    dead_all.append(1 - b["alive"])
                    continue
                snap = b["snap"]
                if snap is not None:
                    st, oyk, acc = snap
                    kill, other, scroll, fin, _ = self.simulate(d, threats, start=st, k0=CANDIDATE_SEEN_FROM_STEP, oy0=oyk, acc=acc)
                else:
                    kill, other, scroll, fin, _ = self.simulate(d, threats)
                ps[d["id"]], vs[d["id"]] = kill, d["v"]
                dead_all.append(kill + other + scroll)
                if kill > 0.2 and any(d["pos"] in th[1] and th[2] < land for th in self.threats):
                    chain = True
                surv = sum(fin.values())
                if surv > 0.05 and kill > 0.05:                         # escape entropy of the survivors
                    bins = [0.0, 0.0, 0.0, 0.0]
                    for (q, _), pr in fin.items():
                        dx, dy = q[0] - cpos[0], q[1] - cpos[1]
                        bins[(0 if dx >= 0 else 1) if abs(dx) >= abs(dy) else (2 if dy >= 0 else 3)] += pr
                    Hn = -sum(x / surv * math.log(x / surv) for x in bins if x > 0) / math.log(4)
                    ent_val += ENTROPY_W * d["v"] * surv * (1 - Hn)
            if not ps:
                continue
            P = sum(ps.values())
            ev = sum(vs[i] * p * (1 + 0.5 * (P - p)) for i, p in ps.items())
            if len(self.dinos) <= CLEAR_MAX_DINOS:
                pall = 1.0
                for x in dead_all:
                    pall *= min(1.0, x)
                ev += CLEAR_BONUS * pall
            ev += ent_val
            n += 1
            typ = "chain" if chain else ("cage" if max(ps.values()) >= 0.5 else "independent")
            cd = {"c": cpos, "tiles": tiles, "ev": ev, "pred": dict(ps), "type": typ, "P": P, "maxp": max(ps.values())}
            self.evaluated.append(cd)
            if best is None or ev > best["ev"]:
                best = cd
        self.n_full.append(n)
        return best

    def fire(self, best):
        self.d["meteors"] += 1
        self.d[best["type"]] += 1
        self.d["pred_kills"] += best["P"]
        self.pending.append({"land": self.t + self.D, "tiles": best["tiles"], "pred": best["pred"], "type": best["type"],
                             **best.get("log", {})})

    # ================================================================ volcano (heuristic logic)
    def try_volcano(self, s, idle):
        c = s["constants"]
        if (len(s["volcanoes"]) >= c["maxVolcanoes"] or not s["mountains"]
                or 1000 * (time.perf_counter() - self.t0) > VOLC_MS):
            self.d["nothing"] += 1
            return []
        ctx = H.Ctx(s, self.hist, [])
        used = set(ctx.volcanoes)
        mtns = [m for m in ctx.mountains if m not in used and ctx.in_window(*m) and any(manh(m, d["pos"]) <= 8 for d in ctx.dinos)]
        if not mtns:
            self.d["nothing"] += 1
            return []
        base = {d["id"]: H.predict_dino_heuristic(ctx, d, None, keep_dists=True) for d in ctx.dinos}
        mtns.sort(key=lambda m: -sum(1.0 / (1 + manh(m, d["pos"])) for d in ctx.dinos))
        best, bv = None, -1e9
        for m in mtns[:8]:
            if 1000 * (time.perf_counter() - self.t0) > HARD_STOP_MS:
                break
            v = H.score_volcano(ctx, m, base, self.last_volcano)
            if v > bv:
                best, bv = m, v
        thr = H.VOLCANO_MIN_SCORE_WHEN_FULL if not idle else H.VOLCANO_MIN_SCORE
        if best is not None and bv >= thr:
            self.d["volcano"] += 1
            self.last_volcano = best
            return [TriggerVolcanoAction(target=WorldPosition(*best))]
        self.d["nothing"] += 1
        return []

    # ================================================================ end of game
    def finish(self, s):
        ms = [1000 * t for t in self.times]
        d = self.d
        ka = sorted(self.kill_ages)
        print(f"Final score: {s['score']} | avg {sum(ms) / len(ms):.0f} ms/tick, max {max(ms):.0f} ms | "
              f">100ms: {d['slow100']} >150ms: {d['slow150']} | biome {self.biome}")
        km = d["kills"] / max(d["meteors"], 1)
        print(f"meteors {d['meteors']} (chain {d['chain']}, cage {d['cage']}, independent {d['independent']}) | "
              f"kills {d['kills']} = {km:.2f}/meteor | zero-kill {d['zero_kill']} | multi 2:{d['multi2']} 3:{d['multi3']} 4+:{d['multi4p']}")
        print(f"predicted kills {d['pred_kills']:.1f} vs actual {d['kills']} | chain kills {d['chain_kills']} | cage kills {d['cage_kills']} | "
              f"kill age avg {sum(ka) / max(len(ka), 1):.1f} median {ka[len(ka) // 2] if ka else 0} | clears {d['clears']} | "
              f"ticks/wave {sum(self.wave_ticks) / max(len(self.wave_ticks), 1):.0f}")
        print(f"avg P(kill) per fired shot {d['pred_kills'] / max(d['meteors'], 1):.2f} | scroll-skipped dino-ticks {d['scroll_skip']}")
        print(f"held {d['held']} | low-EV {d['low_ev']} | volcano {d['volcano']} | nothing {d['nothing']} | "
              f"full sims/tick avg {sum(self.n_full) / max(len(self.n_full), 1):.0f}")
        print("calibration P(kill) -> actual:", {b: (f"{k / max(n, 1):.2f}", n) for b, (n, k) in self.calib.items()})
        if LOG_GAMES:
            self.log.close()
            path = f"logs/scores_{BOT_NAME}.csv"
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("timestamp,log_file,final_tick,final_score\n")
                f.write(f"{self.stamp},{self.log_path},{s['currentTick']},{s['score']}\n")
