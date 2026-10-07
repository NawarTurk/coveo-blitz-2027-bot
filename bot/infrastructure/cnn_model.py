"""
CNN bot = the v2 ML bot (same planner: try meteor spots, simulate dinos until impact,
pick the spot with the most valuable predicted kills) but movement comes from the CNN
(models/cnn_lookahead_1/model.pt) instead of XGBoost. One CNN pass predicts EVERY dino on the map,
so each simulated scenario costs one 20x20 "image" per step.

Use it:  cp cnn_model.py bot.py     (needs state_encoder.py + models/cnn_lookahead_1/model.pt + torch)
Scores:  logs/scores_cnn.csv
"""
from game_message import *
from datetime import datetime
import os, time

import msgspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from state_encoder import encode_state, CHANNELS, CH

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bot/ (models live in bot/models)

# ---- knobs ---------------------------------------------------------------
MAX_CANDIDATES = 25     # upper limit of meteor spots evaluated per tick
MIN_CANDIDATES = 1
TICK_BUDGET_MS = 90     # server allows ~150 ms per tick; stay well under it
HARD_STOP_MS = 115      # past this mid-plan: abort and send the quick fallback
AGE_HALF = 20           # value(age) = 1 / (1 + age/AGE_HALF)
LOG_GAMES = True
BOT_NAME = "cnn"        # scores go to logs/scores_cnn.csv
# CNN test accuracy per species -> prefer predictable targets when picking spots
SPECIES_TRUST = {"Tyrannosaurus": 0.84, "Velociraptor": 0.80, "Triceratops": 0.70, "Stegosaurus": 0.67}
USE_CONFIDENCE = True   # weight each predicted kill by the model's probability of that path
DEVICE = os.environ.get("BOT_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")  # BOT_DEVICE=cpu to test server speed
TORCH_THREADS = int(os.environ.get("BOT_THREADS", "1"))   # 1 = safest on a shared server CPU (more threads fight each other)
# --------------------------------------------------------------------------

DYNAMIC = [CH[c] for c in CHANNELS if c in ("danger_turns", "new_meteor", "age", "prev_dx", "prev_dy") or c.startswith("sp_")]


def age_value(age):
    return 1.0 / (1.0 + age / AGE_HALF)


# ---- same architecture as train_cnn1.py (weights load by name) ----
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class DinoCNN(nn.Module):
    def __init__(self, cin, width, blocks, ncls):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, ncls, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def _fuse(conv, bn):
    """fold BatchNorm into the preceding conv (eval mode): same output, ~20% faster"""
    w = conv.weight
    std = torch.sqrt(bn.running_var + bn.eps)
    f = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding, bias=True)
    f.weight.data = w * (bn.weight / std).view(-1, 1, 1, 1)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    f.bias.data = (b - bn.running_mean) * bn.weight / std + bn.bias
    return f


def fuse_model(net):
    net.stem[0] = _fuse(net.stem[0], net.stem[1]); net.stem[1] = nn.Identity()
    for blk in net.blocks:
        blk.c1 = _fuse(blk.c1, blk.b1); blk.b1 = nn.Identity()
        blk.c2 = _fuse(blk.c2, blk.b2); blk.b2 = nn.Identity()
    return net


class Bot:
    def __init__(self):
        torch.set_num_threads(TORCH_THREADS)
        ck = torch.load(os.path.join(HERE, "models", "cnn_lookahead_1", "model.pt"), map_location="cpu", weights_only=False)
        self.C = len(ck["channels"])
        assert list(ck["channels"]) == CHANNELS, "state_encoder.py channels differ from the trained model"
        self.net = DinoCNN(self.C + 4, ck["width"], ck["blocks"], len(ck["classes"]))
        self.net.load_state_dict(ck["state_dict"])
        self.net = fuse_model(self.net.eval()).to(DEVICE)
        self.scale = torch.tensor(np.asarray(ck["scale"], np.float32)).view(1, -1, 1, 1).to(DEVICE)
        self.moves = [tuple(map(int, str(c).split(","))) for c in ck["classes"]]
        self.H, self.W = ck["H"], ck["W"]
        self.prev_pos = {}
        self.times = []
        # warm up torch and measure cost per scenario-step, so tick 1 is already fast and sized right
        with torch.inference_mode():
            x = torch.zeros(8, self.C + 4, self.H, self.W, device=DEVICE)
            self.net(x)
            t = time.perf_counter()
            for _ in range(3):
                self.net(x)
        self.unit_ms = 1000 * (time.perf_counter() - t) / (3 * 8) * 1.3   # first guess; refined every tick
        self.n_cand = MAX_CANDIDATES
        self.fallbacks = 0
        if LOG_GAMES:
            os.makedirs("logs", exist_ok=True)
            self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = f"logs/game_{self.stamp}.jsonl"
            self.log = open(self.log_path, "ab")

    # ---------------------------------------------------------------- main loop
    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        t0 = time.perf_counter()
        self.tick_t0 = t0
        if game_message.lastTickErrors:
            print(game_message.currentTick, game_message.lastTickErrors)
        s = msgspec.to_builtins(game_message)
        try:
            actions = self.choose_actions(s)
        except Exception as e:          # never crash mid-game
            print("planner error:", repr(e))
            actions = []
        self.prev_pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
        self.times.append(time.perf_counter() - t0)
        if LOG_GAMES:
            rec = {"tick": s["currentTick"], "state": game_message, "actions": actions}
            self.log.write(msgspec.json.encode(rec) + b"\n")
            self.log.flush()
        if s["currentTick"] >= s["constants"]["maxTicks"]:
            self.finish(s)
        return actions

    def finish(self, s):
        print(f"Final score: {s['score']} | avg {1000*np.mean(self.times):.0f} ms/tick, "
              f"max {1000*np.max(self.times):.0f} ms | >150ms ticks: {sum(t > 0.15 for t in self.times)} "
              f"| fallbacks: {self.fallbacks} | spots/tick now: {self.n_cand}")
        if LOG_GAMES:
            self.log.close()
            path = f"logs/scores_{BOT_NAME}.csv"
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("timestamp,log_file,final_tick,final_score\n")
                f.write(f"{self.stamp},{self.log_path},{s['currentTick']},{s['score']}\n")

    # ---------------------------------------------------------------- decision
    def choose_actions(self, s):
        c = s["constants"]
        if s["dinosaurs"] and len(s["meteors"]) < c["maxMeteors"]:
            target = self.plan_meteor(s)
            if target is not None:
                return [LaunchMeteorAction(target=WorldPosition(*target))]
        return self.maybe_volcano(s)

    def footprint(self, r):
        return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if abs(dx) + abs(dy) <= r]

    def in_window(self, x, y):
        return 0 <= y - self.oy < self.h and 0 <= x - self.ox < self.w

    def candidates(self, s):
        r = s["constants"]["meteorRadius"]
        dinos = [d for d in s["dinosaurs"] if self.in_window(d["position"]["x"], d["position"]["y"])]
        spots = set()
        for d in dinos:
            x, y = d["position"]["x"], d["position"]["y"]
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]:
                if self.in_window(x + dx, y + dy):
                    spots.add((x + dx, y + dy))
        pos = np.array([[d["position"]["x"], d["position"]["y"]] for d in dinos])
        ages = np.array([d["age"] for d in dinos])
        trust = np.array([SPECIES_TRUST.get(d["name"], 0.6) for d in dinos])

        def prio(p):
            near = np.abs(pos - p).sum(1) <= r + 2
            return (age_value(ages[near]) * trust[near]).sum()
        return sorted(spots, key=prio, reverse=True)

    # ---------------------------------------------------------------- CNN
    def encode_sim(self, sm, tau, launch):
        """static map layers (once per tick) + this scenario's dinos/meteors at time tau"""
        X = self.static.copy()
        X[DYNAMIC] = 0
        ox, oy = self.ox, self.oy
        for m in sm["meteors"]:
            t = m["turnsUntilImpact"]
            for (a, b) in m["tiles"]:
                i, j = b - oy, a - ox
                if 0 <= i < self.h and 0 <= j < self.w:
                    cur = X[CH["danger_turns"], i, j]
                    X[CH["danger_turns"], i, j] = t if cur == 0 else min(cur, t)
        if launch is not None:
            for dx, dy in self.foot:
                i, j = launch[1] + dy - oy, launch[0] + dx - ox
                if 0 <= i < self.h and 0 <= j < self.w:
                    X[CH["new_meteor"], i, j] = 1
        for i_d, d in sm["dinos"].items():
            x, y = d["position"]["x"], d["position"]["y"]
            i, j = y - oy, x - ox
            ch = CH.get(f"sp_{d['name']}")
            if ch is None or not (0 <= i < self.h and 0 <= j < self.w):
                continue
            X[ch, i, j] = 1
            X[CH["age"], i, j] = min(d["age"], 255)
            pp = sm["prev"].get(i_d)
            if pp is not None:
                X[CH["prev_dx"], i, j] = int(np.clip(x - pp[0], -1, 1)) + 2
                X[CH["prev_dy"], i, j] = int(np.clip(y - pp[1], -1, 1)) + 2
        sc = ((self.shift - tau % self.shift) % self.shift, tau % 100)
        return X, sc

    @torch.inference_mode()
    def predict(self, Xs, scs):
        """batch of raw maps -> move probabilities (B, ncls, H, W), same preprocessing as train_cnn1.py"""
        xb = torch.from_numpy(np.stack(Xs)).to(DEVICE).float() / self.scale
        sc = torch.tensor(scs, dtype=torch.float32, device=DEVICE)
        d = xb[:, CH["danger_turns"]]
        xb[:, CH["danger_turns"]] = torch.where(d > 0, (self.delay + 1 - d) / self.delay, torch.zeros_like(d))
        for c in ("prev_dx", "prev_dy"):
            v = xb[:, CH[c]]
            xb[:, CH[c]] = torch.where(v > 0, v - 2, torch.zeros_like(v))
        B, H, W = xb.shape[0], self.H, self.W
        rows = torch.linspace(0, 1, H, device=DEVICE).view(1, 1, H, 1).expand(B, 1, H, W)
        cols = torch.linspace(0, 1, W, device=DEVICE).view(1, 1, 1, W).expand(B, 1, H, W)
        tshift = (sc[:, 0] / self.shift).view(B, 1, 1, 1).expand(B, 1, H, W)
        tmod = (sc[:, 1] / 100).view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([xb, rows, cols, tshift, tmod], 1)
        return F.softmax(self.net(x), 1).cpu().numpy()

    # ---------------------------------------------------------------- planner
    def plan_meteor(self, s):
        c = s["constants"]
        m = s["map"]
        self.ox, self.oy, self.h, self.w = m["origin"]["x"], m["origin"]["y"], m["height"], m["width"]
        self.delay, self.shift = c["meteorDelay"], c["mapShiftInterval"]
        self.foot = self.footprint(c["meteorRadius"])
        t_start = time.perf_counter()
        ranked = self.candidates(s)
        if not ranked:
            return None
        steps = self.delay                                   # CNN passes per plan
        left = TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 5
        n = int(left / (steps * self.unit_ms)) if self.unit_ms > 0 else MAX_CANDIDATES
        if n < MIN_CANDIDATES:               # can't afford even one spot this tick: answer instantly
            self.n_cand = 0
            self.fallbacks += 1
            return ranked[0]
        n = min(MAX_CANDIDATES, n)
        self.n_cand = n
        cands = ranked[:n]
        t_plan = time.perf_counter()
        self.static, _ = encode_state(s, None, {}, self.foot, self.H, self.W)

        mets0 = [{"tiles": [(p["x"], p["y"]) for p in me["impactedTiles"]], "turnsUntilImpact": me["turnsUntilImpact"],
                  "mine": False} for me in s["meteors"]]
        sims = []
        for cx, cy in cands:
            dinos = {d["id"]: dict(d, position=dict(d["position"])) for d in s["dinosaurs"]}
            sims.append({"target": (cx, cy), "dinos": dinos, "meteors": [dict(mm) for mm in mets0],
                         "prev": dict(self.prev_pos), "kills": [], "conf": {i: 1.0 for i in dinos},
                         "landed": False})

        tick = s["currentTick"]
        for step in range(self.delay + 2):
            n_active = sum(1 for sm in sims if not sm["landed"])
            if 1000 * (time.perf_counter() - self.tick_t0) + self.unit_ms * n_active > HARD_STOP_MS:
                self.fallbacks += 1          # next CNN pass would not fit: answer now with the densest spot
                self.unit_ms *= 1.05         # and be a bit more careful next tick
                return ranked[0]
            active = [sm for sm in sims if not sm["landed"]]
            if not active:
                break
            for sm in active:                          # 1 turn left -> hits before the dinos move
                self.land_meteors(sm)
            active = [sm for sm in active if not sm["landed"]]
            if not active:
                break
            # ---- one CNN pass for all scenarios
            Xs, scs = [], []
            for sm in active:
                X, sc = self.encode_sim(sm, tick + step, sm["target"] if step == 0 else None)
                Xs.append(X); scs.append(sc)
            probs = self.predict(Xs, scs)
            for k, sm in enumerate(active):
                moves = {}
                for i_d, d in sm["dinos"].items():
                    i, j = d["position"]["y"] - self.oy, d["position"]["x"] - self.ox
                    if 0 <= i < self.h and 0 <= j < self.w:
                        p = probs[k, :, i, j]
                        b = int(p.argmax())
                        moves[i_d] = self.moves[b]
                        if USE_CONFIDENCE:
                            sm["conf"][i_d] *= float(p[b])
                self.apply_moves(sm, moves)
                self.tick_down(sm, step)

        # learn the real cost per (spot x CNN pass), including encoding and bookkeeping
        unit = 1000 * (time.perf_counter() - t_plan) / max(1, len(cands) * steps)
        self.unit_ms = 0.7 * self.unit_ms + 0.3 * unit
        best_t, best_v = None, -1
        for sm in sims:
            kills = sm["kills"]
            exp_kills = sum(cf for _, cf in kills)
            v = sum(age_value(a) * cf for a, cf in kills) * (1 + 0.5 * max(exp_kills - 1, 0))
            if v > best_v:
                best_t, best_v = sm["target"], v
        return best_t if best_v > 0 else cands[0]

    def blocked(self, x, y):
        i, j = y - self.oy, x - self.ox
        if not (0 <= i < self.h and 0 <= j < self.w):
            return True
        return bool(self.static[CH["impassable"], i, j] or self.static[CH["lava"], i, j])

    def apply_moves(self, sm, moves):
        dinos = sm["dinos"]
        sm["prev"] = {i: (d["position"]["x"], d["position"]["y"]) for i, d in dinos.items()}
        occupied = set(sm["prev"].values())
        for did, (dx, dy) in moves.items():
            d = dinos.get(did)
            if d is None or (dx, dy) == (0, 0):
                continue
            x, y = d["position"]["x"], d["position"]["y"]
            nx, ny = x + dx, y + dy
            if self.blocked(nx, ny) or (nx, ny) in occupied:
                continue
            occupied.discard((x, y)); occupied.add((nx, ny))
            d["position"]["x"], d["position"]["y"] = nx, ny
        for d in dinos.values():
            d["age"] += 1

    def land_meteors(self, sm):
        keep = []
        for m in sm["meteors"]:
            if m["turnsUntilImpact"] <= 1:
                zone = set(m["tiles"])
                for i in [i for i, d in sm["dinos"].items() if (d["position"]["x"], d["position"]["y"]) in zone]:
                    if m["mine"]:
                        sm["kills"].append((sm["dinos"][i]["age"], sm["conf"].get(i, 1.0)))
                    del sm["dinos"][i]
                if m["mine"]:
                    sm["landed"] = True
            else:
                keep.append(m)
        sm["meteors"] = keep

    def tick_down(self, sm, step):
        for m in sm["meteors"]:
            m["turnsUntilImpact"] -= 1
        if step == 0:                               # our launch appears next tick with turnsUntilImpact = delay
            tx, ty = sm["target"]
            tiles = [(tx + dx, ty + dy) for dx, dy in self.foot if self.in_window(tx + dx, ty + dy)]
            sm["meteors"].append({"tiles": tiles, "turnsUntilImpact": self.delay, "mine": True})

    # ---------------------------------------------------------------- volcano (same as v2)
    def maybe_volcano(self, s):
        if s["volcanoes"] or not s["mountains"] or not s["dinosaurs"]:
            return []
        pos = np.array([[d["position"]["x"], d["position"]["y"]] for d in s["dinosaurs"]])
        best, best_n = None, 1
        for m in s["mountains"]:
            n = int((np.abs(pos - (m["x"], m["y"])).sum(1) <= 4).sum())
            if n > best_n:
                best, best_n = m, n
        if best is None:
            return []
        return [TriggerVolcanoAction(target=WorldPosition(best["x"], best["y"]))]
