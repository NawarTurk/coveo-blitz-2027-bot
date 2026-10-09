"""
CAGE bot, version B = cage version A (cage_planner.py) but METEOR spots are scored by CNN4:

  CNN4  (models/cnn_lookahead_4/model.pt): map + the meteor we'd launch -> P(dino on each tile) at the impact tick.
        ONE pass per spot (CNN1 needed 4) -> ~4x more spots per tick, no rollout errors piling up.
        expected kills of a spot = sum of P(dino) over its 13 blast tiles   (tested: calibrated,
        its top-10% shots killed 7.5x more than the shots your bots actually fired)
  CNN1  (models/cnn_lookahead_1/model.pt): still used for the VOLCANO (lava appears tick by tick -> roll CNN1 with lava)

  Spots come from the cage logic (all 13 centres per dino + chain spots), ranked by the trap-table prior;
  final score = (1-PRIOR_W) * CNN4 reward + PRIOR_W * prior.  Reward = points 160/(1+age/30) x multi-kill
  + clear bonus, same as before.  Out of time -> best prior spot.

Use it:  cp bot_cage_b.py bot.py
Needs:   cage_planner.py, fast_clear.py, hybrid_planner.py, cnn_model.py, heuristic.py, state_encoder.py,
         models/cnn_lookahead_1/model.pt, models/cnn_lookahead_4/model.pt
Scores:  logs/scores_cage_b.csv
"""
from game_message import *
import os, time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import cage_planner as CA

C, FC = CA.C, CA.FC

# ---- knobs ---------------------------------------------------------------
LOG_GAMES = True
MODEL4 = os.path.join(C.HERE, "models", "cnn_lookahead_4", "model.pt")   # or models/cnn4prev7/moves_cnn4prev7.pt
PRIOR_W = 0.3            # CNN4 is calibrated -> trust it more than the trap table
MAX4 = 60                # max spots scored by CNN4 per tick
CHUNK = 16               # CNN4 batch size (time is checked between chunks)
UNIT_CAP = 3.0           # max ms-per-spot estimate. Without it the estimate can spiral up (a slow tick -> fewer spots
                         # -> a small batch costs about as much as a full one -> higher ms/spot -> ...) until CNN4
                         # never runs again and every shot is a blind fallback (seen on the server: 229 fallbacks)
# --------------------------------------------------------------------------

C.BOT_NAME = "cage_cnn4"
C.LOG_GAMES = LOG_GAMES


class Bot(CA.Bot):
    def __init__(self):
        super().__init__()                       # loads CNN1 (volcano) + warms it up
        ck = torch.load(MODEL4, map_location="cpu", weights_only=False)
        self.ch4 = [str(c) for c in ck["channels"]]
        self.K4 = int(ck.get("history", 1))
        self.net4 = C.DinoCNN(len(self.ch4) + 4, ck["width"], ck["blocks"], len(ck["classes"]))
        self.net4.load_state_dict(ck["state_dict"])
        self.net4 = C.fuse_model(self.net4.eval()).to(C.DEVICE)
        self.scale4 = torch.tensor(np.asarray(ck["scale"], np.float32)).view(1, -1, 1, 1).to(C.DEVICE)
        CH4 = {n: i for i, n in enumerate(self.ch4)}
        self.i_danger, self.i_new = CH4["danger_turns"], CH4["new_meteor"]
        self.i_dxy = [CH4[c] for c in self.ch4 if c.endswith("_dx") or c.endswith("_dy")]
        self.hist4 = {}
        x = torch.zeros(CHUNK, len(self.ch4) + 4, self.H, self.W, device=C.DEVICE)
        with torch.inference_mode():
            self.net4(x)
            t = time.perf_counter()
            for _ in range(3):
                self.net4(x)
        self.unit4 = 1000 * (time.perf_counter() - t) / (3 * CHUNK) * 1.3
        self.n4 = 0
        self.capped = 0

    # ---------------------------------------------------------------- inputs
    def update_hist(self, s):
        alive = set()
        for d in s["dinosaurs"]:
            alive.add(d["id"])
            self.hist4.setdefault(d["id"], deque(maxlen=self.K4 + 1)).append((d["position"]["x"], d["position"]["y"]))
        for k in [k for k in self.hist4 if k not in alive]:
            del self.hist4[k]

    def base_input(self, s):
        X, sc = C.encode_state(s, None, self.prev_pos, self.foot, self.H, self.W)
        if self.K4 > 1:
            extra = np.zeros((2 * (self.K4 - 1), self.H, self.W), np.uint8)
            for d in s["dinosaurs"]:
                i, j = d["position"]["y"] - self.oy, d["position"]["x"] - self.ox
                h = list(self.hist4.get(d["id"], ()))
                if not (0 <= i < self.H and 0 <= j < self.W) or not h:
                    continue
                for k in range(2, self.K4 + 1):
                    if len(h) < k + 1:
                        break
                    (x1, y1), (x0, y0) = h[-k], h[-k - 1]
                    extra[2 * (k - 2), i, j] = int(np.clip(x1 - x0, -1, 1)) + 2
                    extra[2 * (k - 2) + 1, i, j] = int(np.clip(y1 - y0, -1, 1)) + 2
            X = np.concatenate([X, extra], 0)
        return X, sc

    @torch.inference_mode()
    def predict4(self, Xs, sc):
        xb = torch.from_numpy(np.stack(Xs)).to(C.DEVICE).float() / self.scale4
        d = xb[:, self.i_danger]
        xb[:, self.i_danger] = torch.where(d > 0, (self.delay + 1 - d) / self.delay, torch.zeros_like(d))
        for c in self.i_dxy:
            v = xb[:, c]
            xb[:, c] = torch.where(v > 0, v - 2, torch.zeros_like(v))
        B, H, W = xb.shape[0], self.H, self.W
        rows = torch.linspace(0, 1, H, device=C.DEVICE).view(1, 1, H, 1).expand(B, 1, H, W)
        cols = torch.linspace(0, 1, W, device=C.DEVICE).view(1, 1, 1, W).expand(B, 1, H, W)
        ts = torch.full((B, 1, H, W), sc[0] / self.shift, device=C.DEVICE)
        tm = torch.full((B, 1, H, W), sc[1] / 100, device=C.DEVICE)
        p = F.softmax(self.net4(torch.cat([xb, rows, cols, ts, tm], 1)), 1)
        return (1 - p[:, 0]).cpu().numpy()                   # P(dino) per tile at impact

    def value_grid(self):
        """points of a kill on each tile = average age value of the dinos that can reach it"""
        g = np.zeros((self.H, self.W), np.float32)
        if not self.dinos:
            return g
        pos = np.array([d["pos"] for d in self.dinos])
        val = np.array([C.age_value(d["age"] + self.D) for d in self.dinos])
        ii, jj = np.mgrid[0:self.H, 0:self.W]
        X, Y = jj + self.ox, ii + self.oy
        dist = np.abs(X[..., None] - pos[:, 0]) + np.abs(Y[..., None] - pos[:, 1])   # (H, W, n)
        near = dist <= self.D + 1
        cnt = near.sum(-1)
        g = np.where(cnt > 0, (near * val).sum(-1) / np.maximum(cnt, 1), val.mean())
        return g.astype(np.float32)

    # ---------------------------------------------------------------- meteor planner (CNN4)
    def choose_actions(self, s):
        self.update_hist(s)                                 # every tick, so the move history has no gaps
        return super().choose_actions(s)

    def plan_meteor(self, s):
        ranked = self.candidates(s)                         # cage + chain spots, sorted by prior (sets geometry)
        if not ranked:
            return None
        self.static = None
        if self.unit4 > UNIT_CAP:
            self.unit4 = UNIT_CAP
            self.capped += 1
        left = C.TICK_BUDGET_MS - 1000 * (time.perf_counter() - self.tick_t0) - 3
        n = min(MAX4, len(ranked), int(left / self.unit4) if self.unit4 > 0 else MAX4)
        if n < 1:
            self.fallbacks += 1
            return ranked[0]
        cands = ranked[:n]
        X0, sc = self.base_input(s)
        vg = self.value_grid()
        land = self.t + self.D
        n_left = sum(1 for d in self.dinos
                     if not any(d["pos"] in tiles and L < land for L, _, tiles in self.falling))
        best_t, best_v, done = None, -1.0, 0
        t_start = time.perf_counter()
        for a in range(0, len(cands), CHUNK):
            if 1000 * (time.perf_counter() - self.tick_t0) + self.unit4 * CHUNK > C.HARD_STOP_MS:
                if done == 0:
                    self.fallbacks += 1
                    return ranked[0]
                break
            chunk = cands[a:a + CHUNK]
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
                v = (1 - PRIOR_W) * v + PRIOR_W * self.prior.get(c, 0.0)
                if v > best_v:
                    best_t, best_v = c, v
            done += len(chunk)
        unit = 1000 * (time.perf_counter() - t_start) / max(done, 1)
        self.unit4 = 0.7 * self.unit4 + 0.3 * unit
        self.n4 = done
        self.n_cand = done
        if best_t is None:
            return ranked[0]
        self.prior[best_t] = max(self.prior.get(best_t, 0.0), best_v)   # the fire/no-fire check uses the final score
        return best_t

    def finish(self, s):
        super().finish(s)
        print(f"CNN4 spots/tick (last): {self.n4} | ms per spot: {self.unit4:.2f} | ms/spot estimate capped {self.capped} times")
