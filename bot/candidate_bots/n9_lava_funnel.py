"""
N9 lava funnel = N8 future spots (all frozen) + ONE change: the volcano is chosen to BLOCK ESCAPE ROUTES.

Why: our strongest measured fact is that dinos with no free escape tile are caught 60% of the time, with
15+ escape tiles only 3%. Lava blocks tiles. The site's "lava funnel" idea is the one strategy claim our
data supports (the others - NE quadrant, CCW herds, double-bounce - are false on 10 server games).

Before (N4-N8): volcano score = 70% "lava kills" (CNN1 rollout) + 30% prior, and inside the prior the
escape tiles cut for dinos under a falling meteor counted only CUT_W = 0.5  ->  funnel value ~ 15% weight.
N9: score = 70% lava kills + 30% kill prior + FUNNEL_W x funnel value (counted in full, outside the mix).
  funnel value = for every dino standing in a falling meteor's blast:
                 (catch rate with the lava - catch rate without) x its kill value,
                 counting only lava that arrives BEFORE the meteor lands (trap table).
Meteors, targeting, timing, follow-ups, survival rule: unchanged.

Logs: [N9] summary: volcanoes, funnel volcanoes, funnel targets and how many of them died + all earlier logs.
Use: ./scripts/ship.sh n9_lava_funnel
"""
import time

import n4_survival_follow as N4
import n8_future_spots as N8

CB, C = N4.CB, N4.C
CA = CB.CA
print = N4.print                      # flushed print

# ---- knobs ---------------------------------------------------------------
FUNNEL_W = 1.0           # weight of the funnel value (N4-N8: effectively ~0.15)
# --------------------------------------------------------------------------

C.BOT_NAME = "n9_lava_funnel"


class Bot(N8.Bot):
    def __init__(self):
        super().__init__()
        self.n9 = dict(volc=0, funnel=0, targets=0, died=0)
        self.funnel_pending = []          # (meteor landing tick, dino ids the lava trapped)
        print(f"[N9] loaded n9_lava_funnel | FUNNEL_W {FUNNEL_W}")

    def split_prior(self, lava):
        """(kill prior, funnel value, ids of dinos whose escape the lava cuts)"""
        kill, funnel, ids = 0.0, 0.0, []
        for d in self.dinos:
            a = lava.get(d["pos"])
            if a is not None and a <= 2:
                kill += 0.5 * C.age_value(d["age"] + a)
            for L, _, tiles in self.falling:
                if d["pos"] in tiles:
                    esc = [q for q in d["reach"] if q not in tiles]
                    if esc:
                        before = CA.p_trap(len(esc))
                        after = CA.p_trap(sum(1 for q in esc if lava.get(q, 99) > L - self.t))
                        gain = (after - before) * C.age_value(d["age"] + max(L - self.t, 0))
                        if gain > 0:
                            funnel += gain
                            ids.append((L, d["id"]))
                    break
        return kill, funnel, ids

    def plan_volcano(self, s, threshold=CA.VOLC_MIN):
        self.setup(s)
        ctx = CA.H.Ctx(s, self.heur.hist, self.heur.recent_impacts)
        used = set(ctx.volcanoes)
        mtns = [m for m in ctx.mountains if m not in used and ctx.in_window(*m)
                and any(CA.manh(m, d["pos"]) <= 8 for d in self.dinos)]
        if not mtns:
            self.used["v_no_mountain_near"] = self.used.get("v_no_mountain_near", 0) + 1
            return None
        scored = []
        for m in mtns:
            lava = CA.H.simulate_lava(ctx, m, CA.LAVA_STEPS)
            if lava:
                kill, funnel, ids = self.split_prior(lava)
                scored.append((CA.PRIOR_W * kill + FUNNEL_W * funnel, m, lava, kill, funnel, ids))
        if not scored:
            return None
        scored.sort(key=lambda z: -z[0])
        top = scored[:CA.VOLC_CANDS]
        left = C.HARD_STOP_MS - 1000 * (time.perf_counter() - self.tick_t0)
        if left < self.unit_ms * len(top) * CA.LAVA_STEPS * 1.3:
            best = top[0]
            ok = best[0] > 0 if threshold <= 0 else best[0] >= threshold
            return self.pick(best) if ok else None
        cnn = self.lava_rollout(s, [(z[0], z[1], z[2]) for z in top])
        best, best_v = None, -1.0
        for z, kv in zip(top, cnn):
            v = (1 - CA.PRIOR_W) * kv + CA.PRIOR_W * z[3] + FUNNEL_W * z[4]
            if v > best_v:
                best, best_v = z, v
        ok = best_v > 0 if threshold <= 0 else best_v >= threshold
        return self.pick(best) if ok else None

    def pick(self, z):
        self.n9["volc"] += 1
        if z[4] > 0:
            self.n9["funnel"] += 1
            self.n9["targets"] += len(z[5])
            self.funnel_pending += z[5]
            print(f"[N9] t={self.t} funnel volcano at {z[1]}: cuts escapes of {len(z[5])} doomed dino(s), value {z[4]:.2f}")
        return z[1]

    def watch(self, s):
        T = s["currentTick"]
        alive = {d["id"] for d in s["dinosaurs"]}
        keep = []
        for L, i in self.funnel_pending:
            if T > L:
                self.n9["died"] += i not in alive
            else:
                keep.append((L, i))
        self.funnel_pending = keep
        super().watch(s)

    def finish(self, s):
        super().finish(s)
        n = self.n9
        print(f"[N9] summary | volcanoes {n['volc']} | funnel volcanoes {n['funnel']} | funnel targets {n['targets']} "
              f"| died after the funnel {n['died']} ({100 * n['died'] / max(n['targets'], 1):.0f}%, "
              f"trap table without lava ~ 20-45%)")
