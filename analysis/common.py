"""
Shared helpers for the analysis scripts (waves, spawns, movement, species, meteors, score, candidates).

Every script:
  - reads local game logs (local_game_logs/game_*.jsonl: one raw game state + our action per tick)
  - answers its block of questions from results/research_questions.md, in order
  - prints each answer as  Q<n>. <question> -> <answer>  (+ detail lines)
  - saves the same text to results/analysis/<local|server>/<script>.txt  ("server" if the logs path says server)
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

try:
    import orjson
    _loads = orjson.loads
except Exception:                          # pip install orjson -> ~3x faster loading
    _loads = json.loads

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SP = {"Stegosaurus": "S", "Velociraptor": "V", "Triceratops": "T", "Tyrannosaurus": "R"}
SP_NAME = {v: k for k, v in SP.items()}
EAT_DIST = 2


# ---------------------------------------------------------------- output
class Report:
    def __init__(self, name):
        self.name, self.lines = name, []

    def line(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)

    def q(self, n, question, answer):
        self.line(f"\nQ{n}. {question}\n    -> {answer}")

    def d(self, text):                      # detail line under a question
        self.line(f"       {text}")

    def save(self, header=""):
        tag = os.environ.get("ANALYSIS_TAG") or log_tag(LOGS)
        path = os.path.join(ROOT, "results", "analysis", tag, f"{self.name}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(header + "\n".join(self.lines) + "\n")
        print(f"\nsaved -> {os.path.relpath(path, ROOT)}")


LOGS = ""


def log_tag(logs):
    """'server' for server logs, otherwise 'local' (decides the output folder / report name)"""
    return "server" if "server" in os.path.basename(os.path.normpath(logs or "")).lower() or "server" in (logs or "").lower() else "local"


def args_and_files(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--games", type=int, default=0, help="only the newest N games (0 = all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    if extra:
        extra(ap)
    a = ap.parse_args()
    global LOGS
    LOGS = a.logs
    files = sorted(glob.glob(os.path.join(a.logs, "game_*.jsonl")))
    if a.games:
        files = files[-a.games:]
    if not files:
        sys.exit(f"no game_*.jsonl in {a.logs}")
    return a, files


def run_games(fn, files, workers):
    with Pool(workers) as p:
        return [g for g in p.imap(fn, files, chunksize=2) if g]


# ---------------------------------------------------------------- game loading
def load(path, min_ticks=100):
    """list of (state, actions) sorted by tick; None if unreadable / too short"""
    try:
        recs = [_loads(l) for l in open(path, "rb") if l.strip()]
    except Exception:
        return None
    recs = [(r["state"], r.get("actions") or []) for r in recs]
    recs.sort(key=lambda r: r[0]["currentTick"])
    return recs if len(recs) >= min_ticks else None


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def val(age):
    return 160.0 / (1 + age / 30.0)


def pos(d):
    return d["position"]["x"], d["position"]["y"]


def origin(s):
    return s["map"]["origin"]["x"], s["map"]["origin"]["y"]


def screen(s, p):
    ox, oy = origin(s)
    return p[0] - ox, p[1] - oy


def tile(s, p):
    """tile dict at world position p, or None outside the window"""
    ox, oy = origin(s)
    i, j = p[1] - oy, p[0] - ox
    t = s["map"]["tiles"]
    return t[i][j] if 0 <= i < len(t) and 0 <= j < len(t[0]) else None


def blocked(s, p):
    t = tile(s, p)
    return t is None or t["isImpassable"] or t["isMountain"] or t["hasLava"]


def danger_tiles(s):
    return {(q["x"], q["y"]) for m in s["meteors"] for q in m["impactedTiles"]}


def death_cause(d, prev, now):
    """why dino d (seen in prev) is missing from now: meteor / lava / scroll / eaten / other"""
    p = pos(d)
    for me in prev["meteors"]:
        if me["turnsUntilImpact"] == 1 and any((q["x"], q["y"]) == p for q in me["impactedTiles"]):
            return "meteor"
    for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        t = tile(now, (p[0] + a, p[1] + b))
        if t is not None and t["hasLava"]:
            return "lava"
    if p[1] < origin(now)[1]:
        return "scroll"
    if d["name"] != "Tyrannosaurus" and any(
            manh(p, pos(t)) <= EAT_DIST for t in prev["dinosaurs"] if t["name"] == "Tyrannosaurus" and t["id"] != d["id"]):
        return "eaten"
    return "other"


# ---------------------------------------------------------------- per-game events (used by several scripts)
CYCLE = ["10S", "8S 2V", "6S 4V", "2V 8T", "6S 4V", "2V 7T 1R", "6S 4V", "2V 7T 1R", "2S 2V 3T 3R", "3S 4T 3R"]


def recipe(c):
    return " ".join(f"{c[s]}{s}" for s in "SVTR" if c[s])


def events(recs):
    """
    born[id]  = dict(t, sp, p (world), sp_screen, age0)
    died[id]  = dict(t, cause, age, p)
    waves     = list of dict(t, ids, c (species Counter), trigger 'start'|'clear'|'timer'|'other', alive_before)
    alive[t]  = number of live dinos at tick t
    """
    born, died, waves, alive = {}, {}, [], {}
    prev, last_spawn, empty_since = None, -99, None
    for s, _ in recs:
        t = s["currentTick"]
        ids = {d["id"] for d in s["dinosaurs"]}
        alive[t] = len(ids)
        if prev is not None and t == prev["currentTick"] + 1:
            for d in prev["dinosaurs"]:
                if d["id"] not in ids and d["id"] not in died:
                    died[d["id"]] = dict(t=t, cause=death_cause(d, prev, s), age=d["age"], p=pos(d),
                                         sp=SP.get(d["name"], "?"))
        new = [d for d in s["dinosaurs"] if d["id"] not in born]
        for d in new:
            born[d["id"]] = dict(t=t, sp=SP.get(d["name"], "?"), p=pos(d), scr=screen(s, pos(d)), age0=d["age"])
        if new:
            if t - last_spawn > 2:
                old = len(ids) - len(new)
                if not waves:
                    trig = "start"
                elif empty_since is not None:
                    trig = "clear"
                elif t - waves[-1]["t"] == 100:
                    trig = "timer"
                else:
                    trig = "other"
                waves.append(dict(t=t, ids=[], c=Counter(), trigger=trig, alive_before=old,
                                  gap_after_empty=(t - empty_since) if empty_since is not None else None))
            last_spawn = t
            for d in new:
                waves[-1]["ids"].append(d["id"])
                waves[-1]["c"][SP.get(d["name"], "?")] += 1
        empty_since = (empty_since if empty_since is not None else t) if not ids else None
        prev = s
    for w in waves:
        w["recipe"] = recipe(w["c"])
        w["n"] = len(w["ids"])
        done = all(i in died for i in w["ids"])
        w["end"] = max(died[i]["t"] for i in w["ids"]) - w["t"] if done else None
        if done:
            last = max(w["ids"], key=lambda i: died[i]["t"])
            w["last_sp"] = born[last]["sp"]
    return born, died, waves, alive


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def merge(dicts):
    """sum Counters / extend lists / add numbers across games, key by key"""
    out = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, Counter):
                out.setdefault(k, Counter()).update(v)
            elif isinstance(v, (defaultdict, dict)):
                o = out.setdefault(k, defaultdict(Counter) if isinstance(v, defaultdict) else {})
                for kk, vv in v.items():
                    if isinstance(vv, Counter):
                        o.setdefault(kk, Counter()).update(vv)
                    elif isinstance(vv, list):
                        o.setdefault(kk, []).extend(vv)
                    else:
                        o[kk] = o.get(kk, 0) + vv
            elif isinstance(v, list):
                out.setdefault(k, []).extend(v)
            else:
                out[k] = out.get(k, 0) + v
    return out
