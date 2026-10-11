#!/usr/bin/env python3
"""
STEP 1-4 of the XGBoost shot scorer (v2: + wave inputs): one row per meteor we actually fired, with what CNN4 predicted
at that moment + trap score + species + positions + mode (scorer_features.py), and what really happened.

  label kills = dinos inside the blast on the landing tick that are gone the tick after
  y1 = kills >= 1,  y2 = kills >= 2,  pts = 160/(1+age/30) summed over the killed dinos

Usage (from the repo root):
  python training/build_scorer_dataset.py --logs logs/local_game_logs --out data/scorer_local.csv.gz
  python training/build_scorer_dataset.py --logs logs/local_game_logs --max-games 20 --out data/scorer_test.csv.gz   # quick try
Server games: convert team logs first, then build a separate file for validation:
  python analysis/team_log_to_jsonl.py logs/server_game_logs/*.txt --out logs/server_jsonl
  python training/build_scorer_dataset.py --logs logs/server_jsonl --out data/scorer_server.csv.gz
Exploration games from play_local.sh are saved as .jsonl in logs/local_<tag>/jsonl -> use that folder as --logs.
--tag NAME prefixes game ids (e.g. srv / exp) so games from different files never share an id.
Runs games in parallel (--workers, default = CPU cores - 1).
"""
import argparse, csv, glob, gzip, json, os, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot")
sys.path[:0] = [BOT, os.path.join(BOT, "candidate_bots"), os.path.join(BOT, "infrastructure")]
os.environ.setdefault("BOT_DEVICE", "cpu")

EX = None            # per-worker extractor
TAG = ""


RULE = None          # behaviour model (rule-model inputs); None -> those columns are empty


def init_worker(tag="", rule_path=""):
    global EX, TAG, RULE
    TAG = tag
    if rule_path and os.path.exists(rule_path):
        import behavior_model as BMOD
        RULE = BMOD.BehaviorModel.load(rule_path)
    import builtins, contextlib, io
    import torch
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):         # the bot prints on load
        import n4_survival_follow as N4
        import n12_modes as N12
        N4.RP.TRACE = False
        N4.C.LOG_GAMES = False
        bot = N4.Bot()
    EX = (N4, N12, bot)


def one_game(path):
    import numpy as np
    import scorer_features as SF
    N4, N12, proto = EX
    bot = proto
    bot.hist4, bot.prev_pos = {}, {}
    waves = SF.WaveTracker()
    moves = SF.MoveTracker()
    states = []
    for l in open(path, "rb"):
        if l.strip():
            try:
                r = json.loads(l)
                states.append((r["state"], r.get("actions") or []))
            except Exception:
                pass
    states.sort(key=lambda x: x[0]["currentTick"])
    if len(states) < 100:
        return []
    by_t = {s["currentTick"]: s for s, _ in states}
    game = TAG + os.path.splitext(os.path.basename(path))[0]
    rows = []
    for s, acts in states:
        bot.update_hist(s)
        waves.update(s)
        moves.update(s)
        shot = None
        for a in acts:
            if a.get("type") == "LAUNCH_METEOR":
                shot = (a["target"]["x"], a["target"]["y"])
        if shot is not None and s["dinosaurs"]:
            try:
                rows.append(row(bot, s, shot, by_t, game, SF, N12, np, waves, moves))
            except Exception as e:
                print(f"{game} t={s['currentTick']}: {e!r}", file=sys.stderr)
        bot.prev_pos = {d["id"]: (d["position"]["x"], d["position"]["y"]) for d in s["dinosaurs"]}
    return rows


def row(bot, s, c, by_t, game, SF, N12, np, waves, moves):
    bot.t = s["currentTick"]
    bot.setup(s)                                       # geometry: free tiles, falling meteors, dinos + reach
    bot.prior = {c: bot.cage_prior(c)}
    mode, flags = N12.board_mode(s)
    names = {d["id"]: d["name"] for d in s["dinosaurs"]}
    follow = set(bot.follow_spots())
    X0, sc = bot.base_input(s)
    X = X0.copy()
    mask = np.zeros((bot.H, bot.W), bool)
    for dx, dy in bot.foot:
        i, j = c[1] + dy - bot.oy, c[0] + dx - bot.ox
        if 0 <= i < bot.h and 0 <= j < bot.w:
            X[bot.i_new, i, j] = 1
            mask[i, j] = True
    P = bot.predict4([X], sc)[0]
    f = SF.features(bot, c, P, bot.value_grid(), mask, names, mode, flags, follow, bot_clear_bonus()) + waves.features(s)
    fc = SF.rule_forecast(RULE, s, moves.last) if RULE is not None else None
    blast = {(c[0] + dx, c[1] + dy) for dx, dy in bot.foot}
    f = f + SF.rule_features(fc, blast)
    # label: what really happened on the landing tick
    L = s["currentTick"] + s["constants"]["meteorDelay"]
    pre, post = by_t.get(L), by_t.get(L + 1)
    kills, pts = 0, 0.0
    if pre and post:
        R = s["constants"]["meteorRadius"]
        gone = {d["id"] for d in pre["dinosaurs"]} - {d["id"] for d in post["dinosaurs"]}
        for d in pre["dinosaurs"]:
            if d["id"] in gone and abs(d["position"]["x"] - c[0]) + abs(d["position"]["y"] - c[1]) <= R:
                kills += 1
                pts += 160 / (1 + (d["age"] + 1) / 30)
    else:
        kills = -1                                        # landing after the game ended: dropped later
    return [game, s["currentTick"], c[0], c[1], *[round(x, 5) if isinstance(x, float) and x == x else x for x in f],
            kills, int(kills >= 1), int(kills >= 2), round(pts, 3)]


def bot_clear_bonus():
    import fast_clear as FC
    return FC.CLEAR_BONUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/local_game_logs")
    ap.add_argument("--out", default="data/scorer_local.csv.gz")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--rule-model", default=os.path.join(BOT, "models", "behavior", "weights.json"),
                    help="behaviour model weights (fit_behavior.py); missing -> rule columns empty")
    ap.add_argument("--tag", default="", help="prefix for game ids, e.g. srv_ or exp_")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    import scorer_features as SF
    files = sorted(glob.glob(os.path.join(a.logs, "*.jsonl")))
    if a.max_games:
        files = files[:a.max_games]
    if not files:
        sys.exit(f"no .jsonl games in {a.logs}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    head = ["game", "tick", "x", "y", *SF.ALL_FEATURES, "kills", "y1", "y2", "pts"]
    t0, n, done = time.time(), 0, 0
    print(f"{len(files)} games, {a.workers} workers -> {a.out} | rule model: "
          f"{a.rule_model if os.path.exists(a.rule_model) else 'NOT FOUND (rule columns empty)'}")
    with gzip.open(a.out, "wt", newline="") as fh, Pool(a.workers, initializer=init_worker, initargs=(a.tag, a.rule_model)) as pool:
        w = csv.writer(fh)
        w.writerow(head)
        for rows in pool.imap_unordered(one_game, files):
            rows = [r for r in rows if r[-4] >= 0]
            w.writerows(rows)
            n += len(rows)
            done += 1
            if done % 20 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done}/{len(files)} games | {n} shots | {el / 60:.1f} min | "
                      f"~{el / done * (len(files) - done) / 60:.0f} min left", flush=True)
    print(f"saved {n} shots from {done} games to {a.out}")


if __name__ == "__main__":
    main()
