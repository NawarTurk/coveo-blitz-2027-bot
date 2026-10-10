#!/usr/bin/env python3
"""
Run every GAME analysis (waves, spawns, movement, species, physics = Q1-Q45 of results/research_questions.md)
on the local game logs and write ONE Markdown report, ready to paste into an LLM
to reason about the next bot.

Usage:   python analysis/run_all_analyses.py --logs logs/local_game_logs [--games 300]     -> ..._local
         python analysis/run_all_analyses.py --logs logs/server_game_logs                  -> ..._server
         (server team logs must first be converted: python analysis/team_log_to_jsonl.py <txt files> --out logs/server_game_logs)
Writes:  results/analysis_report_<local|server>.md   one Markdown report with every answer
         results/analysis/<local|server>/<name>.txt   each script's output on its own
         (+ the CSV / PNG files some analyses write in results/)
"""
import argparse, glob, os, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (title, script, extra args, what the analysis answers)
ANALYSES = [
    ("Waves (Q1-Q7)", "waves.py", [], "Wave rules: when waves come, timer reset, 20 cap, fixed species recipe, 10-wave cycle."),
    ("Spawns (Q8-Q15)", "spawns.py", [], "Where dinos spawn: exact tiles, grids, species, wave number, repeats, clustering, terrain."),
    ("Movement (Q16-Q25)", "movement.py", [], "How dinos move calm and threatened, flee patterns, predictability 1-4 ticks ahead, edges, scrolling."),
    ("Species (Q26-Q36)", "species.py", [], "Species behaviour (Triceratops high ground, raptors and corpses, T-Rex eating, lava) and the site's strategy claims."),
    ("Physics (Q37-Q45)", "physics.py", [], "Meteor catch rates vs escape tiles and obstacles, escapes, follow-up gaps, lava funnels, scoring formula."),
]

CONTEXT = """\
## Game summary (Coveo Blitz 2027)

- 20x20 visible window that scrolls one row every 5 ticks; 1,000 ticks per game; ~150 ms per tick to answer.
- One action per tick: launch a meteor (lands 4 ticks later, 13-tile diamond blast, max 3 in the air)
  or trigger a volcano (lava spreads from a mountain).
- Dinos (Stegosaurus, Velociraptor, Triceratops, T-Rex) move 1 tile per tick and flee meteors.
- Points per kill = 160 / (1 + age/30) x multi-kill bonus (1 + 0.5 per extra kill on the same impact).
  Scroll deaths count; dinos eaten by a T-Rex give 0.
- Waves of 10 dinos 100 ticks after the previous wave started, OR 1 tick after the board is empty; max 20 alive.
- Species per wave follow a fixed 10-wave cycle: 10S / 8S2V / 6S4V / 2V8T / 6S4V / 2V7T1R / 6S4V / 2V7T1R / 2S2V3T3R / 3S4T3R.
"""


TAG = "local"


def run(script, extra, logs, games=0):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        return None, f"(script not found: analysis/{script})", 0.0
    args = [logs] if extra == ["POSITIONAL"] else ["--logs", logs] + extra + (["--games", str(games)] if games else [])
    t0 = time.time()
    p = subprocess.run([sys.executable, path] + args, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "ANALYSIS_TAG": TAG, "PYTHONPATH": os.pathsep.join([HERE, os.path.join(ROOT, "bot", "infrastructure")])})
    dt = time.time() - t0
    out = (p.stdout or "").rstrip()
    if p.returncode != 0:
        out += "\n\n[ERROR]\n" + "\n".join((p.stderr or "").strip().splitlines()[-15:])
    return p.returncode, out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--out", default="", help="default: results/analysis_report_<local|server>.md")
    ap.add_argument("--tag", default="", help="local / server (default: 'server' if the logs path says server)")
    ap.add_argument("--skip", nargs="*", default=[], help="script names to skip, e.g. species.py")
    ap.add_argument("--games", type=int, default=0, help="only the newest N games (0 = all)")
    args = ap.parse_args()
    logs = args.logs
    global TAG
    TAG = args.tag or ("server" if "server" in logs.lower() else "local")
    args.out = args.out or f"results/analysis_report_{TAG}.md"
    n_games = len(glob.glob(os.path.join(ROOT, logs, "game_*.jsonl")))
    md = [f"# Game analysis report\n",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{logs}` ({n_games} {TAG} games"
          + (f", newest {args.games} used" if args.games else "") + ").\n",
          "Answers to the game questions Q1-Q45 (results/research_questions.md). Game facts only, no bot results.\n",
          CONTEXT]
    md.append("## Analyses\n")
    for title, script, extra, what in ANALYSES:
        if script in args.skip:
            continue
        print(f"running {script} ...", flush=True)
        code, out, dt = run(script, extra, logs, args.games)
        status = "" if code == 0 else " (FAILED)" if code is not None else " (MISSING)"
        md += [f"### {title}{status}\n", f"*Question:* {what}\n", f"*Script:* `analysis/{script}` ({dt:.0f} s)\n",
               "```", out or "(no output)", "```\n"]
        print(f"   done in {dt:.0f} s{status}")
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(md))
    print(f"\nwrote {args.out} and results/analysis/{TAG}/*.txt")


if __name__ == "__main__":
    main()
