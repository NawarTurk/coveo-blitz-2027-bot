#!/usr/bin/env python3
"""
Run every analysis on the local game logs and write ONE Markdown report, ready to paste into an LLM
to reason about the next bot.

Usage:   python analysis/run_all_analyses.py --logs local_game_logs
         python analysis/run_all_analyses.py --logs local_game_logs/n1_straggler_hunter --out results/report_n1.md
Writes:  results/analysis_report.md       one Markdown report with everything (feed this to an LLM)
         results/analysis/<name>.txt       each analysis's raw output on its own
         (+ the CSV / PNG files some analyses write in results/)
"""
import argparse, glob, os, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (title, script, extra args, what the analysis answers)
ANALYSES = [
    ("Wave clear timing", "wave_clear_timing.py", [],
     "How many ticks each wave takes to die (first kill, 50%, 80%, all), how the last dino of a wave dies, "
     "per-species lifetimes, and how wave speed relates to score."),
    ("Flee behaviour", "flee_behavior.py", [],
     "How dinos react to a falling meteor: hit rate, first move (away / sideways / toward / stay), "
     "escape paths, catch rate vs free escape tiles (walls), chain shots."),
    ("T-Rex predation", "trex_predation.py", ["POSITIONAL"],
     "How dinos die (meteor / lava / scroll / eaten / other), how many are eaten by a T-Rex and the points lost."),
    ("Spawn sequence (exact tiles)", "spawn_sequence.py", [],
     "Inside one game, can the next wave's spawn tiles be predicted from earlier waves "
     "(same spots, rotation, mirror, shift, screen heatmap)? Scored by % of newborns 3 pre-aimed blasts would catch."),
    ("Spawn sequence per species", "spawn_sequence_per_species.py", [],
     "Same prediction test, one species at a time."),
    ("Spawn blocks (3x3 screen blocks)", "spawn_blocks.py", [],
     "Patterns in WHICH of 9 screen blocks waves spawn: sequence from wave to wave (information in bits vs shuffled), "
     "memory (blocks repeat or cycle), spread inside a wave."),
    ("Spawn blocks per species", "spawn_blocks_per_species.py", [],
     "Same block tests, one species at a time."),
]

CONTEXT = """\
## Game summary (Coveo Blitz 2027)

- 20x20 visible window that scrolls one row every 5 ticks; 1,000 ticks per game; ~150 ms per tick to answer.
- One action per tick: launch a meteor (lands 4 ticks later, 13-tile diamond blast, max 3 in the air)
  or trigger a volcano (lava spreads from a mountain).
- Dinos (Stegosaurus, Velociraptor, Triceratops, T-Rex) move 1 tile per tick and flee meteors.
- Points per kill = 160 / (1 + age/30) x multi-kill bonus (1 + 0.5 per extra kill on the same impact).
  Scroll deaths count; dinos eaten by a T-Rex give 0.
- Waves of ~10 dinos every 100 ticks, OR 2 ticks after the board is empty.
- Leaderboard: top 2 teams ~160k, ranks 3-6 ~36-47k, us ~15-19k.

## Our bots (bot/candidate_bots/)

- cage_cnn4 (best, ~15.4k avg): candidate spots around dinos and their escape tiles, ranked by a measured
  "trap table" (catch rate vs free escape tiles), scored by CNN4 (predicts every dino's position at impact),
  final score = 70% CNN4 + 30% trap table.
- wide_search_cnn4 (~13.4k): same but scores ~150 spots incl. empty tiles near dinos.
- species_heuristic (~8.4k): no ML, hand-written species flee simulation.
- n1_straggler_hunter (~10k, 2 games): cage_cnn4 + bigger clear bonus + stragglers at full value + aim ahead of runners.
"""


def run(script, extra, logs):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        return None, f"(script not found: analysis/{script})", 0.0
    args = [logs] if extra == ["POSITIONAL"] else ["--logs", logs] + extra
    t0 = time.time()
    p = subprocess.run([sys.executable, path] + args, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([HERE, os.path.join(ROOT, "bot", "infrastructure")])})
    dt = time.time() - t0
    out = (p.stdout or "").rstrip()
    if p.returncode != 0:
        out += "\n\n[ERROR]\n" + "\n".join((p.stderr or "").strip().splitlines()[-15:])
    return p.returncode, out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    ap.add_argument("--out", default="results/analysis_report.md")
    ap.add_argument("--skip", nargs="*", default=[], help="script names to skip, e.g. flee_behavior.py")
    args = ap.parse_args()
    logs = args.logs
    n_games = len(glob.glob(os.path.join(ROOT, logs, "game_*.jsonl")))
    md = [f"# Analysis report\n",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{logs}` ({n_games} local games).\n",
          "Purpose: give an LLM the full picture to reason about the next bot improvement.\n",
          CONTEXT]
    res = os.path.join(ROOT, "results", "bot_results.txt")
    if os.path.exists(res):
        md += ["## Server results so far\n", "```", open(res).read().rstrip(), "```\n"]
    md.append("## Analyses\n")
    txt_dir = os.path.join(ROOT, os.path.dirname(args.out) or "results", "analysis")
    os.makedirs(txt_dir, exist_ok=True)
    for title, script, extra, what in ANALYSES:
        if script in args.skip:
            continue
        print(f"running {script} ...", flush=True)
        code, out, dt = run(script, extra, logs)
        status = "" if code == 0 else " (FAILED)" if code is not None else " (MISSING)"
        md += [f"### {title}{status}\n", f"*Question:* {what}\n", f"*Script:* `analysis/{script}` ({dt:.0f} s)\n",
               "```", out or "(no output)", "```\n"]
        with open(os.path.join(txt_dir, os.path.splitext(script)[0] + ".txt"), "w") as f:
            f.write(f"{title}\n{what}\nlogs: {logs} ({n_games} games)\n\n{out}\n")
        print(f"   done in {dt:.0f} s{status}")
    md += ["## Questions for the next step\n",
           "1. Where does the time per wave go, and which change would clear waves fastest?",
           "2. Is anything predictable that the bot does not use yet (spawns, flee direction, species)?",
           "3. Which single change to cage_cnn4 should be tested next, and how to measure it fairly "
           "(same seeds, 100 local games)?\n"]
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(md))
    print(f"\nwrote {args.out} and {os.path.relpath(txt_dir, ROOT)}/*.txt")


if __name__ == "__main__":
    main()
