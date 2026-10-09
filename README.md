# Coveo Blitz 2027 Bot

Bot for the Coveo Blitz 2027 registration challenge: drop meteors and trigger volcanoes to
eliminate dinosaurs. Best score: **18,888** (team Costanza).

The main bot traps dinosaurs against walls and uses a CNN that predicts where every dinosaur will
be when a meteor lands (4 ticks ahead) to pick the best meteor spot.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt matplotlib pandas scikit-learn
```

---

## 1. Ship a bot

```bash
./scripts/ship.sh cage_cnn4          # -> cage_cnn4.zip, ready to upload
```

`ship.sh` copies `bot/` to a temporary folder, makes `bot.py` load the bot you name, turns game logging off,
checks that it imports, and zips it. Your `bot/` folder is never changed.

To pick the bot that runs locally, edit the one import line in `bot/bot.py`:

```python
from cage_cnn4 import Bot
```

---

## 2. Bots (`bot/candidate_bots/`)

| Bot | Idea | Server avg |
|---|---|---|
| `cage_cnn4` | **Best.** Spots around dinos and their escape tiles, ranked by a measured trap table, scored by CNN4. | ~15.4k (7 games) |
| `cage_cnn4_report` | Same decisions as `cage_cnn4`, prints a per-tick trace and an end-of-game report for analysis. | - |
| `wide_search_cnn4` | CNN4 also scores empty tiles near dinos (~150 spots). | ~13.4k |
| `n1_straggler_hunter` | `cage_cnn4` + bigger clear bonus, stragglers at full value, aims ahead of running dinos. | ~10k (2 games) |
| `n2_straggler_priority` | N1 + stragglers worth 1000x. Untested. | - |
| `species_heuristic` | No ML: hand-written species flee simulation. | ~8.4k |

All bots run on the shared engine in `bot/infrastructure/` (planners, scoring, CNN loading, map encoding).

**Models** (`bot/models/`):

| Folder | Predicts | Used for |
|---|---|---|
| `cnn_lookahead_4/model.pt` | What stands on every tile when a meteor lands (4 ticks ahead) | Meteors |
| `cnn_lookahead_1/model.pt` | Each dinosaur's next move (1 tick ahead) | Volcano |

---

## 3. Train the CNN

```bash
export PYTHONPATH=bot/infrastructure

# play games locally (local game server + LOG_GAMES = True), then:
mv bot/logs/game_*.jsonl local_game_logs/

python training/build_cnn4_dataset.py --logs local_game_logs --out data/cnn4
python training/tune_cnn4.py --data data/cnn4                     # optional: 4-fold CV hyper-parameter search
python training/train_cnn4.py --data data/cnn4 --out data/new_model/model.pt
python training/eval_cnn4.py --model data/new_model/model.pt --split data/cnn4

cp data/new_model/model.pt bot/models/cnn_lookahead_4/model.pt   # if it is better
```

---

## 4. Analyse games

```bash
python analysis/run_all_analyses.py --logs local_game_logs
```

Runs every analysis and writes `results/analysis_report.md` (one document, ready to give to an LLM)
plus each analysis on its own in `results/analysis/`.

| Script | Question |
|---|---|
| `wave_clear_timing.py` (+ `_per_species`) | How many ticks a wave takes to die, who the last survivor is |
| `flee_behavior.py` | How dinosaurs react to a falling meteor |
| `trex_predation.py` | How dinosaurs die; points lost to T-Rex meals |
| `spawn_sequence.py`, `spawn_blocks.py` (+ `_per_species`) | Can spawn positions be predicted? |
| `build_meteor_table.py`, `fit_meteor_models.py` | Do heuristics add information beyond CNN4? |

---

## Key findings

- **Score follows wave speed.** A new wave comes 2 ticks after the board is cleared; our best games clear a wave
  in ~55 ticks, our worst in ~240. The last 20% of a wave takes ~35% of its time.
- **Spawns are random**, per tile, per screen block and per species; pre-aiming does not beat blind aim.
- **CNN4 is well calibrated** (0.24 predicted vs 0.22 actual kills per meteor); heuristics on top add ~3% at best.
- **T-Rex meals cost only ~4%** of the score.

---

## Repo

| Folder | Content |
|---|---|
| `bot/` | Uploaded code: `bot.py` (picks the bot), starter kit, `candidate_bots/`, `infrastructure/`, `models/` |
| `training/` | Build datasets, tune, train and evaluate the CNNs |
| `analysis/` | Studies on game logs |
| `results/` | Bot scores, analysis outputs and the combined report |
| `scripts/` | `ship.sh` |
| `local_game_logs/`, `data/` | Local only (git-ignored): recorded games and datasets |
