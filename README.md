# Coveo Blitz 2027 Bot

Bot for the Coveo Blitz 2027 registration challenge: drop meteors and trigger volcanoes to
eliminate dinosaurs. Best score: **22,771** (team Costanza, bot `n4_survival_follow`).

## Results

Each score is one server game. Maps change every game, so one game can swing by 5k+: compare averages.

| Ver | Bot | Idea | Server scores | Avg |
|---|---|---|---|---|
| - | **`n4_survival_follow`** | **Best.** Cage B + timing fix + "will it survive?" check + follow-up shots where fleeing dinos stop | 16,335 / 22,771 / 19,361 | **19,489** |
| - | `n10_wave_modes` | N8 + candidate strategy per wave group (runner / pack / T-Rex) | pending | - |
| - | `n9_lava_funnel` | N8 + volcano chosen to block escape routes | pending | - |
| - | `n8_future_spots` | N7 + spots where dinos WILL be (one no-meteor CNN4 forecast) | pending | - |
| - | `n7_safe_timing` | N6 + timing recovery on slow servers (cooldown, 4-spot probe, ramp up) | 17,902 / 19,218 / 15,766 | 17,629 |
| - | `n6_follow_budget` | N4 + follow-up spots capped to ~25% of the CNN4 budget | 19,701 / 14,702 | 17,202 |
| - | `n5_hold_trapped` | N4 + hold when a doomed dino has only 1-3 escape tiles | 14,267 / 14,626 | 14,447 |
| - | N4 before timing fix | same as N4, but CNN4 got stuck (229 blind shots) | 10,222 | 10,222 |
| v32 | `cage_cnn4_report` | Cage B + per-tick trace + end-of-game report | 14,741 | 14,741 |
| v31 | `n1_straggler_hunter` | Cage B + clear bonus 10 + stragglers at full value + aim ahead of runners | 9,710 / 10,345 | 10,028 |
| v30 | N0 | Cage B + new CNN4 (1,151 games, 6 blocks) | 14,416 / 16,843 / 11,312 | 14,190 |
| v27 | Z1L | Z1 + end-of-game report | 10,390 / 15,624 | 13,007 |
| v26 | ZW | Z1 + Z4 timing guards | 13,689 / 8,455 | 11,072 |
| v25 | ZH2.1 | ZH2 with timing fix (CNN4 ran every tick) | 10,131 | 10,131 |
| v24 | ZH2 | ZH1 + CNN4 second opinion (CNN barely ran) | 15,318 / 13,010 | 14,164 |
| v22 | ZH1 (as ZH2) | ZH2 zip with old ZH1 file: CNN never ran | 8,698 / 7,402 | 8,050 |
| v21 | `species_heuristic` (ZH1) | pure heuristic, species flee simulation | 9,143 | 9,143 |
| v20 | Z4 | Z3 + safer timing, no next-wave reserve | 10,381 / 14,007 | 12,194 |
| v19 | Z3 | Z1 + 5 fixes (smart hold, scroll rules, ...) | 17,904 / 10,564 / 11,378 | 13,282 |
| v18 | Z2 / mix-up | Z1 without hold rule (zip mix-up, unclear) | 17,739 / 13,998 / 13,338 | 15,025 |
| v17 | `wide_search_cnn4` (Z1) | Cage B + ~150 spots (empty tiles near dinos too) | 15,437 / 12,250 | 13,844 |
| v16 | `cage_cnn4` (Cage B) | meant as Z1, but the zip held Cage B; previous best | 16,228 / 18,888 / 16,986 | 17,367 |
| v15 | `cage_cnn4` (Cage B) | plain models again (repeat test) | 14,999 / 15,774 / 8,803 | 13,192 |
| v14 | Cage B mix | plain CNN1 + CNN4 prev7 | 10,765 | 10,765 |
| v13 | Cage B mix | CNN1 prev7 + plain CNN4 | 14,476 | 14,476 |
| v12 | Cage B prev7 | both CNNs replaced by prev7 versions | 15,617 / 15,847 | 15,732 |
| v11 | Cage C | Cage B + volcano whenever idle | 11,363 | 11,363 |
| v10 | `cage_cnn4` (Cage B) | Cage A, but CNN4 scores the meteor spots | 16,293 | 16,293 |
| v9 | `cage_planner` (Cage A) | trap dinos against walls (trap table) | 13,904 | 13,904 |
| v8 | Heuristic | fast-clear reward, no CNN | 6,716 | 6,716 |
| v6-7 | `fast_clear` | reward: young dinos + multi-kill + fast clear | 8,477 / 8,470 | 8,474 |
| v2-5 | early | first bots, heuristic + ML experiments | 6,741 / 293 / 6,199 / 7,634 | - |

CNN1 = predicts each dino's next move (1 tick ahead). CNN4 = predicts where dinos are when a meteor lands
(4 ticks ahead). prev7 = model variant that looks at the last 7 moves.

### What we learned

- Trapping dinos against walls (Cage A) was the first big jump: ~8k -> ~14k.
- Letting CNN4 score meteor spots (Cage B) was the next: -> ~16k.
- **Timing bug (fixed):** one slow tick could push the CNN4 time estimate so high that CNN4 never ran again
  (every shot blind). Fixed in `cage_cnn4.py` with `UNIT_CAP = 3` ms. Some old low scores may be this bug.
- **Follow-up shots (N4):** Cage B held fire when every dino was under a falling meteor, but 71% of those dinos
  escaped. Firing a follow-up where they stop: ~15k -> ~19.5k.
- Holding fire for dinos with 1-3 escape tiles (N5) scored worse (~14.4k): when the last dino escapes, the whole
  wave waits, so a cheap wasted shot beats a missed straggler.
- Capping follow-up spots (N6) and safe timing (N7) changed little in score (~17-18k, within map luck); N7 cut
  missed ticks from 2-6 to 1-2 per game.
- Did not help: aiming at empty tiles (Z1), prev7 models, hand-written behaviour rules (ZH1), the new CNN4 (N0),
  big straggler bonuses (N1).
- Spawns are random; CNN4 is well calibrated; T-Rex meals cost only ~4%.

### Game facts (1,054 local games, `results/research_answers.md`)

- Next wave: 1 tick after the board is empty, otherwise 100 ticks after the previous wave started (timer resets).
  Max 20 dinos alive; a wave only fills the free slots.
- Species per wave follow a fixed 10-wave cycle: 10S | 8S 2V | 6S 4V | 2V 8T | 6S 4V | 2V 7T 1R | 6S 4V | 2V 7T 1R |
  2S 2V 3T 3R | 3S 4T 3R. Spawn **positions** are random (tile, grid, species, wave, terrain: all tested).
- Catch rate by free escape tiles: 0 -> 97%, 1-3 -> 60%, 4-7 -> 35%, 8-14 -> 13%, 15+ -> 3%.
  Open ground 3% vs ~27-30% next to walls, mountains, lava or the window edge.
- Escaped dinos end 3-5 tiles from the blast centre; a second blast 3 tiles out along the escape line covers 69%.
- A dino next to a T-Rex is eaten the next tick 65% of the time (always from 1 tile).
- Stegosaurus almost never stands still (14%); raptors and T-Rex stand still ~50%.
- Scoring: 160 / (1 + age/30) exactly; multi-kill x1.5 (2) and x1.93 (3).
- The site's "Strategy" claims (NE quadrant, counter-clockwise herds, double-bounce) and raptors-to-corpses
  are not supported by the data.

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
./scripts/ship.sh n4_survival_follow   # -> dist/n4_survival_follow.zip, ready to upload
./scripts/move_zips_to_dist.sh         # tidy: move any stray *.zip from the repo root into dist/
```

`ship.sh` copies `bot/` to a temporary folder, makes `bot.py` load the bot you name, turns game logging off,
checks that it imports, and zips it. Your `bot/` folder is never changed.

To pick the bot that runs locally, edit the one import line in `bot/bot.py`:

```python
from n4_survival_follow import Bot
```

---

## 2. Bots (`bot/candidate_bots/`)

Scores are in **Results** above.

| Bot | Idea |
|---|---|
| `n4_survival_follow` | **Best.** `cage_cnn4` + when every dino is under a falling meteor, fire a follow-up if it will likely survive (escape-tile count) + follow-up spots where fleeing dinos stop. Full trace + `[N4]` logs. |
| `n5_hold_trapped` | N4, but holds when a doomed dino has only 1-3 escape tiles (cutoff 0.5 -> 0.6). Worse. |
| `n6_follow_budget` | N4 + follow-up spots sorted by trap score, only max(2, budget/4) of them go first. |
| `n7_safe_timing` | N6 + timing recovery: after a slow tick, cooldown 3 ticks, probe 4 spots, ramp 8 -> 16 -> normal. |
| `n8_future_spots` | N7 + one no-meteor CNN4 forecast; the 6 best new "where dinos will be" spots get guaranteed slots. |
| `n9_lava_funnel` | N8 + volcano chosen to cut escape routes of dinos under a falling meteor. |
| `n10_wave_modes` | N8 + per wave group: runner (10 future spots), pack (Triceratops group centres), T-Rex (spots on the T-Rex and its prey). `[N10]` logs ticks per wave type. |
| `cage_cnn4` | Base of all bots: spots around dinos and their escape tiles, ranked by a measured trap table, scored by CNN4. Includes the timing fix (`UNIT_CAP`). |
| `cage_cnn4_report` | Same decisions as `cage_cnn4`, prints a per-tick trace and an end-of-game report for analysis. |
| `wide_search_cnn4` | CNN4 also scores empty tiles near dinos (~150 spots). |
| `n1_straggler_hunter` | `cage_cnn4` + bigger clear bonus, stragglers at full value, aims ahead of running dinos. |
| `n2_straggler_priority` | N1 + stragglers worth 1000x. Untested. |
| `species_heuristic` | No ML: hand-written species flee simulation. |

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
mv bot/logs/game_*.jsonl logs/local_game_logs/

python training/build_cnn4_dataset.py --logs logs/local_game_logs --out data/cnn4
python training/tune_cnn4.py --data data/cnn4                     # optional: 4-fold CV hyper-parameter search
python training/train_cnn4.py --data data/cnn4 --out data/new_model/model.pt
python training/eval_cnn4.py --model data/new_model/model.pt --split data/cnn4

cp data/new_model/model.pt bot/models/cnn_lookahead_4/model.pt   # if it is better
```

---

## 4. Analyse the game

```bash
python analysis/run_all_analyses.py --logs logs/local_game_logs [--games 300]
# server games: convert the downloaded team logs first
python analysis/team_log_to_jsonl.py logs/server_game_logs/*.txt --out logs/server_game_logs
python analysis/run_all_analyses.py --logs logs/server_game_logs
```

Answers the 45 game questions in `results/research_questions.md` (rules, spawns, dino behaviour, physics):
`results/analysis_report_<local|server>.md` (everything in one file) and `results/analysis/<local|server>/*.txt`.
Summary of the answers: `results/research_answers.md`. Bot-performance questions (Q46-Q74) are listed but not built yet.

| Script | Questions |
|---|---|
| `waves.py` | Q1-Q7 wave rules: timing, timer reset, 20 cap, species recipe, 10-wave cycle |
| `spawns.py` | Q8-Q15 where dinos spawn |
| `movement.py` | Q16-Q25 how dinos move and react, predictability, window edge, scrolling |
| `species.py` | Q26-Q36 species behaviour and the site's strategy claims |
| `physics.py` | Q37-Q45 catch rates, escapes, follow-ups, lava, scoring formula |
| `common.py` | shared loading and helpers |
| `team_log_to_jsonl.py` | server team logs -> local log format |

---

## Repo

| Folder | Content |
|---|---|
| `bot/` | Uploaded code: `bot.py` (picks the bot), starter kit, `candidate_bots/`, `infrastructure/`, `models/` |
| `training/` | Build datasets, tune, train and evaluate the CNNs |
| `analysis/` | Game analysis scripts (Q1-Q45) |
| `results/` | Research questions and answers, analysis reports |
| `scripts/` | `ship.sh` (build a zip), `move_zips_to_dist.sh`, `benchmark.sh` |
| `dist/` | Built zips (git-ignored) |
| `logs/` | `local_game_logs/`, `server_game_logs/` (git-ignored) |
| `data/` | Datasets for training (git-ignored) |
