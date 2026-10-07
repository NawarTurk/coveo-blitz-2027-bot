# Coveo Blitz 2027 Bot

Bot for the Coveo Blitz 2027 registration challenge: drop meteors and trigger volcanoes to
eliminate dinosaurs. Best score: **18,888**.

The main bot traps dinosaurs against walls and uses a CNN that predicts where every dinosaur will
be when a meteor lands (4 ticks ahead) to pick the best meteor spot.

---

## 1. Ship a bot

The server needs the **contents** of `bot/` in a zip.

```bash
cd bot
zip -r ../bot.zip . -x "*__pycache__*" "logs/*" "README.txt"
cd ..
```

Upload `bot.zip`.

**Which bot plays?** Edit one line in `bot/bot.py`:

```python
from cage_cnn4 import Bot           # best
# from wide_search_cnn4 import Bot
# from species_heuristic import Bot
```

---

## 2. Train the CNN

Run everything from the repo root.

```bash
export PYTHONPATH=bot/infrastructure

# 1. play games locally (local game server + LOG_GAMES = True), then move them:
mv bot/logs/game_*.jsonl local_game_logs/

# 2. build the dataset
python training/build_cnn4_dataset.py --logs local_game_logs --out data/cnn4

# 3. (optional) find the best settings with 4-fold cross-validation
python training/tune_cnn4.py --data data/cnn4

# 4. train
python training/train_cnn4.py --data data/cnn4 --out data/new_model/model.pt

# 5. test it on games it never saw
python training/eval_cnn4.py --model data/new_model/model.pt --split data/cnn4

# 6. if it is better, use it
cp data/new_model/model.pt bot/models/cnn_lookahead_4/model.pt
```

---

## 3. Repo

| Folder | What's inside |
|---|---|
| `bot/` | What gets uploaded. `bot.py` picks the bot; `application.py` and `game_message.py` are the starter kit. |
| `bot/candidate_bots/` | The bots: `cage_cnn4` (best), `wide_search_cnn4`, `species_heuristic`. |
| `bot/infrastructure/` | Shared engine: planners, scoring, CNN loading, map encoding. |
| `bot/models/` | `cnn_lookahead_1` (next move, volcano) and `cnn_lookahead_4` (position at meteor impact). |
| `training/` | Build datasets, tune, train and evaluate the CNNs. |
| `analysis/` | Studies on game logs: flee behaviour, T-Rex predation, learned meteor scoring. |
| `results/` | Scores of every bot version. |
| `scripts/` | Helper scripts. |
| `local_game_logs/`, `data/` | Local only (git-ignored): recorded games and datasets. |
