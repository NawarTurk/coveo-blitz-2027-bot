#!/bin/bash
# Play N local games with one bot against the local game server (it must already be running).
#   ./scripts/play_local.sh n14_xgb_scorer 100                 -> logs/local_n14_xgb_scorer/
#   ./scripts/play_local.sh n15_xgb_v2 300 explore             -> logs/local_explore/   (3rd arg = folder name)
#   BLITZ_EXPLORE=0.2 ./scripts/play_local.sh n15_xgb_v2 300 explore      (exploration games for training)
# Per game: output with T| traces -> <folder>/game_<time>_<i>.txt, the bot's own .jsonl game log -> <folder>/jsonl/
# Scores -> <folder>/scores.txt. bot/bot.py is switched to BOT and put back at the end (also on Ctrl-C).
B=${1:?bot name, e.g. n14_xgb_scorer}
N=${2:-100}
TAG=${3:-$B}
R=$(cd "$(dirname "$0")/.." && pwd)
OUT="$R/logs/local_$TAG"
mkdir -p "$OUT/jsonl" "$R/bot/logs"
cp "$R/bot/bot.py" "$R/bot/bot.py.bak"
trap 'mv "$R/bot/bot.py.bak" "$R/bot/bot.py"; echo "bot.py restored"' EXIT
sed -i '' "s/^from .* import Bot.*/from $B import Bot/" "$R/bot/bot.py"
cd "$R/bot"
echo "bot $B | $N games | folder logs/local_$TAG | BLITZ_EXPLORE=${BLITZ_EXPLORE:-0}"
for i in $(seq 1 "$N"); do
  f="$OUT/game_$(date +%Y%m%d_%H%M%S)_$i.txt"
  touch "$OUT/.start"
  python application.py > "$f" 2>&1
  find "$R/bot/logs" -maxdepth 1 -name "game_*.jsonl" -newer "$OUT/.start" -exec mv {} "$OUT/jsonl/" \;
  s=$(grep -o "Final score: [0-9]*" "$f" | grep -o "[0-9]*$")
  echo "$i ${s:-NO_SCORE}" | tee -a "$OUT/scores.txt"
done
awk '$2 ~ /^[0-9]+$/ {n++; t+=$2} END {if (n) printf "%s: %d games, average %.0f\n", "'"$TAG"'", n, t/n}' "$OUT/scores.txt"
