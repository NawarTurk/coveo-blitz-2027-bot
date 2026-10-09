#!/bin/bash
# usage: ./scripts/ship.sh BOT_NAME   (a file in bot/candidate_bots/, without .py) -> BOT_NAME.zip
set -e
B=${1:?bot name}; R=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d)
cp -R "$R/bot/." "$T/"
sed -i '' "s/^from .* import Bot.*/from $B import Bot/" "$T/bot.py"
sed -i '' 's/^LOG_GAMES = True/LOG_GAMES = False/' "$T"/candidate_bots/*.py
cd "$T"; rm -rf logs README.txt; find . -name __pycache__ -prune -exec rm -rf {} +
python -c "import bot; b=bot.Bot(); print('ok', type(b).__module__)"
find . -name __pycache__ -prune -exec rm -rf {} +
rm -f "$R/$B.zip"; zip -qr "$R/$B.zip" .; rm -rf "$T"; echo "built $B.zip"
