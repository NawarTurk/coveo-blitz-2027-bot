#!/bin/bash
# Build an upload-ready zip of one bot:  ./scripts/ship.sh BOT_NAME  ->  dist/BOT_NAME.zip
set -e
B=${1:?bot name, e.g. n10_wave_modes}
R=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
cp -R "$R/bot/." "$T/"
sed -i '' "s/^from .* import Bot.*/from $B import Bot/" "$T/bot.py"
sed -i '' 's/^LOG_GAMES = True/LOG_GAMES = False/' "$T"/candidate_bots/*.py
cd "$T"
rm -rf logs README.txt
find . -name __pycache__ -prune -exec rm -rf {} +
python -c "import bot; b=bot.Bot(); print('ok', type(b).__module__)"
find . -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$R/dist"
rm -f "$R/dist/$B.zip"
zip -qr "$R/dist/$B.zip" .
cd "$R"; rm -rf "$T"
echo "built dist/$B.zip"
