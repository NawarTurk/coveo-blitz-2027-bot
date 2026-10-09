"""
N5 hold trapped = N4 survival follow with ONE change: SURVIVE_MIN 0.5 -> 0.6.

Why: over 4 server games of N4, dinos the bot rated 50-70% to survive (1-3 free escape tiles) survived only
9 of 54 times (17%). Shots fired at them were mostly wasted. With 0.6 the bot holds for those and fires
only when a doomed dino has 4+ free escape tiles.

Logs are N4's ([N4] lines show SURVIVE_MIN 0.6 on the "loaded" line) + full T| trace + [REPORT].
Use: ./scripts/ship.sh n5_hold_trapped
"""
import n4_survival_follow as N4

N4.SURVIVE_MIN = 0.6
N4.C.BOT_NAME = "n5_hold_trapped"


class Bot(N4.Bot):
    pass
