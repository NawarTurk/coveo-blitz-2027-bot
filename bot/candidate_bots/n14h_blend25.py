"""
N14h blend 25 = N14c (final = (1-w) x XGBoost + w x rule model) with the rule model's share set to 25%
(N14c = 30%, N14g = 20%). Everything else identical to N14c.
Needs n14c_rule_blend.py, bot/models/behavior/weights.json and bot/models/scorer/.
Use: ./scripts/ship.sh n14h_blend25
"""
import n14c_rule_blend as N14C

N14C.BLEND_W = 0.25
N14C.C.BOT_NAME = "n14h_blend25"


class Bot(N14C.Bot):
    pass
