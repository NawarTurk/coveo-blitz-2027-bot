"""
N14e blend 40 = N14c (final = (1-w) x XGBoost + w x rule model) with the rule model's share raised
from 30% to 40%. Everything else identical to N14c.
Needs n14c_rule_blend.py, bot/models/behavior/weights.json and bot/models/scorer/.
Use: ./scripts/ship.sh n14e_blend40
"""
import n14c_rule_blend as N14C

N14C.BLEND_W = 0.40
N14C.C.BOT_NAME = "n14e_blend40"


class Bot(N14C.Bot):
    pass
