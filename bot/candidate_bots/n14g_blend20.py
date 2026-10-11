"""
N14g blend 20 = N14c (final = (1-w) x XGBoost + w x rule model) with the rule model's share raised
from 30% to 20%. Everything else identical to N14c.
Needs n14c_rule_blend.py, bot/models/behavior/weights.json and bot/models/scorer/.
Use: ./scripts/ship.sh n14g_blend20
"""
import n14c_rule_blend as N14C

N14C.BLEND_W = 0.20
N14C.C.BOT_NAME = "n14g_blend20"


class Bot(N14C.Bot):
    pass
