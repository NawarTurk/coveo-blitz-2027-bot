"""
Which bot plays. The server runs application.py, which imports Bot from here.
To play another bot, change the import below:  cage_cnn4 (best) | wide_search_cnn4 | species_heuristic   (files in candidate_bots/)
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, "candidate_bots"), os.path.join(_here, "infrastructure")]

from cage_cnn4 import Bot  # noqa: E402,F401   <- best so far (Cage B)
