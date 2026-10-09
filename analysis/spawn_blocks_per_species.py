#!/usr/bin/env python3
"""
spawn_blocks for each species separately (Stegosaurus, Velociraptor, Triceratops, Tyrannosaurus):
same tests as spawn_blocks.py, but only the spawns of one species at a time.

Usage:   python analysis/spawn_blocks_per_species.py --logs local_game_logs
"""
import argparse, sys

import spawn_blocks as M

SPECIES = ["Stegosaurus", "Velociraptor", "Triceratops", "Tyrannosaurus"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="local_game_logs")
    logs = ap.parse_args().logs
    for sp in SPECIES:
        print("\n" + "=" * 30 + f" {sp} " + "=" * 30)
        sys.argv = [sys.argv[0], "--logs", logs, "--species", sp]
        M.main()
