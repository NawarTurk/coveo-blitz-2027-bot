#!/usr/bin/env python3
"""
Wave clear timing per species: for each species, how long it lives, how it dies, and how often it is
the last survivor of a wave (the straggler that delays the next wave).

Usage:   python analysis/wave_clear_timing_per_species.py --logs local_game_logs
Writes:  results/wave_timing.csv + results/wave_timing.png (same as wave_clear_timing.py)
"""
import contextlib, io, sys

import wave_clear_timing as M

if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.main()
    out = buf.getvalue().splitlines()
    start = next(i for i, l in enumerate(out) if l.startswith("PER SPECIES"))
    end = next(i for i in range(start, len(out)) if out[i].startswith("  last/share"))
    print("\n".join(out[start:end + 1]))
