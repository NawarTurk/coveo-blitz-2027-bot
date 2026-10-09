# Analysis report

Generated 2026-10-08 08:01 from `local_game_logs` (1151 local games).

Purpose: give an LLM the full picture to reason about the next bot improvement.

## Game summary (Coveo Blitz 2027)

- 20x20 visible window that scrolls one row every 5 ticks; 1,000 ticks per game; ~150 ms per tick to answer.
- One action per tick: launch a meteor (lands 4 ticks later, 13-tile diamond blast, max 3 in the air)
  or trigger a volcano (lava spreads from a mountain).
- Dinos (Stegosaurus, Velociraptor, Triceratops, T-Rex) move 1 tile per tick and flee meteors.
- Points per kill = 160 / (1 + age/30) x multi-kill bonus (1 + 0.5 per extra kill on the same impact).
  Scroll deaths count; dinos eaten by a T-Rex give 0.
- Waves of ~10 dinos every 100 ticks, OR 2 ticks after the board is empty.
- Leaderboard: top 2 teams ~160k, ranks 3-6 ~36-47k, us ~15-19k.

## Our bots (bot/candidate_bots/)

- cage_cnn4 (best, ~15.4k avg): candidate spots around dinos and their escape tiles, ranked by a measured
  "trap table" (catch rate vs free escape tiles), scored by CNN4 (predicts every dino's position at impact),
  final score = 70% CNN4 + 30% trap table.
- wide_search_cnn4 (~13.4k): same but scores ~150 spots incl. empty tiles near dinos.
- species_heuristic (~8.4k): no ML, hand-written species flee simulation.
- n1_straggler_hunter (~10k, 2 games): cage_cnn4 + bigger clear bonus + stragglers at full value + aim ahead of runners.

## Server results so far

```
BOT RESULTS (Coveo Blitz 2027 server tryouts)
=============================================

Each score is one server game. Maps change every game, so single games swing by 5k+;
compare averages, not single games.


CANDIDATE BOTS (bot/candidate_bots/)

BOT                  WHAT IT DOES                                   VERSIONS        GAMES  SCORES                                         AVG      BEST
-------------------  ---------------------------------------------  --------------  -----  ---------------------------------------------  -------  ------
cage_cnn4            traps dinos; CNN4 scores ~60 meteor spots      v10 v15 v16     7      16,293 14,999 15,774 8,803 16,228 18,888 16,986  15,424   18,888
wide_search_cnn4     same, but CNN4 scores ~150 spots               v17 v27         4      15,437 12,250 10,390 15,624                     13,425   15,624
species_heuristic    no ML: rule-based species flee simulation      v21 v22         3      9,143 8,698 7,402                              8,414    9,143


ENGINE LAYERS (bot/infrastructure/) - each was once a full bot

BOT                  WHAT IT DOES                                   VERSIONS        GAMES  SCORES                                         AVG      BEST
-------------------  ---------------------------------------------  --------------  -----  ---------------------------------------------  -------  ------
cage_planner         traps dinos, CNN1 only (Cage A)                v9              1      13,904                                         13,904   13,904
fast_clear           better reward: young dinos, multi-kill, clear  v6 v7           2      8,477 8,470                                    8,474    8,477


NOTES

- cage_cnn4 = our leaderboard score (18,888, v16).
- v18 is left out: the zip was mixed up, so we don't know which bot ran.
- wide_search_cnn4 includes v27 (Z1L): same decisions, it only adds an end-of-game report.
- species_heuristic includes v22: that was meant as ZH2, but the CNN never ran, so it played as ZH1.

NEXT TEST
- N0 = cage_cnn4 with the new CNN4 (trained on 1,151 games)   -> pending
```

## Analyses

### Wave clear timing

*Question:* How many ticks each wave takes to die (first kill, 50%, 80%, all), how the last dino of a wave dies, per-species lifetimes, and how wave speed relates to score.

*Script:* `analysis/wave_clear_timing.py` (112 s)

```
1054 games, 13156 waves (12.5 per game), 88% of waves fully cleared (rest: still alive at game end or overlapping)
dinos per wave: median 10

TICKS AFTER A WAVE SPAWNS (cleared waves, median):
  first death        6
  50% dead          29
  80% dead          55
  all dead          87   <- ticks per wave
  share of the wave's time spent on the last 20% of dinos: 35% (median)
  next wave came after a clear: 6964 waves, median gap 1 ticks | came on the 100-tick schedule while the previous wave was still alive: 4584

HOW THE LAST DINO OF EACH WAVE DIED:
  meteor    75.4%
  scroll    12.0%
  lava       9.7%
  eaten      2.8%
  other      0.1%
  species of the last dino: Stegosaurus 57%, Triceratops 19%, Velociraptor 14%, Tyrannosaurus 11%

ALL DEATHS IN WAVES:
  meteor_kills   68.4%
  lava_kills     15.0%
  scroll          8.2%
  eaten           3.8%

PER SPECIES:
  species         spawns life (med)  meteor   lava  scroll  eaten  alive  is last  last/share
  Stegosaurus        46%         38     69%    14%      8%     3%     6%      57%        1.2x
  Triceratops        26%         25     71%    13%      7%     6%     3%      19%        0.7x
  Velociraptor       21%         30     65%    21%      9%     3%     2%      14%        0.6x
  Tyrannosaurus       7%         46     64%    14%     11%     0%    10%      11%        1.6x
  last/share > 1 = this species is the straggler more often than its numbers explain

lowest 10% games: score 3,953 | waves/game 9.8 | ticks per wave 240

highest 10% games: score 17,361 | waves/game 17.4 | ticks per wave 55
saved results/wave_timing.png

wrote results/wave_timing.csv
```

### Flee behaviour

*Question:* How dinos react to a falling meteor: hit rate, first move (away / sideways / toward / stay), escape paths, catch rate vs free escape tiles (walls), chain shots.

*Script:* `analysis/flee_behavior.py` (139 s)

```
reading 1151 games...
1050 complete games

==========================================================================
1. HIT RATE
   meteors: 583,930 | kills per meteor 0.15 | meteors with 0 kills  85.7% | 2+ kills   0.7%
   dinos inside the blast when it appears: 442,780
     still inside at impact (killed):  17.4% | escaped:  71.9% | died another way first:  10.7%
   by distance from centre at launch (0 = on the centre):
     0: caught  12.7%  (n=23,374)
     1: caught  17.9%  (n=154,835)
     2: caught  17.6%  (n=264,571)

==========================================================================
2. REACTION: first move after the meteor appears
   threatened: away  74.7%  sideways   0.0%  toward   4.9%  stay  20.4%
   calm dinos stay  26.2% of the time
   species            away    side  toward    stay  calm stay  caught
   Stegosaurus       73.9%    0.0%    3.9%   22.2%      14.2%   16.4%
   Triceratops       78.2%    0.0%    5.0%   16.9%      26.6%   16.6%
   Tyrannosaurus     79.8%    0.0%    8.9%   11.3%      50.9%   18.0%
   Velociraptor      70.4%    0.0%    6.7%   22.9%      51.7%   21.3%

==========================================================================
3. ESCAPE: distance from the blast centre at impact (escaped dinos)
   d=1:   0.0%  d=2:   0.0%  d=3:  41.8%  d=4:  37.6%  d=5:  20.6%
   average 3.8 tiles (blast radius is 2: 3 = just outside)
   ran (almost) straight away from the centre:  69.4%

==========================================================================
4. WALLS: caught rate vs free escape tiles around the dino
    0-0  free escape tiles: caught  60.3%  (n=1,545)
    1-3  free escape tiles: caught  46.2%  (n=29,323)
    4-7  free escape tiles: caught  28.9%  (n=111,044)
    8-14 free escape tiles: caught  11.9%  (n=243,527)
   15-24 free escape tiles: caught   2.7%  (n=57,341)

==========================================================================
5. CHAINS: a 2nd meteor landing 1-2 ticks later next to the first
   single meteor: caught  16.8% (n=204,739)
   with follow-up: caught by 1st  18.0%, by 1st OR 2nd  29.4% (n=238,041)
```

### T-Rex predation

*Question:* How dinos die (meteor / lava / scroll / eaten / other), how many are eaten by a T-Rex and the points lost.

*Script:* `analysis/trex_predation.py` (63 s)

```
1054 games

How dinos die (per game avg, % of all deaths):
  meteor    83.4    71.7%
  lava      18.3    15.7%
  scroll    10.0     8.6%
  eaten      4.6     4.0%
  other      0.1     0.1%

EATEN: 4.6 dinos per game
  points lost per game: avg 410  (median 352, max 1903)  = 4.2% of our average score 9716
  age of eaten dinos: median 22, mean 66  (value at median age 92 pts)
  species eaten: {'Triceratops': 2043, 'Velociraptor': 911, 'Stegosaurus': 1907}
  T-Rex distance just before: {1: 4861}

T-REX: 8.0 per game | meals per T-Rex: avg 0.58, max 11 | % of T-Rexes that eat at least once: 32% | ticks alive avg 65
  meals per T-Rex per 100 ticks alive: 0.90
  -> killing a T-Rex ~X ticks earlier saves about X * 0.0090 meals x ~92 pts
```

### Spawn sequence (exact tiles)

*Question:* Inside one game, can the next wave's spawn tiles be predicted from earlier waves (same spots, rotation, mirror, shift, screen heatmap)? Scored by % of newborns 3 pre-aimed blasts would catch.

*Script:* `analysis/spawn_sequence.py` (90 s)

```
species: all
1054 games, 12102 waves (11.5 per game), median 10 dinos per wave | train 843 / test 211 games

SPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:
  0:4 1:4 2:4 3:5 4:4 5:4 6:5 7:5 8:5 9:5 10:5 11:5 12:6 13:6 14:6 15:6 16:6 17:6 18:6 19:3
SPAWN COLUMN (0 = left), % of spawns:
  0:4 1:6 2:5 3:5 4:5 5:5 6:5 7:5 8:5 9:5 10:5 11:5 12:5 13:5 14:5 15:5 16:5 17:5 18:6 19:4

WAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:
  -180deg:6 -135deg:12 -90deg:12 -45deg:13 +0deg:14 +45deg:14 +90deg:13 +135deg:12 +180deg:6
  (a real rotation would put most pairs in one bin; ~equal bins = random)

% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):
  perfect                     61.0%
  screen heatmap              12.5%
  blind                       10.2%
  same as last                 9.5%
  mirror x                     9.4%
  best shift (-1, 0)           9.3%
  rot90                        9.1%
  rot270                       8.8%
  mirror y                     8.7%
  rot180                       8.6%

If nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.
```

### Spawn sequence per species

*Question:* Same prediction test, one species at a time.

*Script:* `analysis/spawn_sequence_per_species.py` (15792 s)

```

============================== Stegosaurus ==============================
species: Stegosaurus
1054 games, 8463 waves (8.0 per game), median 6 dinos per wave | train 843 / test 211 games

SPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:
  0:4 1:4 2:4 3:4 4:4 5:4 6:4 7:5 8:5 9:5 10:5 11:5 12:6 13:8 14:6 15:6 16:6 17:6 18:6 19:3
SPAWN COLUMN (0 = left), % of spawns:
  0:4 1:5 2:5 3:5 4:5 5:5 6:5 7:5 8:5 9:5 10:5 11:5 12:5 13:5 14:5 15:5 16:5 17:5 18:6 19:4

WAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:
  -180deg:6 -135deg:11 -90deg:13 -45deg:14 +0deg:14 +45deg:14 +90deg:12 +135deg:11 +180deg:5
  (a real rotation would put most pairs in one bin; ~equal bins = random)

% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):
  perfect                     72.9%
  screen heatmap              13.6%
  blind                       10.5%
  best shift (1, 0)            9.1%
  rot270                       8.7%
  mirror x                     8.6%
  rot180                       8.5%
  mirror y                     8.3%
  same as last                 8.3%
  rot90                        8.2%

If nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.

============================== Velociraptor ==============================
species: Velociraptor
1053 games, 9145 waves (8.7 per game), median 2 dinos per wave | train 842 / test 211 games

SPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:
  0:4 1:3 2:3 3:6 4:5 5:5 6:5 7:5 8:5 9:5 10:5 11:5 12:6 13:6 14:6 15:6 16:6 17:6 18:6 19:3
SPAWN COLUMN (0 = left), % of spawns:
  0:5 1:5 2:5 3:5 4:5 5:5 6:5 7:6 8:5 9:5 10:5 11:5 12:5 13:5 14:5 15:5 16:5 17:5 18:5 19:4

WAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:
  -180deg:6 -135deg:13 -90deg:12 -45deg:12 +0deg:13 +45deg:13 +90deg:13 +135deg:12 +180deg:6
  (a real rotation would put most pairs in one bin; ~equal bins = random)

% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):
  perfect                     93.6%
  screen heatmap              11.1%
  blind                       10.0%
  same as last                 8.1%
  best shift (1, 0)            8.0%
  rot270                       8.0%
  mirror x                     7.9%
  rot90                        7.7%
  rot180                       7.3%
  best shift (-1, 2)           7.2%
  mirror y                     6.7%

If nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.

============================== Triceratops ==============================
species: Triceratops
1050 games, 4702 waves (4.5 per game), median 6 dinos per wave | train 840 / test 210 games

SPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:
  0:4 1:4 2:5 3:4 4:4 5:4 6:5 7:5 8:5 9:5 10:5 11:5 12:5 13:6 14:6 15:6 16:6 17:6 18:6 19:3
SPAWN COLUMN (0 = left), % of spawns:
  0:4 1:6 2:6 3:5 4:5 5:5 6:5 7:5 8:5 9:5 10:5 11:5 12:5 13:5 14:5 15:6 16:5 17:6 18:6 19:4

WAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:
  -180deg:6 -135deg:11 -90deg:12 -45deg:14 +0deg:13 +45deg:12 +90deg:12 +135deg:13 +180deg:6
  (a real rotation would put most pairs in one bin; ~equal bins = random)

% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):
  perfect                     82.7%
  screen heatmap              12.8%
  blind                       10.8%
  best shift (-1, 0)           9.5%
  same as last                 9.5%
  best shift (1, 0)            9.5%
  rot90                        9.1%
  best shift (-1, 2)           8.8%
  rot270                       8.7%
  mirror x                     8.7%
  mirror y                     8.5%
  rot180                       7.9%

If nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.

============================== Tyrannosaurus ==============================
species: Tyrannosaurus
1037 games, 3228 waves (3.1 per game), median 3 dinos per wave | train 829 / test 208 games

SPAWN ROW on screen (0 = top, 19 = bottom), % of spawns:
  0:3 1:5 2:3 3:6 4:4 5:5 6:4 7:5 8:5 9:5 10:5 11:5 12:6 13:5 14:6 15:6 16:5 17:7 18:8 19:2
SPAWN COLUMN (0 = left), % of spawns:
  0:2 1:9 2:5 3:5 4:5 5:5 6:5 7:6 8:5 9:4 10:4 11:5 12:5 13:5 14:6 15:5 16:4 17:5 18:8 19:2

WAVE CENTRE: angle change from one wave to the next (around the screen centre), % of wave pairs:
  -180deg:7 -135deg:12 -90deg:12 -45deg:13 +0deg:13 +45deg:11 +90deg:13 +135deg:12 +180deg:6
  (a real rotation would put most pairs in one bin; ~equal bins = random)

% OF NEW-WAVE DINOS CAUGHT BY 3 PRE-AIMED BLASTS (test games):
  perfect                    100.0%
  screen heatmap              12.8%
  blind                        9.7%
  best shift (1, 0)            7.2%
  same as last                 6.8%
  rot90                        6.7%
  best shift (-1, 2)           6.4%
  rot180                       6.3%
  mirror x                     6.2%
  best shift (-1, 0)           6.1%
  mirror y                     6.0%
  rot270                       5.8%
  best shift (3, 1)            5.5%

If nothing beats 'blind' / 'screen heatmap' clearly, spawn spots are random within a game too.
```

### Spawn blocks (3x3 screen blocks)

*Question:* Patterns in WHICH of 9 screen blocks waves spawn: sequence from wave to wave (information in bits vs shuffled), memory (blocks repeat or cycle), spread inside a wave.

*Script:* `analysis/spawn_blocks.py` (1932 s)

```
species: all
1053 games, 12100 waves

WHERE DINOS SPAWN (% per block):
   b1:10.7%  b2: 9.9%  b3: 9.0%
   b4:13.2%  b5:13.0%  b6:11.3%
   b7:11.4%  b8:11.8%  b9: 9.8%
   (block areas differ slightly: middle row/column are 6 tiles, edges 7)

INSIDE ONE WAVE: dinos land in 5.83 different blocks on average (random would give 6.10)
   more than random = spread out on purpose; fewer = clustered

IDEA 1 - SEQUENCE of each wave's main block (11047 wave pairs):
   information the previous block gives about the next: 0.0043 bits (no-order baseline 0.0036; max possible ~3.17)
   next main block = same as previous: 13.2% (no-order baseline 13.2%)
   transition table: row = this wave's main block, column = next wave's (% of rows)
          b1   b2   b3   b4   b5   b6   b7   b8   b9
   b1    18   15   10   17   13    9    7    8    5
   b2    19   15   10   16   14    9    8    5    4
   b3    17   16   12   15   13    8    7    7    5
   b4    19   14   11   16   13    9    6    8    4
   b5    19   13   10   17   12    8    8    8    4
   b6    17   15   10   15   13   10    7    8    5
   b7    16   15   10   15   14    9    8    8    4
   b8    18   14    8   16   14   10    8    8    3
   b9    18   13    9   16   14    9    7    8    5
   strongest transitions: b2->b1 19%, b4->b1 19%, b5->b1 19%  (random ~11%)

IDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)
   block used in the previous wave        :  65.8% vs not used  63.1%   | shuffled:  65.4% vs  63.7%
   block used in any of the last 3 waves  :  65.3% vs not used  60.3%   | shuffled:  65.0% vs  60.6%
   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle

VERDICT: NO usable pattern
```

### Spawn blocks per species

*Question:* Same block tests, one species at a time.

*Script:* `analysis/spawn_blocks_per_species.py` (7659 s)

```

============================== Stegosaurus ==============================
species: Stegosaurus
1052 games, 8459 waves

WHERE DINOS SPAWN (% per block):
   b1:10.1%  b2: 9.6%  b3: 8.8%
   b4:13.7%  b5:13.8%  b6:11.7%
   b7:11.2%  b8:11.6%  b9: 9.6%
   (block areas differ slightly: middle row/column are 6 tiles, edges 7)

INSIDE ONE WAVE: dinos land in 4.09 different blocks on average (random would give 4.25)
   more than random = spread out on purpose; fewer = clustered

IDEA 1 - SEQUENCE of each wave's main block (7407 wave pairs):
   information the previous block gives about the next: 0.0071 bits (no-order baseline 0.0052; max possible ~3.17)
   next main block = same as previous: 13.3% (no-order baseline 13.1%)
   transition table: row = this wave's main block, column = next wave's (% of rows)
          b1   b2   b3   b4   b5   b6   b7   b8   b9
   b1    19   15   11   16   15    8    7    7    3
   b2    18   15   10   17   14    8    7    7    5
   b3    17   15   10   16   15    8    9    5    4
   b4    19   15   11   16   14    8    7    7    3
   b5    20   14   11   16   14   10    6    7    3
   b6    18   14   12   16   14   10    7    6    4
   b7    16   15    8   19   15    9    7    8    2
   b8    17   10   11   17   12   12    8    9    4
   b9    21   10   11   12   16   11    7    8    4
   strongest transitions: b9->b1 21%, b5->b1 20%, b7->b4 19%  (random ~11%)

IDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)
   block used in the previous wave        :  45.9% vs not used  42.1%   | shuffled:  45.4% vs  45.5%
   block used in any of the last 3 waves  :  42.0% vs not used  42.2%   | shuffled:  46.0% vs  45.1%
   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle

VERDICT: POSSIBLE pattern - worth a closer look

============================== Velociraptor ==============================
species: Velociraptor
1052 games, 9143 waves

WHERE DINOS SPAWN (% per block):
   b1:11.0%  b2:10.4%  b3: 9.0%
   b4:13.2%  b5:12.7%  b6:10.8%
   b7:11.1%  b8:11.9%  b9: 9.8%
   (block areas differ slightly: middle row/column are 6 tiles, edges 7)

INSIDE ONE WAVE: dinos land in 2.44 different blocks on average (random would give 2.46)
   more than random = spread out on purpose; fewer = clustered

IDEA 1 - SEQUENCE of each wave's main block (8091 wave pairs):
   information the previous block gives about the next: 0.0078 bits (no-order baseline 0.0067; max possible ~3.17)
   next main block = same as previous: 15.1% (no-order baseline 14.1%)
   transition table: row = this wave's main block, column = next wave's (% of rows)
          b1   b2   b3   b4   b5   b6   b7   b8   b9
   b1    23   17   11   16   12    8    6    5    2
   b2    24   20   13   14   10    7    6    5    2
   b3    24   19   13   15   10    9    5    5    2
   b4    21   16   13   16   12    8    8    4    2
   b5    22   18   13   15   11    7    5    6    4
   b6    21   17    9   14   11    8    8    8    3
   b7    21   16   11   17   12    7    7    6    3
   b8    21   14   12   15   13    8    8    5    3
   b9    19   13    9   19   12   10    7    8    2
   strongest transitions: b3->b1 24%, b2->b1 24%, b1->b1 23%  (random ~11%)

IDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)
   block used in the previous wave        :  25.4% vs not used  26.3%   | shuffled:  27.7% vs  27.0%
   block used in any of the last 3 waves  :  25.4% vs not used  25.0%   | shuffled:  27.5% vs  26.7%
   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle

VERDICT: NO usable pattern

============================== Triceratops ==============================
species: Triceratops
1050 games, 4702 waves

WHERE DINOS SPAWN (% per block):
   b1:11.1%  b2: 9.7%  b3: 9.5%
   b4:12.8%  b5:12.2%  b6:10.8%
   b7:11.5%  b8:12.0%  b9:10.3%
   (block areas differ slightly: middle row/column are 6 tiles, edges 7)

INSIDE ONE WAVE: dinos land in 3.76 different blocks on average (random would give 4.05)
   more than random = spread out on purpose; fewer = clustered

IDEA 1 - SEQUENCE of each wave's main block (3652 wave pairs):
   information the previous block gives about the next: 0.0108 bits (no-order baseline 0.0114; max possible ~3.17)
   next main block = same as previous: 13.1% (no-order baseline 14.2%)
   transition table: row = this wave's main block, column = next wave's (% of rows)
          b1   b2   b3   b4   b5   b6   b7   b8   b9
   b1    23   15   11   14   11    7    6    7    5
   b2    18   16   12   15   12    8    5    8    5
   b3    19   16   12   14   10    7    6    7    7
   b4    21   15   16   11   13    7    7    8    4
   b5    24   13   10   15   13    6    7    7    6
   b6    22   16   15   11   11    7    7    7    5
   b7    23   10   12   15    9   10    6    9    5
   b8    20   15   12   16   10   10    6    6    4
   b9    19   10   12   19   12    9    6    7    5
   strongest transitions: b5->b1 24%, b7->b1 23%, b1->b1 23%  (random ~11%)

IDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)
   block used in the previous wave        :  40.3% vs not used  38.5%   | shuffled:  41.5% vs  42.2%
   block used in any of the last 3 waves  :  40.0% vs not used  40.1%   | shuffled:  42.4% vs  42.0%
   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle

VERDICT: NO usable pattern

============================== Tyrannosaurus ==============================
species: Tyrannosaurus
944 games, 3042 waves

WHERE DINOS SPAWN (% per block):
   b1:11.0%  b2:10.0%  b3: 9.3%
   b4:12.3%  b5:12.0%  b6:11.9%
   b7:12.5%  b8:11.2%  b9: 9.9%
   (block areas differ slightly: middle row/column are 6 tiles, edges 7)

INSIDE ONE WAVE: dinos land in 2.02 different blocks on average (random would give 2.03)
   more than random = spread out on purpose; fewer = clustered

IDEA 1 - SEQUENCE of each wave's main block (2098 wave pairs):
   information the previous block gives about the next: 0.0182 bits (no-order baseline 0.0254; max possible ~3.17)
   next main block = same as previous: 13.2% (no-order baseline 11.4%)
   transition table: row = this wave's main block, column = next wave's (% of rows)
          b1   b2   b3   b4   b5   b6   b7   b8   b9
   b1    26   16   15   14   10    7    6    3    3
   b2    25   16   12   12   10    8    8    3    5
   b3    22   20   13   13   10    6    6    7    2
   b4    25   17   14   14   12    8    2    4    3
   b5    25   18   10   20    9    5    6    4    4
   b6    27   18   13   15    8    7    5    3    5
   b7    24   19   14   13    9    8    5    5    2
   b8    24   19   19   15    7    4    4    3    5
   b9    25   18    8   16   11    6    7    5    4
   strongest transitions: b6->b1 27%, b1->b1 26%, b2->b1 25%  (random ~11%)

IDEA 2 - MEMORY: chance a block gets spawns in the next wave (shuffled = same waves, order removed)
   block used in the previous wave        :  27.3% vs not used  27.5%   | shuffled:  21.5% vs  22.5%
   block used in any of the last 3 waves  :  12.7% vs not used  12.9%   | shuffled:  17.4% vs  21.1%
   real gap = shuffled gap -> no memory | bigger -> blocks repeat | smaller -> blocks rotate / cycle

VERDICT: POSSIBLE pattern - worth a closer look
```

## Questions for the next step

1. Where does the time per wave go, and which change would clear waves fastest?
2. Is anything predictable that the bot does not use yet (spawns, flee direction, species)?
3. Which single change to cage_cnn4 should be tested next, and how to measure it fairly (same seeds, 100 local games)?
