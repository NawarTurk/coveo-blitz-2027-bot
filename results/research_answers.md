# Research answers - the game

**Built from 1,054 local games** (`logs/local_game_logs`: 1,151 log files, 97 skipped as incomplete),
analysed 2026-10-10 with `analysis/run_all_analyses.py`. Full numbers: `results/analysis_report_local.md`.
Server checks (11-21 server games) are marked *(server)*.

These are facts about the game, true whichever bot plays. Questions: `results/research_questions.md`.

## 1. Wave rules (Q1-Q7, `waves.py`)

1. **When does a new wave arrive?** 1 tick after the board is empty (6,562 of 6,562), otherwise 100 ticks after the previous wave started.
2. **Does the wave timer reset after a clear?** Yes. Timed waves land on all 100 possible tick-mod-100 values, never on fixed multiples of 100. *(server: 10 of 10 exactly 100 ticks after the previous wave)*
3. **How many dinos can be alive; what happens at the cap?** Max 20. A wave only fills the free slots: 11 alive -> 9 new, 12 alive -> 8 new. 314 waves filled exactly to 20.
4. **At the cap, which members are omitted?** Missing dinos (1,689 partial waves): Triceratops 943, raptors 643, Stegosaurus 627, T-Rex 235, roughly in proportion to the recipes. *(includes some wave-grouping noise)*
5. **Is each wave's species mix fixed?** Yes: wave 1 = 10 Stegosaurus in 100% of games; other waves 68-96% exact (misses = cap and grouping).
6. **Does the mix repeat? How long?** Every 10 waves (wave k = wave k+10 in 86.5% of 2,638 pairs):
   10S | 8S 2V | 6S 4V | 2V 8T | 6S 4V | 2V 7T 1R | 6S 4V | 2V 7T 1R | 2S 2V 3T 3R | 3S 4T 3R.
7. **Same cycle locally and on the server?** Yes *(server: same recipes on 11 games)*; not yet re-run with `--server`.

## 2. Spawns (Q8-Q15, `spawns.py`) - 128,445 spawns

8. **Exact spawn tile predictable?** No: the same wave number in two games shares 2.7% of tiles vs 2.4% by chance.
9. **Predictable with a 3x3 / 4x4 grid?** No. Blocks 9.2-13.1% (even 11.1%) and 5.0-7.2% (even 6.25%): a mild lean to the middle/lower rows only. The best fixed blast (hindsight) covers 4.2% of spawns vs 3.2% random.
10. **Differ by species?** No meaningful difference (all species 4.8-7.7% per 4x4 block).
11. **Depend on the wave number?** No (every wave 8.7-14.1% per 3x3 block).
12. **Where the previous wave spawned?** No: 6.6% vs 6.25% chance.
13. **Clustered within a wave?** No: mean distance 12.6 vs 13.3 for random placement.
14. **Favour terrain (elevation, mountains)?** No: elevation bands within ~2 points of random free tiles; near mountains 27.8% vs 30.3%.
15. **Depend on occupied tiles, lava, corpses?** No: distance to nearest dino 4.0 vs 4.2; never on lava; on a corpse 1.2% vs 0.8%.

## 3. Movement (Q16-Q25, `movement.py`)

16. **Moves with no threat:**
    Stegosaurus stay 14%, down 28%, left/right 24% each, up 10% (keeps drifting).
    Raptor stay 51%. Triceratops stay 27%, down 27%. T-Rex stay 54%.
17. **Reaction when inside a falling blast:** run away 68-80% (T-Rex 80%), stay 13-27%, toward 3-7%. *(sideways shows 0%: script bug, grid distance - to fix)*
18. **By distance from the centre:** on the centre 98% run, 1 tile 91%, 2 tiles 64% (32% stay: already at the edge).
19. **By ticks to impact:** 4 ticks 80% run, 3 ticks 70%, 2 ticks 52% (more stay as they reach safety).
20. **Flee type overall:** away 72%, stay 24%, toward 4%.
21. **Where does a fleeing dino end up?** Always outside the blast: 3 tiles 42%, 4 tiles 38%, 5 tiles 21%. Ran straight away 69%.
22. **Predictable 1-4 ticks ahead?** Simple rules fail: "keeps its last move" hits 40-59% at 1 tick, 7-25% at 4 ticks; at 4 ticks "stays put" is better (12-29%). -> CNN4 is needed.
23. **Does history help simple prediction?** No (4 ticks: stay 18%, last move 12%, majority of last 3 12%).
24. **Window edge:** near the top (trailing) edge every species moves down 28-32%; Stegosaurus near the bottom turns back up (28%).
25. **Scroll deaths:** 10,491 (8% of deaths, all species). A dino on the top row when the window shifts dies 39.5% of the time.

## 4. Species behaviour and the site's claims (Q26-Q36, `species.py`)

26. **Gather in the NE / at elevation 7+?** No: NW 27 / NE 27 / SW 23 / SE 23%; elevation 7+ 60.0% of dinos vs 60.8% of tiles. *(server: 23/24/26/27%)*
27. **Triceratops rotate counter-clockwise (claim 83%)?** No: exactly 50.0 / 50.0% (738,604 moves).
28. **"Double-bounce" shockwave?** No: 96 unexplained deaths in 1,054 games, 18 near an impact (misclassified edge cases). *(server: 0 in 21 games)*
29. **Triceratops toward high ground?** Barely: uphill 30.0% vs 29.0% random (x1.03).
30. **Follow increasing elevation?** Barely: +0.07 per move vs -0.10 random.
31. **Pack tightness / centroid 4 ticks ahead:** loose, 5.3 tiles from the centre on average; "stays put" (1.8 tiles error) beats constant velocity (3.1).
32. **Raptors toward corpses? Radius?** No: lift 0.81-1.03 at every distance (1.0 = no effect). The docs' claim is not visible in the data.
33. **Corpse distance predictive of raptor moves?** No (overall lift 0.95).
34. **Attraction when threatened?** None (0.97).
35. **When does a T-Rex eat?** Only from 1 tile away (4,861 of 4,861 meals). A dino next to a T-Rex is eaten the next tick 65.4% of the time. Prey: Triceratops 2,043, Stegosaurus 1,907, raptors 911.
36. **Lava avoidance:** never step next to lava (0% from 1 tile); avoid it up to 4 tiles (x0.67-0.79).

## 5. Physics and scoring (Q37-Q45, `physics.py`)

37. **Catch rate by free escape tiles:** 0 -> 97%, 1-3 -> 60%, 4-7 -> 35%, 8-14 -> 13%, 15+ -> 2.8%.
    (bot trap table today: 79 / 47 / 26 / 10 / 3.6% - too flat)
38. **Other dinos blocking escapes:** small effect: +2-6 points when an escape tile is occupied (8-14 exits: 15.0% vs 11.9%).
39. **Obstacles:** open ground 3.0% vs window edge 30.0%, lava 29.3%, wall / mountain 27.1%.
40. **Inside a falling blast, survival:** escaped 69.9%, caught 16.9%, died another way 13.1% (455,790 dinos).
41. **After escaping; where should a second meteor land?** 3-5 tiles from the first centre. A blast 3 tiles out along its escape line covers it 69% vs 40% on its start tile.
42. **Follow-up gap 1 / 2 / 3 ticks:** no difference, 11-12% of escapees caught each.
43. **Lava funnel:** at the same escape-tile count, lava nearby changes little (e.g. 4-7 exits: 34.2% vs 34.8%). Lava helps only by cutting escape tiles.
44. **Age-to-points formula:** exact. points = 160 / (1 + age/30), age = the dino's age the tick before it dies (ratio 1.000, 77,658 kills).
45. **Multi-kill multiplier:** 2 kills x1.50, 3 kills x1.93 (rule: 1 + 0.5 per extra kill).

## Known gaps

- Q17-Q20: "sideways" is always 0% (grid distance changes by exactly 1 per move) - script fix pending.
- Q4 includes some wave-grouping noise.
- Q7: run `waves.py --server <team logs>` for a formal local-vs-server check.
