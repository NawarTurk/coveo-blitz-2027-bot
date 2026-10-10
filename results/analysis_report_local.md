# Game analysis report

Generated 2026-10-10 16:59 from `logs/local_game_logs` (1151 local games).

Answers to the game questions Q1-Q45 (results/research_questions.md). Game facts only, no bot results.

## Game summary (Coveo Blitz 2027)

- 20x20 visible window that scrolls one row every 5 ticks; 1,000 ticks per game; ~150 ms per tick to answer.
- One action per tick: launch a meteor (lands 4 ticks later, 13-tile diamond blast, max 3 in the air)
  or trigger a volcano (lava spreads from a mountain).
- Dinos (Stegosaurus, Velociraptor, Triceratops, T-Rex) move 1 tile per tick and flee meteors.
- Points per kill = 160 / (1 + age/30) x multi-kill bonus (1 + 0.5 per extra kill on the same impact).
  Scroll deaths count; dinos eaten by a T-Rex give 0.
- Waves of 10 dinos 100 ticks after the previous wave started, OR 1 tick after the board is empty; max 20 alive.
- Species per wave follow a fixed 10-wave cycle: 10S / 8S2V / 6S4V / 2V8T / 6S4V / 2V7T1R / 6S4V / 2V7T1R / 2S2V3T3R / 3S4T3R.

## Analyses

### Waves (Q1-Q7)

*Question:* Wave rules: when waves come, timer reset, 20 cap, fixed species recipe, 10-wave cycle.

*Script:* `analysis/waves.py` (111 s)

```
waves.py | 1054 games from logs/local_game_logs

Q1. When does a new wave arrive?
    -> after a clear: {1: 6562} ticks after the board went empty (n=6562); otherwise on the timer. Triggers: {'start': 1054, 'timer': 5540, 'clear': 6562}

Q2. Does the wave timer reset after we clear the board?
    -> timed waves come exactly 100 ticks after the previous wave started: 5540 waves; their ticks mod 100 spread over 100 different values (resets: not fixed multiples of 100)

Q3. How many dinos can be alive at once, and what happens to a wave at the cap?
    -> max alive seen 20; waves arriving with dinos still alive: 5446, of which 314 filled exactly up to 20
       waves arriving with N alive -> new dinos: 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:9, 12:8

Q4. At the 20 cap, which members of a wave are omitted?
    -> 1689 partial waves; missing dinos by species: {'V': 643, 'T': 943, 'S': 627, 'R': 235}

Q5. Is the species mix of each wave fixed?
    -> yes
       wave  1: 10S            100.0% of 1054 games
       wave  2: 8S 2V          85.0% of 1054 games
       wave  3: 6S 4V          80.3% of 1054 games
       wave  4: 2V 8T          80.5% of 1053 games
       wave  5: 6S 4V          85.1% of 1052 games
       wave  6: 2V 7T 1R       74.9% of 1051 games
       wave  7: 6S 4V          85.8% of 1050 games
       wave  8: 2V 7T 1R       76.2% of 1050 games
       wave  9: 2S 2V 3T 3R    67.7% of 1050 games
       wave 10: 3S 4T 3R       68.5% of 1050 games
       wave 11: 10S            84.9% of 817 games
       wave 12: 8S 2V          93.0% of 571 games
       wave 13: 6S 4V          95.5% of 449 games
       wave 14: 2V 8T          95.9% of 290 games
       wave 15: 6S 4V          96.0% of 201 games
       wave 16: 2V 7T 1R       90.3% of 155 games
       wave 17: 6S 4V          95.9% of 98 games
       wave 18: 2V 7T 1R       87.2% of 39 games
       wave 19: 2S 2V 3T 3R    85.7% of 14 games

Q6. Does the mix repeat in a cycle? How long?
    -> wave k = wave k+10 in 86.5% of 2638 pairs; cycle: 10S | 8S 2V | 6S 4V | 2V 8T | 6S 4V | 2V 7T 1R | 6S 4V | 2V 7T 1R | 2S 2V 3T 3R | 3S 4T 3R

Q7. Same cycle locally and on the server?
    -> pass --server <folder of team logs> to compare

saved -> results/analysis/local/waves.txt
```

### Spawns (Q8-Q15)

*Question:* Where dinos spawn: exact tiles, grids, species, wave number, repeats, clustering, terrain.

*Script:* `analysis/spawns.py` (119 s)

```
spawns.py | 1054 games, 128445 spawns from logs/local_game_logs

Q8. Can we predict the exact spawn tile?
    -> same wave number in two games shares 2.7% of tiles vs 2.4% by chance
       3x3 grid: blocks 9.2-13.1% (even 11.1%, noise +-0.2)
           10.8  10.0   9.2
           13.1  12.9  11.2
           11.3  11.7   9.7
       4x4 grid: blocks 5.0-7.2% (even 6.2%, noise +-0.1)
            5.3   5.2   5.0   5.3
            6.1   5.9   5.9   6.0
            7.0   7.1   7.2   7.0
            6.6   6.9   6.8   6.5

Q9. Can we predict spawns with a 3x3 or 4x4 grid?
    -> best single fixed blast (hindsight) covers 4.2% of spawns vs 3.2% for a random spot

Q10. Do spawn positions differ by species?
    -> 4x4 block range per species (even 6.25%):
       S: n=59560 range 5.0-7.7% (noise +-0.2)
       V: n=27486 range 5.1-7.1% (noise +-0.3)
       T: n=33010 range 4.8-7.1% (noise +-0.3)
       R: n=8389 range 5.3-7.2% (noise +-0.5)

Q11. Do spawn positions depend on the wave number?
    -> 3x3 block range per cycle position (even 11.1%):
       wave  1: n=18565 range 9.6-13.1% (noise +-0.5)
       wave  2: n=16024 range 9.1-13.8% (noise +-0.5)
       wave  3: n=14666 range 8.7-13.6% (noise +-0.5)
       wave  4: n=13044 range 8.8-13.1% (noise +-0.6)
       wave  5: n=12299 range 9.0-14.1% (noise +-0.6)
       wave  6: n=11681 range 9.4-12.6% (noise +-0.6)
       wave  7: n=11282 range 9.0-13.7% (noise +-0.6)
       wave  8: n=10538 range 9.3-12.7% (noise +-0.6)
       wave  9: n=10218 range 8.9-13.5% (noise +-0.6)
       wave 10: n=10128 range 9.5-12.9% (noise +-0.6)

Q12. Does a wave spawn where the previous wave spawned?
    -> next wave's spawns in the previous wave's busiest 4x4 block: 6.6% (chance 6.25%)

Q13. Do dinos of one wave spawn close together?
    -> mean distance between dinos of a wave 12.6 vs 13.3 for random placement

Q14. Do spawns favour certain terrain (elevation, mountains)?
    -> elevation of spawn tiles vs random free tiles: 0-3: 16.1% vs 17.7%, 4-6: 25.1% vs 23.3%, 7-9: 21.1% vs 18.8%, 10-+: 37.6% vs 40.3%
       mountain within 2 tiles: spawns 27.8% vs random free tiles 30.3%

Q15. Do spawns depend on occupied tiles, lava, corpses?
    -> distance to the nearest other dino: spawns 4.0 vs random free tile 4.2; spawns on lava 0; on a corpse 1.2% vs random 0.8%

saved -> results/analysis/local/spawns.txt
```

### Movement (Q16-Q25)

*Question:* How dinos move calm and threatened, flee patterns, predictability 1-4 ticks ahead, edges, scrolling.

*Script:* `analysis/movement.py` (135 s)

```
movement.py | 1054 games from logs/local_game_logs

Q16. How does each species move with no threat?
    -> calm moves per species (down = +y = the way the window scrolls):
       S: stay 13.7% up 9.7% down 27.6% left 24.7% right 24.4% (n=1915255)
       V: stay 50.5% up 9.1% down 17.2% left 11.6% right 11.5% (n=683062)
       T: stay 27.4% up 13.8% down 26.6% left 16.0% right 16.2% (n=592425)
       R: stay 54.3% up 7.0% down 18.3% left 10.1% right 10.3% (n=358965)

Q17. How does each species react to a meteor launched near it?
    -> first move while inside a falling blast:
       S: away 70.7% sideways 0.0% toward 3.0% stay 26.2% (n=376477) | calm stay 13.7%
       V: away 68.0% sideways 0.0% toward 5.1% stay 26.9% (n=125880) | calm stay 50.5%
       T: away 76.3% sideways 0.0% toward 3.9% stay 19.8% (n=173101) | calm stay 27.4%
       R: away 80.1% sideways 0.0% toward 6.7% stay 13.2% (n=45602) | calm stay 54.3%

Q18. Does reaction depend on distance from the blast centre?
    -> by distance from the centre (0 = on it):
       0: away 98.0% sideways 0.0% toward 0.0% stay 2.0% (n=29528)
       1: away 91.1% sideways 0.0% toward 5.5% stay 3.4% (n=176251)
       2: away 64.2% sideways 0.0% toward 3.5% stay 32.3% (n=515281)

Q19. Does reaction change with 4 / 3 / 2 / 1 ticks to impact?
    -> by turns until impact:
       4: away 79.5% sideways 0.0% toward 4.0% stay 16.5% (n=371139)
       3: away 70.4% sideways 0.0% toward 4.0% stay 25.6% (n=233503)
       2: away 52.4% sideways 0.0% toward 3.0% stay 44.6% (n=116418)

Q20. When fleeing: straight away, sideways, toward, or stay?
    -> away 72.2% sideways 0.0% toward 3.8% stay 24.0% (n=721060)

Q21. How far from the centre does a fleeing dino end up?
    -> d=1: 0.0%, d=2: 0.0%, d=3: 41.8%, d=4: 37.6%, d=5: 20.6% | ran straight away 69.4%

Q22. How predictable is a dino 1-4 ticks ahead, by species?
    -> exact tile hit rate, predictor 'last move' (stay in brackets), mean error in tiles:
       S: 1t: 40.2% (15.5%) err 1.1 | 2t: 20.9% (22.8%) err 2.0 | 3t: 11.1% (8.5%) err 3.1 | 4t: 6.7% (12.4%) err 4.2
       V: 1t: 58.5% (49.8%) err 0.6 | 2t: 41.2% (42.3%) err 1.3 | 3t: 31.1% (34.0%) err 2.0 | 4t: 25.2% (29.3%) err 2.7
       T: 1t: 43.7% (26.0%) err 1.0 | 2t: 26.3% (34.0%) err 1.9 | 3t: 17.2% (17.7%) err 2.8 | 4t: 13.1% (22.2%) err 3.8
       R: 1t: 49.3% (48.0%) err 0.7 | 2t: 33.3% (41.9%) err 1.4 | 3t: 23.8% (30.2%) err 2.2 | 4t: 18.5% (26.4%) err 2.9

Q23. How much does movement history help prediction?
    -> exact hit rate 4 ticks ahead, all species:
       stay                 18.2%
       last move            12.1%
       majority of last 3   11.9%

Q24. How does the window edge affect movement per species?
    -> calm moves by screen row band:
       S top 0-2      : stay 32.0% up 7.3% down 28.7% left 16.2% right 15.8% (n=171617)
       S middle       : stay 11.9% up 9.4% down 27.8% left 25.5% right 25.3% (n=1699738)
       S bottom 17-19 : stay 12.4% up 28.4% down 12.9% left 24.4% right 21.8% (n=43900)
       V top 0-2      : stay 49.0% up 4.1% down 29.3% left 8.7% right 8.9% (n=121247)
       V middle       : stay 51.9% up 10.2% down 14.4% left 11.9% right 11.7% (n=512372)
       V bottom 17-19 : stay 39.7% up 11.0% down 16.8% left 16.6% right 15.8% (n=49443)
       T top 0-2      : stay 35.7% up 6.6% down 27.6% left 14.8% right 15.2% (n=101442)
       T middle       : stay 26.7% up 15.0% down 26.7% left 15.8% right 15.8% (n=424754)
       T bottom 17-19 : stay 18.7% up 16.5% down 24.9% left 19.6% right 20.3% (n=66229)
       R top 0-2      : stay 46.9% up 4.3% down 32.2% left 8.1% right 8.5% (n=67299)
       R middle       : stay 56.8% up 8.7% down 14.3% left 10.1% right 10.1% (n=239713)
       R bottom 17-19 : stay 52.4% up 2.3% down 18.7% left 13.0% right 13.6% (n=51953)

Q25. Which dinos die by scrolling, and can it be predicted?
    -> 10491 scroll deaths (S 4706 of 56006 deaths, V 2410 of 26819 deaths, T 2444 of 32182 deaths, R 931 of 7545 deaths); dinos on the top row just before a shift die by scroll 39.5%

saved -> results/analysis/local/movement.txt
```

### Species (Q26-Q36)

*Question:* Species behaviour (Triceratops high ground, raptors and corpses, T-Rex eating, lava) and the site's strategy claims.

*Script:* `analysis/species.py` (167 s)

```
species.py | 1054 games from logs/local_game_logs

Q26. Do dinos gather in the NE / near elevation 7+?
    -> NW 27.0% NE 27.1% SW 22.8% SE 23.0% | on elevation 7+: dinos 60.0% vs tiles 60.8%

Q27. Do Triceratops herds rotate counter-clockwise?
    -> counter-clockwise 50.0% vs clockwise 50.0% (n=738604; claim: 83%)

Q28. Does a "double-bounce" shockwave exist?
    -> unexplained deaths: 96, of which near a landing meteor: 18 (check these)

Q29. Do Triceratops move toward high ground?
    -> calm Triceratops step uphill 30.0% vs 29.0% for a random open move (lift x1.03)

Q30. Does Triceratops movement follow increasing elevation?
    -> mean elevation change per move +0.07 vs -0.10 for a random open move

Q31. How tight are Triceratops packs; is the centroid predictable 4 ticks ahead?
    -> mean distance to the pack centre 5.3 tiles; centroid error 4 ticks ahead: constant velocity 3.1 vs stay 1.8 tiles (n=256699)

Q32. Do raptors move toward corpses? Radius?
    -> calm raptors stepping closer to the nearest corpse, by distance:
       corpse 1-3   tiles: 24.5% vs random open move 26.7% (lift x0.92, n=261246)
       corpse 4-6   tiles: 37.2% vs random open move 36.0% (lift x1.03, n=184968)
       corpse 7-12  tiles: 39.5% vs random open move 38.9% (lift x1.01, n=208636)
       corpse 13+   tiles: 35.1% vs random open move 43.1% (lift x0.81, n=125118)

Q33. Is distance to the nearest corpse predictive of raptor moves?
    -> overall lift x0.95 (1.0 = no effect); see the distances above

Q34. Does corpse attraction stop when a raptor is threatened?
    -> threatened raptors step toward a corpse 33.0% vs random 34.1% (lift x0.97, n=105071)

Q35. When exactly does a T-Rex eat another dino?
    -> 4861 meals; prey: {'T': 2043, 'V': 911, 'S': 1907}; T-Rex distance the tick before: {1: 4861}; a dino next to a T-Rex is eaten the next tick 65.4%

Q36. How strongly / from how far do dinos avoid lava?
    -> calm dinos stepping closer to lava, by distance:
       lava 1 tiles away: 0.0% vs random open move 27.5% (lift x0.00, n=298973)
       lava 2 tiles away: 21.7% vs random open move 32.2% (lift x0.67, n=433820)
       lava 3 tiles away: 23.8% vs random open move 33.0% (lift x0.72, n=416512)
       lava 4 tiles away: 26.5% vs random open move 33.7% (lift x0.79, n=422398)

saved -> results/analysis/local/species.txt
```

### Physics (Q37-Q45)

*Question:* Meteor catch rates vs escape tiles and obstacles, escapes, follow-up gaps, lava funnels, scoring formula.

*Script:* `analysis/physics.py` (150 s)

```
physics.py | 1054 games from logs/local_game_logs

Q37. How do free escape tiles affect catch probability?
    -> 0: 97.2% (n=959) | 1-3: 60.3% (n=22477) | 4-7: 34.6% (n=93036) | 8-14: 13.0% (n=222983) | 15-99: 2.8% (n=56406)

Q38. Do other dinos block escapes enough to matter?
    -> catch rate by free escape tiles, with / without other dinos on them:
       exits 0    : occupied n/a (n=0) | free 97.2% (n=959)
       exits 1-3  : occupied 65.8% (n=1487) | free 59.9% (n=20990)
       exits 4-7  : occupied 36.4% (n=15499) | free 34.2% (n=77537)
       exits 8-14 : occupied 15.0% (n=80068) | free 11.9% (n=142915)
       exits 15-99: occupied 3.9% (n=37025) | free 0.5% (n=19381)

Q39. Do walls, mountains, lava, edges raise catch probability?
    -> lava: 29.3% (n=62097) | open ground: 3.0% (n=100127) | wall / mountain: 27.1% (n=218879) | window edge: 30.0% (n=101489)

Q40. When a dino is inside a falling blast, how often does it survive?
    -> escaped 69.9%, caught 16.9%, died another way 13.1% (n=455790)

Q41. Where does a dino go after escaping; where should a second meteor land?
    -> distance from the first centre at impact: d=1: 0.0%, d=2: 0.0%, d=3: 41.8%, d=4: 37.6%, d=5: 20.6% | a blast 3 tiles out along its axis covers it 69.0% vs one on its start tile 39.5%

Q42. Catch rate of a follow-up meteor by gap: 1, 2, 3 ticks?
    -> 1 tick later: 11.7% of escapees caught (n=139565) | 2 ticks later: 11.0% of escapees caught (n=99593) | 3 ticks later: 11.2% of escapees caught (n=101393)

Q43. Does lava block escape routes (funnel), and how much does it raise catch probability?
    -> catch rate with / without lava inside the dino's reach, same escape-tile band:
       exits 0    : lava 98.8% (n=736) | no lava 91.9% (n=223)
       exits 1-3  : lava 62.7% (n=10489) | no lava 58.3% (n=11988)
       exits 4-7  : lava 34.2% (n=32303) | no lava 34.8% (n=60733)
       exits 8-14 : lava 12.5% (n=47442) | no lava 13.1% (n=175541)
       exits 15-99: lava 3.6% (n=3545) | no lava 2.7% (n=52861)

Q44. Exact age-to-points formula?
    -> single meteor kills (n=77658): (score change - 0 per tick) / [160/(1+age/30)] median 1.0 (with age+1: 1.021); 1.000 = formula exact
       2 kills by one meteor: points / sum of single values = 1.499 (n=3669)
       3 kills by one meteor: points / sum of single values = 1.929 (n=207)

Q45. Exact multi-kill multiplier?
    -> 2 kills: x1.499 (rule 1 + 0.5 per extra kill = x1.5)

saved -> results/analysis/local/physics.txt
```
