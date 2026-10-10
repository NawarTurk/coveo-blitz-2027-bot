# Research questions - the game

Facts about the game itself (rules, spawns, dino behaviour, physics): true whichever bot plays, so they can
use ALL game logs. Each script answers one block, in order, printing `Q<n>. <question> -> <answer>`,
and saves to `results/analysis/<script>.txt`.

## 1. `analysis/waves.py` - wave rules (Q1-Q7)

1. When does a new wave arrive?
2. Does the wave timer reset after we clear the board?
3. How many dinos can be alive at once, and what happens to a wave at the cap?
4. At the 20 cap, which members of a wave are omitted?
5. Is the species mix of each wave fixed?
6. Does the mix repeat in a cycle? How long?
7. Same cycle locally and on the server?

## 2. `analysis/spawns.py` - where dinos spawn (Q8-Q15)

8. Can we predict the exact spawn tile?
9. Can we predict spawns with a 3x3 or 4x4 grid?
10. Do spawn positions differ by species?
11. Do spawn positions depend on the wave number?
12. Does a wave spawn where the previous wave spawned?
13. Do dinos of one wave spawn close together?
14. Do spawns favour certain terrain (elevation, mountains)?
15. Do spawns depend on occupied tiles, lava, corpses?

## 3. `analysis/movement.py` - how dinos move and react (Q16-Q25)

16. How does each species move with no threat?
17. How does each species react to a meteor launched near it?
18. Does reaction depend on distance from the blast centre?
19. Does reaction change with 4 / 3 / 2 / 1 ticks to impact?
20. When fleeing: straight away, sideways, toward, or stay?
21. How far from the centre does a fleeing dino end up?
22. How predictable is a dino 1-4 ticks ahead, by species?
23. How much does movement history help prediction?
24. How does the window edge affect movement per species?
25. Which dinos die by scrolling, and can it be predicted?

## 4. `analysis/species.py` - species-specific behaviour and the site's strategy claims (Q26-Q36)

26. Do dinos gather in the NE / near elevation 7+?
27. Do Triceratops herds rotate counter-clockwise?
28. Does a "double-bounce" shockwave exist?
29. Do Triceratops move toward high ground?
30. Does Triceratops movement follow increasing elevation?
31. How tight are Triceratops packs; is the centroid predictable 4 ticks ahead?
32. Do raptors move toward corpses? Radius?
33. Is distance to the nearest corpse predictive of raptor moves?
34. Does corpse attraction stop when a raptor is threatened?
35. When exactly does a T-Rex eat another dino?
36. How strongly / from how far do dinos avoid lava?

## 5. `analysis/physics.py` - meteors, lava and scoring rules (Q37-Q45)

37. How do free escape tiles affect catch probability?
38. Do other dinos block escapes enough to matter?
39. Do walls, mountains, lava, edges raise catch probability?
40. When a dino is inside a falling blast, how often does it survive?
41. Where does a dino go after escaping; where should a second meteor land?
42. Catch rate of a follow-up meteor by gap: 1, 2, 3 ticks?
43. Does lava block escape routes (funnel), and how much does it raise catch probability?
44. Exact age-to-points formula?
45. Exact multi-kill multiplier?

---

## Our bot's performance (Q46-Q74) - answer per bot, on that bot's own games (scripts to come)

Run on the games of the bot being judged (server team logs converted with `team_log_to_jsonl.py`,
or local benchmark runs), never on old-bot logs.

### `analysis/bot_waves.py` - how fast we clear (Q46-Q52)

46. How many ticks does each wave type take us?
47. How long are the last-1 / last-2 dino phases?
48. Which species survives us longest (last survivor vs its spawn share)?
49. How does a last-surviving Stegosaurus escape us?
50. How often does a wave reach the 100-tick timer?
51. Does a timed wave arriving on leftovers make clearing faster or harder?
52. Can we predict scroll deaths and avoid wasting a meteor on them?

### `analysis/bot_meteors.py` - our shots, chains and volcano (Q53-Q64)

53. Kills per meteor vs dinos alive (1-2 / 3-5 / 6+)?
54. Which species is easiest / hardest for us to catch?
55. How often is "hold, it's doomed" wrong?
56. Do our chained meteors beat independent ones (kills per shot, wave speed)?
57. All 3 slots busy vs reserving one for a follow-up?
58. How much score do we lose to T-Rex eating?
59. Would killing T-Rex early raise total score?
60. Does our volcano help later meteors even with few kills?
61. Which wave phase wastes most meteors: first 50%, middle, final 20%?
62. What decides our score: kills, kill value, waves, ticks per wave?
63. Score share by cause (meteor / lava / scroll)?
64. Which wave recipes give us the most points per tick?

### `analysis/bot_candidates.py` - candidate spots and CNN4 (Q65-Q74)

65. How often is the best kill spot missing from our candidate list?
66. How often is a dino's true future position covered by some candidate?
67. Bad shot: prediction problem or candidate problem?
68. Do future-map candidates (N8) beat normal cage candidates?
69. Which candidate sources produce the successful shots (cage, chain, follow-up, future-map)?
70. How many CNN4 candidates before returns flatten (8 / 16 / 24 / 32 / 40)?
71. Is more computation better spent on the last 1-2 dinos or early in a wave?
72. Biggest gain from a 25% cut in: last-dino time, zero-kill meteors, T-Rex eating, missed ticks?
73. What would 30k need?
74. How do the top teams reach ~160k?
