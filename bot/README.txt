BOT VERSIONS
============

Every version uploaded to the Coveo Blitz 2027 server, what it tried, and how it scored.
Scores are single server games; maps change every game, so one game can swing by 5k+.
The code in this folder is Cage B (our best).

CNN1 = predicts each dino's next move (1 tick ahead)
CNN4 = predicts where dinos will be when a meteor lands (4 ticks ahead)
prev7 = model variant that looks at the last 7 moves instead of 1


VER   BOT              IDEA                                              MODELS              SCORES                   AVG
----  ---------------  ------------------------------------------------  ------------------  -----------------------  ------
v2    early            first working bot                                 -                   6,741                     6,741
v3    early            broken build                                      -                   293 / 232                   263
v4    early            heuristic + ML experiments                        -                   6,199                     6,199
v5    early            heuristic + ML experiments                        -                   7,634                     7,634
v6    Fast-clear       new reward: young dinos + multi-kill + clear      CNN1 + heuristic    8,477                     8,477
v7    Fast-clear       same, with a fix                                  CNN1 + heuristic    8,470                     8,470
v8    Heuristic        same reward, no CNN                               none                6,716                     6,716
v9    Cage A           trap dinos against walls (trap table)             CNN1                13,904                   13,904
v10   Cage B           Cage A, but CNN4 scores the meteor spots          CNN1 + CNN4         16,293                   16,293
v11   Cage C           Cage B + volcano whenever idle                    CNN1 + CNN4         11,363                   11,363
v12   Cage B prev7     both CNNs replaced by prev7 versions              CNN1p7 + CNN4p7     15,617 / 15,847          15,732
v13   Cage B mix       CNN1 prev7 + plain CNN4                           CNN1p7 + CNN4       14,476                   14,476
v14   Cage B mix       plain CNN1 + CNN4 prev7                           CNN1 + CNN4p7       10,765                   10,765
v15   Cage B           plain models again (repeat test)                  CNN1 + CNN4         14,999 / 15,774 / 8,803  13,192
v16   Cage B *         BEST. Meant as Z1, but the zip held Cage B        CNN1 + CNN4         16,228 / 18,888 / 16,986 17,367
v17   Z1               Cage B + ~150 spots (empty tiles near dinos too)  CNN1 + CNN4         15,437 / 12,250          13,844
v18   Z2 / mix-up      Z1 without hold rule (zip mix-up, unclear)        CNN1 + CNN4         17,739 / 13,998 / 13,338 15,025
v19   Z3               Z1 + 5 fixes (smart hold, scroll rules, ...)      CNN1 + CNN4         17,904 / 10,564 / 11,378 13,282
v20   Z4               Z3 + safer timing, no next-wave reserve           CNN1 + CNN4         10,381 / 14,007          12,194
v21   ZH1              pure heuristic, species flee simulation           none                9,143                     9,143
v22   ZH1 (as ZH2)     ZH2 zip with old ZH1 file: CNN never ran          none                8,698 / 7,402             8,050
v24   ZH2              ZH1 + CNN4 second opinion (CNN barely ran)        CNN4                15,318 / 13,010          14,164
v25   ZH2.1            ZH2 with timing fix (CNN4 ran every tick)         CNN4                10,131                   10,131
v26   ZW               Z1 + Z4 timing guards                             CNN1 + CNN4         13,689 / 8,455           11,072
v27   Z1L              Z1 + end-of-game report                           CNN1 + CNN4         10,390 / 15,624          13,007

* v16 = 18,888 is our leaderboard score.


WHAT WE LEARNED
---------------

  - Trapping dinos against walls (Cage A) was the biggest jump: ~8k -> ~14k.
  - Letting CNN4 score meteor spots (Cage B) was the next: -> ~16-17k.
  - Aiming only at spots that cover a dino NOW (Cage B) beats also aiming at
    empty tiles dinos might walk into (Z1): ~17k vs ~13k.
  - Looking at 7 past moves (prev7) did not help.
  - Hand-written dino behaviour rules (ZH1) were much worse than the CNN.
  - Map luck is huge: always compare bots on the same seeds.


NEXT
----

  N0   Cage B + new CNN4 (trained on 1,151 games, 6 blocks)   -> testing now
