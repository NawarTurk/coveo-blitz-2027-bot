# Spec hypotheses: game rules to test

Every behaviour the game docs claim, written as a testable hypothesis.
All tests run on local game logs (`logs/local_game_logs`, 1,151 games); confirm the important ones on server logs.

## Stegosaurus

- **H1** Stegosaurus moves in a slow, predictable drift.
  *Test:* stay / up / down / left / right rates when no meteor is near; how often it repeats its last move.
- **H2** Stegosaurus moves toward a shifting target point ahead of the rolling window.
  *Test:* its average move direction vs. its position in the window (rows from the top / bottom, columns from the sides).
- **H3** Stegosaurus avoids the trailing (top) edge of the window.
  *Test:* P(move down) as a function of distance to the top edge; compare with other species.
- **H4** Stegosaurus flees a meteor blast.
  *Test:* move direction relative to the blast centre in the 3 ticks before impact.

## Velociraptor

- **H5** Velociraptors are attracted to corpses.
  *Test:* change in distance to the nearest corpse per move vs. random.
- **H6** Velociraptors idle when safe.
  *Test:* stay rate with no meteor or T-Rex nearby.
- **H7** When safe, Velociraptors move toward the nearest corpse.
  *Test:* same as H5, only on ticks with no threat.
- **H8** Velociraptors dodge threats with the smallest possible move.
  *Test:* tiles moved when a meteor is falling nearby: is it the minimum needed to leave the blast?

## Triceratops

- **H9** Triceratops form packs.
  *Test:* distance to the nearest Triceratops vs. random placement.
- **H10** A pack moves together.
  *Test:* how well the pack centroid predicts each member's next position vs. its own velocity.
- **H11** Triceratops move toward favourable high ground.
  *Test:* elevation change per move vs. the average elevation change of the possible moves.

## T-Rex

- **H12** The T-Rex chooses the move that maximizes its survival.
  *Test:* free escape tiles after its chosen move vs. after the other possible moves.
- **H13** The T-Rex spreads away from other dinosaurs.
  *Test:* change in distance to the nearest non-T-Rex dino per move, when no prey is adjacent.
- **H14** The T-Rex hunts and eats other species.
  *Test:* P(prey dies next tick) by distance to the T-Rex.

## All dinosaurs

- **H15** Dinos know where a meteor lands as soon as it is launched.
  *Test:* flee rate on the first tick after launch.
- **H16** Dinos avoid lava.
  *Test:* how often a dino steps next to or onto lava vs. chance.
- **H17** Dinos know where the rolling-window edges are.
  *Test:* move direction vs. distance to each edge, per species.
- **H18** Two dinos cannot share a tile; on a collision one of them stays.
  *Test:* any two dinos on the same tile? Stay rate when the chosen tile is occupied.
- **H19** Impassable terrain and mountains block movement.
  *Test:* any dino ever on an impassable or mountain tile? Stay rate next to them.

## Rolling window

- **H20** The window moves toward increasing y.
- **H21** The window shifts every 5 turns.
- **H22** A dino outside the window dies immediately.

## Corpses

- **H23** A corpse stays at a fixed world position.
- **H24** Corpses do not block dinosaur or lava movement.
  *Test:* any dino stepping onto a corpse tile? Lava spreading over a corpse tile?
- **H25** Later meteor impacts do not affect a corpse.
  *Test:* corpses inside a later blast: still there the tick after?
- **H26** A corpse persists until it scrolls out of the window.
  *Test:* corpse lifetime vs. the tick its row leaves the window.

## Waves

- **H27** Each wave has a fixed species mix (10-wave recipe).
- **H28** The next wave comes 100 ticks after the previous one started, or 1 tick after a clear.
- **H29** At most 20 dinos are alive; extra spawns are dropped.
- **H30** The 10-wave cycle repeats.
