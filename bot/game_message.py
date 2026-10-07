from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Optional


@dataclass(slots=True)
class WorldPosition:
    """A world position on the game map. Use x and y for world coordinates. To convert to tile array indices, use i = y - map.origin.y and j = x - map.origin.x."""

    x: int
    y: int


@dataclass(slots=True)
class Tile:
    """A single tile on the visible map."""

    position: WorldPosition
    """World position of this tile."""
    elevation: int
    """Elevation of this tile."""
    isImpassable: bool
    """Whether this tile blocks movement and lava flow (mountains, crevasses)."""
    isMountain: bool
    """Whether this tile is a mountain and therefore a valid volcano target."""
    hasLava: bool
    """Whether this tile is covered in lava."""


@dataclass(slots=True)
class Meteor:
    """A meteor in flight that has not yet hit its target."""

    target: WorldPosition
    """World position where the meteor will land."""
    turnsUntilImpact: int
    """Number of turns remaining before impact."""
    impactedTiles: list[WorldPosition]
    """World positions of all tiles in the meteor's impact area. Dinosaurs occupying these tiles when the meteor hits are killed."""


@dataclass(slots=True)
class Volcano:
    """An active volcano producing lava."""

    position: WorldPosition
    """World position of the erupting mountain tile."""


@dataclass(slots=True)
class Dinosaur:
    """A dinosaur on the map."""

    id: int
    """Unique identifier for this dinosaur. This identifier remains unchanged across turns."""
    position: WorldPosition
    """Current world position of the dinosaur."""
    age: int
    """Number of turns since this dinosaur spawned. Use this value with the decay curve to calculate the points awarded for eliminating it."""
    name: str
    """Name of this dinosaur. Dinosaurs sharing a name always behave the same way. See the documentation for details."""


@dataclass(slots=True)
class GameMap:
    """The visible game map. Only tiles within the rolling window are exposed. Access tiles using array indices [i][j]. To convert world coordinates to array indices, use i = y - origin.y and j = x - origin.x."""

    width: int
    """Width of the visible frame in tiles. The column index j ranges from 0 to width - 1."""
    height: int
    """Height of the visible frame in tiles. The row index i ranges from 0 to height - 1."""
    origin: WorldPosition
    """World position of the top-left corner of the visible frame. Use this position to convert between world coordinates (x, y) and array indices (i, j)."""
    tiles: list[list[Tile]]
    """Grid of tiles in the visible frame. Access tiles using tiles[i][j], where i is the row index (y-axis) and j is the column index (x-axis)."""


@dataclass(slots=True)
class Constants:
    """Game constants for this match. These values remain fixed throughout a match but may change between matches as the game is balanced."""

    maxTicks: int
    """Maximum number of turns in the match."""
    maxMeteors: int
    """Maximum number of meteors in flight simultaneously."""
    maxVolcanoes: int
    """Maximum number of active volcanoes simultaneously."""
    meteorDelay: int
    """Number of turns between meteor launch and impact."""
    meteorRadius: int
    """Radius of the meteor's impact area."""
    mapShiftInterval: int
    """Number of turns between successive shifts of the visible map frame. The frame shifts on every turn whose number is a positive multiple of this value."""
    waveCadence: int
    """Number of turns between successive waves of dinosaurs."""
    dinosPerWave: int
    """Number of dinosaurs in each wave before applying the limit on living dinosaurs."""
    activeDinoCap: int
    """Maximum number of dinosaurs alive at any time. Waves only spawn enough to fill remaining slots."""


@dataclass(slots=True)
class TeamGameState:
    """The full game state sent to the player each turn."""

    tick: int
    """Current turn number (starts at 1)."""
    currentTick: int
    """Same value as tick."""
    score: int
    """Current score."""
    lastTickErrors: list[str]
    """Errors from the previous turn, such as invalid actions."""
    constants: Constants
    """Game constants for this match."""
    map: GameMap
    """The visible map frame. Only tiles inside the rolling window are exposed."""
    dinosaurs: list[Dinosaur]
    """All dinosaurs currently alive within the visible frame."""
    meteors: list[Meteor]
    """All meteors currently in flight with targets within the visible frame."""
    volcanoes: list[Volcano]
    """All active volcanoes within the visible frame."""
    mountains: list[WorldPosition]
    """World positions of all mountain tiles within the visible frame."""
    corpses: list[WorldPosition]
    """World positions of dinosaur corpses within the visible frame. Each position identifies a tile where a dinosaur died."""


class Action:
    type: str


@dataclass
class LaunchMeteorAction(Action):
    """Launch a meteor at the target world position. The meteor hits after meteorDelay turns. At most maxMeteors meteors can be in flight at once."""

    target: WorldPosition
    type: str = "LAUNCH_METEOR"


@dataclass
class TriggerVolcanoAction(Action):
    """Trigger a volcanic eruption at the world position of a mountain tile. At most maxVolcanoes volcanoes can be active at once."""

    target: WorldPosition
    type: str = "TRIGGER_VOLCANO"
