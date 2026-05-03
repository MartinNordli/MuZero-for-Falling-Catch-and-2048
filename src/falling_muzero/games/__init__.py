"""Game environments supported by the shared MuZero trainer."""

from falling_muzero.games.falling_catch import FallingCatchGame, GameState
from falling_muzero.games.game_2048 import Game2048, Game2048State
from falling_muzero.games.types import MuZeroGame, StepResult

__all__ = [
    "FallingCatchGame",
    "Game2048",
    "Game2048State",
    "GameState",
    "MuZeroGame",
    "StepResult",
]
