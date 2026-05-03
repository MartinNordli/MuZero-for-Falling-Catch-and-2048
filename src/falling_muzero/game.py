"""Compatibility imports for the Falling Catch game.

New code should import from ``falling_muzero.games.falling_catch``.
"""

from falling_muzero.games.falling_catch import FallingCatchGame, GameState
from falling_muzero.games.types import StepResult

__all__ = ["FallingCatchGame", "GameState", "StepResult"]
