"""Single dispatch point that turns a ``GameConfig.kind`` into a concrete simulator.

Keeping the dispatch in one place is what lets the trainer / MCTS / replay
buffer stay game-agnostic: they only depend on the shared ``MuZeroGame``
Protocol from :mod:`falling_muzero.games.types`.
"""

from __future__ import annotations

from typing import TypeAlias

from falling_muzero.config import GameConfig
from falling_muzero.games.falling_catch import FallingCatchGame
from falling_muzero.games.game_2048 import Game2048


GameType: TypeAlias = FallingCatchGame | Game2048


def create_game(config: GameConfig) -> GameType:
    """Return the simulator matching ``config.kind`` (``"catch"`` or ``"2048"``)."""

    if config.kind == "catch":
        return FallingCatchGame(config)
    if config.kind == "2048":
        return Game2048(config)
    raise ValueError(f"unknown game kind {config.kind!r}; expected 'catch' or '2048'")
