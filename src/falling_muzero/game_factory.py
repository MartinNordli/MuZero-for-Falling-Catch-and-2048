from __future__ import annotations

from typing import TypeAlias

from falling_muzero.config import GameConfig
from falling_muzero.games.falling_catch import FallingCatchGame
from falling_muzero.games.game_2048 import Game2048


GameType: TypeAlias = FallingCatchGame | Game2048


def create_game(config: GameConfig) -> GameType:
    if config.kind == "catch":
        return FallingCatchGame(config)
    if config.kind == "2048":
        return Game2048(config)
    raise ValueError(f"unknown game kind {config.kind!r}; expected 'catch' or '2048'")
