"""Compatibility imports for the deterministic 2048 game.

New code should import from ``falling_muzero.games.game_2048``.
"""

from falling_muzero.games.game_2048 import Game2048, Game2048State

__all__ = ["Game2048", "Game2048State"]
