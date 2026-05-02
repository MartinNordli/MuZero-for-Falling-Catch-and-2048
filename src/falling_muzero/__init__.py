"""Compact MuZero-style implementation for the Falling Catch grid game."""

from falling_muzero.config import AppConfig, load_config
from falling_muzero.game import FallingCatchGame, GameState

__all__ = ["AppConfig", "FallingCatchGame", "GameState", "load_config"]
