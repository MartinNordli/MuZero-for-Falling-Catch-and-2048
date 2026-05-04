"""Shared interface every game simulator must implement.

The trainer, MCTS, and replay buffer all program against ``MuZeroGame``; they
never import a concrete game directly. To add a third game, implement the
Protocol and register it in :mod:`falling_muzero.game_factory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from falling_muzero.config import GameConfig


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of a single environment step.

    ``state`` is the simulator's internal frozen state object (different per
    game), ``observation`` is the network-facing tensor for the *new* state,
    ``reward`` is the scalar reward incurred by the action, and ``done`` flags
    the end of an episode.
    """

    state: object
    observation: np.ndarray
    reward: float
    done: bool


class MuZeroGame(Protocol):
    """Common contract for environments wired into the MuZero pipeline."""

    ACTION_NAMES: tuple[str, ...]
    config: GameConfig

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions the agent can choose from."""
        ...

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape ``(C, H, W)`` of a single-frame observation tensor."""
        ...

    @property
    def stacked_observation_shape(self) -> tuple[int, int, int]:
        """Shape ``(C', H, W)`` of the history-stacked observation fed to the network."""
        ...

    def reset(self) -> np.ndarray:
        """Reset to the initial state and return the first stacked observation."""
        ...

    def step(self, action: int) -> StepResult:
        """Apply ``action`` and return the resulting :class:`StepResult`."""
        ...

    def legal_actions(self, state: object | None = None) -> tuple[int, ...]:
        """Return the indices of currently legal actions (defaults to all if ``None``)."""
        ...

    def heuristic_action(self, state: object | None = None) -> int:
        """Return the hand-written baseline action for ``state``."""
        ...

    def coerce_action(self, action: int, policy: np.ndarray | None = None) -> int:
        """Map a raw action to a legal one, optionally guided by a policy distribution."""
        ...

    def stack_observations(self, observations: list[np.ndarray], index: int) -> np.ndarray:
        """Build a history-stacked observation centred on ``observations[index]``."""
        ...

    def render_ascii(self, state: object | None = None) -> str:
        """Return a textual rendering of ``state`` (used for debugging and tests)."""
        ...
