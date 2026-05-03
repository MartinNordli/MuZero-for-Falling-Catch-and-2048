from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from falling_muzero.config import GameConfig


@dataclass(frozen=True, slots=True)
class StepResult:
    state: object
    observation: np.ndarray
    reward: float
    done: bool


class MuZeroGame(Protocol):
    ACTION_NAMES: tuple[str, ...]
    config: GameConfig

    @property
    def action_space_size(self) -> int: ...

    @property
    def observation_shape(self) -> tuple[int, int, int]: ...

    @property
    def stacked_observation_shape(self) -> tuple[int, int, int]: ...

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> StepResult: ...

    def legal_actions(self, state: object | None = None) -> tuple[int, ...]: ...

    def heuristic_action(self, state: object | None = None) -> int: ...

    def coerce_action(self, action: int, policy: np.ndarray | None = None) -> int: ...

    def stack_observations(self, observations: list[np.ndarray], index: int) -> np.ndarray: ...

    def render_ascii(self, state: object | None = None) -> str: ...
