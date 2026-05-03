from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from falling_muzero.config import GameConfig
from falling_muzero.games.types import StepResult


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete simulator state for the deterministic Falling Catch game."""

    ball_row: int
    ball_col: int
    paddle_left: int
    step: int
    spawn_count: int
    catches: int = 0
    misses: int = 0
    done: bool = False


class FallingCatchGame:
    """Small grid arcade game used as the real environment for MuZero.

    The player controls a horizontal paddle at the bottom of the grid. A single
    object falls one row per timestep. The action is applied first, then the
    object either falls or is scored if it is already on the bottom row.
    """

    ACTIONS: ClassVar[tuple[int, int, int]] = (-1, 0, 1)
    ACTION_NAMES: ClassVar[tuple[str, str, str]] = ("left", "stay", "right")

    def __init__(self, config: GameConfig):
        if config.width < 3:
            raise ValueError("width must be at least 3")
        if config.height < 3:
            raise ValueError("height must be at least 3")
        if not 1 <= config.paddle_width <= config.width:
            raise ValueError("paddle_width must be in [1, width]")
        if config.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if config.history_length < 1:
            raise ValueError("history_length must be positive")
        self.config = config
        self._state = self.initial_state()

    @property
    def action_space_size(self) -> int:
        return len(self.ACTIONS)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (2, self.config.height, self.config.width)

    @property
    def stacked_observation_shape(self) -> tuple[int, int, int]:
        channels, height, width = self.observation_shape
        return (channels * self.config.history_length, height, width)

    @property
    def state(self) -> GameState:
        return self._state

    def reset(self) -> np.ndarray:
        self._state = self.initial_state()
        return self.observation(self._state)

    def initial_state(self) -> GameState:
        return GameState(
            ball_row=0,
            ball_col=self.spawn_column(0),
            paddle_left=(self.config.width - self.config.paddle_width) // 2,
            step=0,
            spawn_count=0,
            catches=0,
            misses=0,
            done=False,
        )

    def legal_actions(self, state: GameState | None = None) -> tuple[int, ...]:
        active_state = self._state if state is None else state
        if active_state.done:
            return ()
        return tuple(range(self.action_space_size))

    def spawn_column(self, spawn_count: int) -> int:
        """Deterministic pseudo-random-looking sequence with no hidden RNG."""

        width = self.config.width
        return (self.config.spawn_seed + spawn_count * 3 + spawn_count * spawn_count) % width

    def step(self, action: int) -> StepResult:
        result = self.transition(self._state, action)
        self._state = result.state
        return result

    def transition(self, state: GameState, action: int) -> StepResult:
        if action not in range(self.action_space_size):
            raise ValueError(f"invalid action index {action}")
        if state.done:
            return StepResult(state=state, observation=self.observation(state), reward=0.0, done=True)

        move = self.ACTIONS[action]
        max_paddle_left = self.config.width - self.config.paddle_width
        paddle_left = int(np.clip(state.paddle_left + move, 0, max_paddle_left))
        reward = self._distance_shaping(state.ball_col, paddle_left)

        spawn_count = state.spawn_count
        catches = state.catches
        misses = state.misses
        if state.ball_row == self.config.height - 1:
            caught = paddle_left <= state.ball_col < paddle_left + self.config.paddle_width
            reward += self.config.catch_reward if caught else self.config.miss_reward
            if caught:
                catches += 1
            else:
                misses += 1
            spawn_count += 1
            next_ball_row = 0
            next_ball_col = self.spawn_column(spawn_count)
        else:
            next_ball_row = state.ball_row + 1
            next_ball_col = state.ball_col

        next_step = state.step + 1
        done = next_step >= self.config.episode_length
        next_state = GameState(
            ball_row=next_ball_row,
            ball_col=next_ball_col,
            paddle_left=paddle_left,
            step=next_step,
            spawn_count=spawn_count,
            catches=catches,
            misses=misses,
            done=done,
        )
        return StepResult(
            state=next_state,
            observation=self.observation(next_state),
            reward=float(reward),
            done=done,
        )

    def observation(self, state: GameState | None = None) -> np.ndarray:
        active_state = self._state if state is None else state
        grid = np.zeros(self.observation_shape, dtype=np.float32)
        if not active_state.done:
            grid[0, active_state.ball_row, active_state.ball_col] = 1.0
        row = self.config.height - 1
        for col in range(active_state.paddle_left, active_state.paddle_left + self.config.paddle_width):
            grid[1, row, col] = 1.0
        return grid

    def stack_observations(self, observations: list[np.ndarray], index: int) -> np.ndarray:
        """Return a fixed-length history stack ending at ``index``.

        Missing early history is padded with blank frames, which mirrors the
        assignment pseudocode's q-lookback behavior.
        """

        channels, height, width = self.observation_shape
        frames: list[np.ndarray] = []
        for offset in range(self.config.history_length - 1, -1, -1):
            source_index = index - offset
            if source_index < 0:
                frames.append(np.zeros((channels, height, width), dtype=np.float32))
            else:
                frames.append(observations[source_index].astype(np.float32, copy=False))
        return np.concatenate(frames, axis=0)

    def heuristic_action(self, state: GameState | None = None) -> int:
        """A simple non-learning baseline that moves the paddle toward the ball."""

        active_state = self._state if state is None else state
        paddle_center = active_state.paddle_left + (self.config.paddle_width - 1) / 2.0
        if active_state.ball_col < paddle_center:
            return 0
        if active_state.ball_col > paddle_center:
            return 2
        return 1

    def render_ascii(self, state: GameState | None = None) -> str:
        active_state = self._state if state is None else state
        rows: list[str] = []
        for row in range(self.config.height):
            cells = []
            for col in range(self.config.width):
                ball = (row, col) == (active_state.ball_row, active_state.ball_col) and not active_state.done
                paddle = (
                    row == self.config.height - 1
                    and active_state.paddle_left <= col < active_state.paddle_left + self.config.paddle_width
                )
                cells.append("X" if ball and paddle else "o" if ball else "=" if paddle else ".")
            rows.append("".join(cells))
        return "\n".join(rows)

    def _distance_shaping(self, ball_col: int, paddle_left: int) -> float:
        if self.config.distance_shaping == 0:
            return 0.0
        paddle_center = paddle_left + (self.config.paddle_width - 1) / 2.0
        max_distance = max(1.0, self.config.width - 1)
        normalized_distance = abs(ball_col - paddle_center) / max_distance
        return float(self.config.distance_shaping * (1.0 - 2.0 * normalized_distance))
