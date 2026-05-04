"""Deterministic 4×4 2048 — the harder grid game extension.

This is a stripped-down 2048: the board is a fixed-size grid of *exponents*
(0 means empty, 1 means tile value 2, 2 means 4, …). Every move is one of
the four swipes. After a successful swipe, a new tile spawns at a position
chosen by a closed-form pseudo-random function of a counter, so the game
remains deterministic and the MuZero search tree stays a normal action tree
(no stochastic chance nodes are needed).

Observations include one-hot tile planes plus a small set of metadata planes
encoding step / spawn-count progress, which gives the learned dynamics model
visibility into the deterministic spawn schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from falling_muzero.config import GameConfig
from falling_muzero.games.types import StepResult


@dataclass(frozen=True, slots=True)
class Game2048State:
    """Frozen 2048 state. ``board`` stores tile exponents (so ``2`` means a 4 tile)."""

    board: tuple[tuple[int, ...], ...]
    step: int
    spawn_count: int
    score: int
    done: bool = False


class Game2048:
    """Deterministic 2048 environment with one-hot grid observations.

    Tiles are represented by exponents: 0 means empty, 1 means tile value 2,
    2 means 4, and so on. New tiles are deterministic so the existing MuZero
    tree can remain a normal action tree without stochastic chance nodes. The
    observation also includes small constant metadata planes for the deterministic
    spawn schedule so the learned dynamics model sees more of the true state.
    """

    ACTIONS: ClassVar[tuple[int, int, int, int]] = (0, 1, 2, 3)
    ACTION_NAMES: ClassVar[tuple[str, str, str, str]] = ("up", "down", "left", "right")
    METADATA_CHANNELS: ClassVar[int] = 6

    def __init__(self, config: GameConfig):
        if config.size < 2:
            raise ValueError("2048 size must be at least 2")
        if config.history_length < 1:
            raise ValueError("history_length must be positive")
        if config.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if config.max_tile_exponent < 3:
            raise ValueError("max_tile_exponent must be at least 3")
        self.config = config
        self._state = self.initial_state()

    @property
    def action_space_size(self) -> int:
        return len(self.ACTIONS)

    @property
    def tile_channel_count(self) -> int:
        return self.config.max_tile_exponent + 1

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.tile_channel_count + self.METADATA_CHANNELS, self.config.size, self.config.size)

    @property
    def stacked_observation_shape(self) -> tuple[int, int, int]:
        channels, height, width = self.observation_shape
        return (channels * self.config.history_length, height, width)

    @property
    def state(self) -> Game2048State:
        return self._state

    def reset(self) -> np.ndarray:
        self._state = self.initial_state()
        return self.observation(self._state)

    def initial_state(self) -> Game2048State:
        board = np.zeros((self.config.size, self.config.size), dtype=np.int16)
        spawn_count = 0
        board, spawn_count = self._spawn_tile(board, spawn_count)
        board, spawn_count = self._spawn_tile(board, spawn_count)
        return Game2048State(
            board=self._to_tuple(board),
            step=0,
            spawn_count=spawn_count,
            score=0,
            done=False,
        )

    def legal_actions(self, state: Game2048State | None = None) -> tuple[int, ...]:
        active_state = self._state if state is None else state
        if active_state.done:
            return ()
        board = np.asarray(active_state.board, dtype=np.int16)
        return tuple(action for action in self.ACTIONS if self._move_board(board, action)[1])

    def step(self, action: int) -> StepResult:
        result = self.transition(self._state, action)
        self._state = result.state
        return result

    def transition(self, state: Game2048State, action: int) -> StepResult:
        """Pure functional step. Illegal swipes incur ``invalid_move_penalty`` and don't spawn a tile."""

        if action not in self.ACTIONS:
            raise ValueError(f"invalid action index {action}")
        if state.done:
            return StepResult(state=state, observation=self.observation(state), reward=0.0, done=True)

        board = np.asarray(state.board, dtype=np.int16)
        moved_board, moved, merge_score = self._move_board(board, action)
        spawn_count = state.spawn_count
        score = state.score + merge_score

        if moved:
            moved_board, spawn_count = self._spawn_tile(moved_board, spawn_count)
            reward = merge_score / self.config.merge_reward_scale
        else:
            reward = self.config.invalid_move_penalty

        next_step = state.step + 1
        done = next_step >= self.config.episode_length or not self._has_legal_move(moved_board)
        next_state = Game2048State(
            board=self._to_tuple(moved_board),
            step=next_step,
            spawn_count=spawn_count,
            score=score,
            done=done,
        )
        return StepResult(
            state=next_state,
            observation=self.observation(next_state),
            reward=float(reward),
            done=done,
        )

    def coerce_action(self, action: int, policy: np.ndarray | None = None) -> int:
        """Return a legal move when possible.

        Human 2048 play normally ignores impossible swipes. For learning runs,
        repeatedly executing impossible swipes creates long penalty loops that
        teach little. This method maps an illegal chosen action to the legal
        action with the highest policy probability, or to the heuristic action
        if no policy is available.
        """

        legal = self.legal_actions()
        if not legal or action in legal:
            return action
        if policy is None:
            return self.heuristic_action()
        return max(legal, key=lambda legal_action: float(policy[legal_action]))

    def observation(self, state: Game2048State | None = None) -> np.ndarray:
        """One-hot tile planes (one per exponent up to ``max_tile_exponent``) + metadata planes."""

        active_state = self._state if state is None else state
        board = np.asarray(active_state.board, dtype=np.int16)
        channels, size, _ = self.observation_shape
        observation = np.zeros((channels, size, size), dtype=np.float32)
        tile_channels = self.tile_channel_count
        clipped = np.clip(board, 0, tile_channels - 1)
        for row in range(size):
            for col in range(size):
                observation[clipped[row, col], row, col] = 1.0
        observation[tile_channels:] = self._metadata_planes(active_state)
        return observation

    def stack_observations(self, observations: list[np.ndarray], index: int) -> np.ndarray:
        """Concatenate the last ``history_length`` frames along the channel axis (zero-padded if early)."""

        channels, height, width = self.observation_shape
        frames: list[np.ndarray] = []
        for offset in range(self.config.history_length - 1, -1, -1):
            source_index = index - offset
            if source_index < 0:
                frames.append(np.zeros((channels, height, width), dtype=np.float32))
            else:
                frames.append(observations[source_index].astype(np.float32, copy=False))
        return np.concatenate(frames, axis=0)

    def heuristic_action(self, state: Game2048State | None = None) -> int:
        """Strong non-learning baseline. Falls back to a 1-ply look-ahead with the snake heuristic.

        This heuristic is intentionally **not** used for the 2048 warm-start
        (see ``configs/2048.yaml``) so a learned actor cannot trivially imitate
        it — we keep it as an evaluation baseline only.
        """

        active_state = self._state if state is None else state
        legal = self.legal_actions(active_state)
        if not legal:
            return 0
        board = np.asarray(active_state.board, dtype=np.int16)
        depth = max(0, self.config.heuristic_search_depth)
        if depth > 0:
            return self._lookahead_heuristic_action(board, active_state.spawn_count, depth, legal)
        return self._static_heuristic_action(board, legal)

    def _static_heuristic_action(self, board: np.ndarray, legal: tuple[int, ...]) -> int:
        scored_actions = []
        for action in legal:
            moved_board, _, merge_score = self._move_board(board, action)
            empty = int(np.count_nonzero(moved_board == 0))
            max_tile = int(moved_board.max())
            corner_bonus = 1 if moved_board[0, 0] == max_tile or moved_board[-1, 0] == max_tile else 0
            monotonic_bonus = self._monotonicity_bonus(moved_board)
            score = merge_score + 4 * empty + 2 * max_tile + 3 * corner_bonus + monotonic_bonus
            scored_actions.append((score, -action, action))
        return max(scored_actions)[2]

    def _lookahead_heuristic_action(
        self,
        board: np.ndarray,
        spawn_count: int,
        depth: int,
        legal: tuple[int, ...],
    ) -> int:
        cache: dict[tuple[tuple[int, ...], int, int], float] = {}
        best_score = float("-inf")
        best_action = legal[0]
        for action in legal:
            moved_board, _, merge_score = self._move_board(board, action)
            spawned_board, next_spawn_count = self._spawn_tile(moved_board, spawn_count)
            score = merge_score + 0.99 * self._lookahead_value(
                spawned_board,
                next_spawn_count,
                depth - 1,
                cache,
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _lookahead_value(
        self,
        board: np.ndarray,
        spawn_count: int,
        depth: int,
        cache: dict[tuple[tuple[int, ...], int, int], float],
    ) -> float:
        key = (tuple(int(value) for value in board.reshape(-1)), spawn_count, depth)
        if key in cache:
            return cache[key]
        legal = tuple(action for action in self.ACTIONS if self._move_board(board, action)[1])
        if depth <= 0 or not legal:
            value = self._evaluate_board(board)
        else:
            value = float("-inf")
            for action in legal:
                moved_board, _, merge_score = self._move_board(board, action)
                spawned_board, next_spawn_count = self._spawn_tile(moved_board, spawn_count)
                value = max(
                    value,
                    merge_score + 0.99 * self._lookahead_value(
                        spawned_board,
                        next_spawn_count,
                        depth - 1,
                        cache,
                    ),
                )
        cache[key] = value
        return value

    def _evaluate_board(self, board: np.ndarray) -> float:
        """Hand-tuned static evaluation: snake weighting + empties + max tile + smoothness + corner."""

        # Standard 2048 heuristic combination: a weighted "snake" pattern that
        # rewards keeping the largest tiles in one row, plus terms for free
        # space, the maximum tile, smoothness between neighbouring tiles, and
        # a corner bonus that further pins the largest tile to ``[0, 0]``.
        values = np.where(board > 0, 2.0**board, 0.0)
        weights = np.asarray(
            [
                [15, 14, 13, 12],
                [8, 9, 10, 11],
                [7, 6, 5, 4],
                [0, 1, 2, 3],
            ],
            dtype=np.float64,
        )
        weights = 4.0**weights
        empty = int(np.count_nonzero(board == 0))
        max_tile = int(board.max())
        snake_score = float((values * weights).sum()) / 1e8
        smoothness = 0.0
        for row in range(self.config.size):
            for col in range(self.config.size):
                if board[row, col] == 0:
                    continue
                if row + 1 < self.config.size and board[row + 1, col] != 0:
                    smoothness -= abs(int(board[row, col]) - int(board[row + 1, col]))
                if col + 1 < self.config.size and board[row, col + 1] != 0:
                    smoothness -= abs(int(board[row, col]) - int(board[row, col + 1]))
        corner_bonus = 2000.0 if board[0, 0] == max_tile else 0.0
        return snake_score + 200.0 * empty + 100.0 * max_tile + 5.0 * smoothness + corner_bonus

    def render_ascii(self, state: Game2048State | None = None) -> str:
        """ASCII rendering used by tests and debug logs (tile values rendered, not exponents)."""

        active_state = self._state if state is None else state
        board = np.asarray(active_state.board, dtype=np.int16)
        rows = []
        for row in board:
            rows.append(" ".join("." if exponent == 0 else str(2**int(exponent)).rjust(4) for exponent in row))
        return "\n".join(rows)

    def board_from_observation(self, observation: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`observation`: recover the tile-exponent grid for visualisation."""

        frame_channels = self.observation_shape[0]
        tile_channels = self.tile_channel_count
        if observation.shape[0] >= frame_channels and observation.shape[0] % frame_channels == 0:
            current_frame = observation[-frame_channels:]
        else:
            current_frame = observation
        current = current_frame[:tile_channels]
        return current.argmax(axis=0).astype(np.int16)

    def _metadata_planes(self, state: Game2048State) -> np.ndarray:
        size = self.config.size
        planes = np.zeros((self.METADATA_CHANNELS, size, size), dtype=np.float32)
        max_spawn_count = max(1.0, float(self.config.episode_length + 2))
        max_cells = max(1, size * size)
        spawn_phase = (self.config.spawn_seed + state.spawn_count * 7 + state.spawn_count * state.spawn_count) % max_cells

        planes[0].fill(min(1.0, state.step / max(1.0, float(self.config.episode_length))))
        planes[1].fill(min(1.0, state.spawn_count / max_spawn_count))
        planes[2].fill(1.0 if (self.config.spawn_seed + state.spawn_count) % 10 == 0 else 0.0)
        planes[3].fill((state.spawn_count % 10) / 9.0)
        planes[4].fill((state.spawn_count % max_cells) / float(max_cells - 1) if max_cells > 1 else 0.0)
        planes[5].fill(spawn_phase / float(max_cells - 1) if max_cells > 1 else 0.0)
        return planes

    def _spawn_tile(self, board: np.ndarray, spawn_count: int) -> tuple[np.ndarray, int]:
        result = board.copy()
        empties = list(zip(*np.where(result == 0), strict=False))
        if not empties:
            return result, spawn_count
        index = (self.config.spawn_seed + spawn_count * 7 + spawn_count * spawn_count) % len(empties)
        row, col = empties[index]
        result[row, col] = 2 if (self.config.spawn_seed + spawn_count) % 10 == 0 else 1
        return result, spawn_count + 1

    def _move_board(self, board: np.ndarray, action: int) -> tuple[np.ndarray, bool, int]:
        """Apply a single swipe. Returns ``(moved_board, did_move, merge_score)``."""

        # All four swipes reduce to "merge each line, with optional reversal".
        # Up/down operate on the transposed board (so we move along columns
        # without rewriting the merge code).
        if action == 0:
            oriented = board.T
            lines = [oriented[col, :] for col in range(self.config.size)]
            reverse = False
        elif action == 1:
            oriented = board.T
            lines = [oriented[col, ::-1] for col in range(self.config.size)]
            reverse = True
        elif action == 2:
            lines = [board[row, :] for row in range(self.config.size)]
            reverse = False
        else:
            lines = [board[row, ::-1] for row in range(self.config.size)]
            reverse = True

        merged_lines = []
        merge_score = 0
        for line in lines:
            merged, line_score = self._merge_line(line)
            merge_score += line_score
            merged_lines.append(merged[::-1] if reverse else merged)

        if action in (0, 1):
            moved_board = np.stack(merged_lines, axis=0).T
        else:
            moved_board = np.stack(merged_lines, axis=0)
        return moved_board.astype(np.int16), bool(not np.array_equal(board, moved_board)), merge_score

    def _merge_line(self, line: np.ndarray) -> tuple[np.ndarray, int]:
        """Compress one row, merging adjacent equal tiles once each. Returns ``(new_line, merge_score)``."""

        nonzero = [int(value) for value in line if value != 0]
        output: list[int] = []
        score = 0
        index = 0
        while index < len(nonzero):
            if index + 1 < len(nonzero) and nonzero[index] == nonzero[index + 1]:
                merged = min(nonzero[index] + 1, self.config.max_tile_exponent)
                output.append(merged)
                score += 2**merged
                index += 2
            else:
                output.append(nonzero[index])
                index += 1
        output.extend([0] * (self.config.size - len(output)))
        return np.asarray(output, dtype=np.int16), score

    def _has_legal_move(self, board: np.ndarray) -> bool:
        return any(self._move_board(board, action)[1] for action in self.ACTIONS)

    def _monotonicity_bonus(self, board: np.ndarray) -> int:
        descending_rows = sum(int(np.all(row[:-1] >= row[1:])) for row in board)
        descending_cols = sum(int(np.all(col[:-1] >= col[1:])) for col in board.T)
        return descending_rows + descending_cols

    @staticmethod
    def _to_tuple(board: np.ndarray) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in board)
