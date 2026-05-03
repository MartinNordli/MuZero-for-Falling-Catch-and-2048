from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from falling_muzero.games.types import MuZeroGame


@dataclass(slots=True)
class Episode:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    search_values: list[float] = field(default_factory=list)
    search_value_masks: list[float] = field(default_factory=list)
    heuristic_matches: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        policy: np.ndarray,
        search_value: float,
        has_search_value: bool = False,
        heuristic_match: bool | None = None,
    ) -> None:
        self.observations.append(observation.astype(np.float32, copy=True))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.policies.append(policy.astype(np.float32, copy=True))
        self.search_values.append(float(search_value))
        self.search_value_masks.append(1.0 if has_search_value else 0.0)
        if heuristic_match is not None:
            self.heuristic_matches.append(1.0 if heuristic_match else 0.0)

    def __len__(self) -> int:
        return len(self.actions)

    def finalize_returns(self, discount: float) -> None:
        running = 0.0
        returns: list[float] = []
        for reward in reversed(self.rewards):
            running = reward + discount * running
            returns.append(float(running))
        self.returns = list(reversed(returns))


@dataclass(slots=True)
class TrainingBatch:
    observations: np.ndarray
    actions: np.ndarray
    target_rewards: np.ndarray
    target_policies: np.ndarray
    target_values: np.ndarray
    masks: np.ndarray


class EpisodeBuffer:
    def __init__(
        self,
        capacity: int,
        discount: float,
        seed: int = 0,
        priority_alpha: float = 0.0,
        priority_epsilon: float = 0.001,
        search_value_target_weight: float = 0.5,
    ):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if priority_alpha < 0:
            raise ValueError("priority_alpha must be non-negative")
        if priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")
        if not 0.0 <= search_value_target_weight <= 1.0:
            raise ValueError("search_value_target_weight must be in [0, 1]")
        self.capacity = capacity
        self.discount = discount
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon
        self.search_value_target_weight = search_value_target_weight
        self._episodes: list[Episode] = []
        self._priorities: list[float] = []
        self._rng = Random(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(self._episodes)

    def add(self, episode: Episode) -> None:
        if len(episode) == 0:
            return
        episode.finalize_returns(self.discount)
        self._episodes.append(episode)
        self._priorities.append(self._episode_priority(episode))
        if len(self._episodes) > self.capacity:
            self._episodes = self._episodes[-self.capacity :]
            self._priorities = self._priorities[-self.capacity :]

    def can_sample(self, batch_size: int) -> bool:
        return bool(self._episodes) and sum(len(episode) for episode in self._episodes) >= batch_size

    def sample(
        self,
        batch_size: int,
        unroll_steps: int,
        game: MuZeroGame,
    ) -> TrainingBatch:
        if not self.can_sample(batch_size):
            raise ValueError("not enough data in episode buffer")

        obs_batch: list[np.ndarray] = []
        action_batch: list[list[int]] = []
        reward_batch: list[list[float]] = []
        policy_batch: list[list[np.ndarray]] = []
        value_batch: list[list[float]] = []
        mask_batch: list[list[float]] = []

        action_space = game.action_space_size
        zero_policy = np.ones(action_space, dtype=np.float32) / action_space

        for _ in range(batch_size):
            episode = self._sample_episode()
            start = self._rng.randrange(len(episode))
            obs_batch.append(game.stack_observations(episode.observations, start))

            actions: list[int] = []
            rewards: list[float] = []
            policies: list[np.ndarray] = []
            values: list[float] = []
            masks: list[float] = []

            for offset in range(unroll_steps + 1):
                target_index = start + offset
                if target_index < len(episode):
                    policies.append(episode.policies[target_index])
                    values.append(self._target_value(episode, target_index))
                    masks.append(1.0)
                else:
                    policies.append(zero_policy)
                    values.append(0.0)
                    masks.append(0.0)

                if offset < unroll_steps:
                    if target_index < len(episode):
                        actions.append(episode.actions[target_index])
                        rewards.append(episode.rewards[target_index])
                    else:
                        actions.append(1)
                        rewards.append(0.0)

            action_batch.append(actions)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            value_batch.append(values)
            mask_batch.append(masks)

        return TrainingBatch(
            observations=np.stack(obs_batch).astype(np.float32),
            actions=np.asarray(action_batch, dtype=np.int64),
            target_rewards=np.asarray(reward_batch, dtype=np.float32),
            target_policies=np.asarray(policy_batch, dtype=np.float32),
            target_values=np.asarray(value_batch, dtype=np.float32),
            masks=np.asarray(mask_batch, dtype=np.float32),
        )

    def _sample_episode(self) -> Episode:
        if self.priority_alpha <= 0:
            return self._rng.choice(self._episodes)
        total = sum(self._priorities)
        if total <= 0:
            return self._rng.choice(self._episodes)
        return self._rng.choices(self._episodes, weights=self._priorities, k=1)[0]

    def _episode_priority(self, episode: Episode) -> float:
        if self.priority_alpha <= 0:
            return 1.0
        reward_sum = max(0.0, sum(episode.rewards))
        return float((reward_sum + self.priority_epsilon) ** self.priority_alpha)

    def _target_value(self, episode: Episode, index: int) -> float:
        discounted_return = float(episode.returns[index])
        if index >= len(episode.search_values) or index >= len(episode.search_value_masks):
            return discounted_return
        if episode.search_value_masks[index] <= 0.0:
            return discounted_return
        weight = self.search_value_target_weight
        search_value = float(episode.search_values[index])
        return float((1.0 - weight) * discounted_return + weight * search_value)
