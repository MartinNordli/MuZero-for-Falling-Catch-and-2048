from __future__ import annotations

from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from falling_muzero.config import AppConfig, ensure_parent
from falling_muzero.game_factory import create_game
from falling_muzero.mcts import MuZeroMCTS, SearchResult
from falling_muzero.networks import MuZeroNetwork, policy_loss, scalar_loss
from falling_muzero.replay_buffer import Episode, EpisodeBuffer

PolicyMode = Literal["actor", "mcts", "random", "heuristic"]
STAT_KEYS = (
    "reward",
    "score",
    "max_tile",
    "steps",
    "1024_reached",
    "catches",
    "misses",
    "miss_percentage",
    "heuristic_agreement_percentage",
)
STAT_PREFIXES = (
    "train",
    "actor_eval",
    "random_eval",
    "mcts_eval",
    "heuristic_eval",
    "best_actor_eval",
)


class Trainer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cpu")
        self._set_seeds(config.training.seed)

        self.game = create_game(config.game)
        self.network = MuZeroNetwork(
            observation_shape=self.game.stacked_observation_shape,
            action_space_size=self.game.action_space_size,
            config=config.network,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.buffer = EpisodeBuffer(
            capacity=config.training.buffer_size,
            discount=config.training.discount,
            seed=config.training.seed,
            priority_alpha=config.training.replay_priority_alpha,
            priority_epsilon=config.training.replay_priority_epsilon,
            search_value_target_weight=config.training.search_value_target_weight,
        )
        self.mcts = MuZeroMCTS(
            network=self.network,
            action_space_size=self.game.action_space_size,
            config=config.mcts,
            discount=config.training.discount,
            device=self.device,
            seed=config.training.seed,
        )
        self.metrics: dict[str, list[float | int | None]] = self._empty_metrics()
        self.best_actor_eval = float("-inf")
        self.best_actor_stats: dict[str, float | None] | None = None

    def train(self) -> dict[str, list[float | int | None]]:
        self._bootstrap_replay_buffer()
        for episode_index in range(1, self.config.training.episodes + 1):
            episode, reward = self.run_episode(mode="mcts", training=True)
            train_stats = self._episode_stats(episode, reward)
            self.buffer.add(episode)

            loss_info: dict[str, float | None] = {
                "loss": None,
                "policy_loss": None,
                "value_loss": None,
                "reward_loss": None,
            }
            if (
                episode_index >= self.config.training.train_after_episodes
                and episode_index % self.config.training.train_every == 0
                and self.buffer.can_sample(self.config.training.batch_size)
            ):
                loss_info = self._run_gradient_steps()

            actor_eval_stats = random_eval_stats = mcts_eval_stats = heuristic_eval_stats = None
            if episode_index % self.config.training.eval_every == 0 or episode_index == self.config.training.episodes:
                actor_eval_stats = self.evaluate_stats("actor", self.config.training.eval_episodes)
                random_eval_stats = self.evaluate_stats("random", self.config.training.eval_episodes)
                if self.config.training.mcts_eval_episodes > 0:
                    mcts_eval_stats = self.evaluate_stats("mcts", self.config.training.mcts_eval_episodes)
                heuristic_eval_stats = self.evaluate_stats("heuristic", self.config.training.eval_episodes)
                self._save_if_best(actor_eval_stats)

            self._append_metrics(
                episode_index,
                train_stats,
                loss_info,
                actor_eval_stats,
                random_eval_stats,
                mcts_eval_stats,
                heuristic_eval_stats,
            )
            if actor_eval_stats is not None:
                self.save_metrics(self.config.training.metrics_path)

        self.save_checkpoint(self.config.training.final_checkpoint_path)
        self.save_metrics(self.config.training.metrics_path)
        return self.metrics

    def run_episode(
        self,
        mode: PolicyMode = "actor",
        training: bool = False,
        include_terminal_observation: bool = False,
    ) -> tuple[Episode, float]:
        observation = self.game.reset()
        observations = [observation]
        episode = Episode()
        total_reward = 0.0

        while True:
            current_observation = observations[-1]
            observation_stack = self.game.stack_observations(observations, len(observations) - 1)
            heuristic_action = self.game.heuristic_action() if self.config.game.kind == "catch" else None
            action, policy, search_value, has_search_value = self._choose_action(mode, observation_stack, training)
            result = self.game.step(action)
            episode.append(
                observation=current_observation,
                action=action,
                reward=result.reward,
                policy=policy,
                search_value=search_value,
                has_search_value=has_search_value,
                heuristic_match=None if heuristic_action is None else action == heuristic_action,
            )
            total_reward += result.reward
            observations.append(result.observation)
            if result.done:
                if include_terminal_observation:
                    episode.observations.append(result.observation.astype(np.float32, copy=True))
                break

        episode.finalize_returns(self.config.training.discount)
        return episode, float(total_reward)

    def evaluate(self, mode: PolicyMode = "actor", episodes: int = 20) -> float:
        return self.evaluate_stats(mode, episodes)["reward"]

    def evaluate_stats(self, mode: PolicyMode = "actor", episodes: int = 20) -> dict[str, float | None]:
        stats = []
        for _ in range(episodes):
            episode, reward = self.run_episode(mode=mode, training=False)
            stats.append(self._episode_stats(episode, reward))
        return self._mean_stats(stats)

    def save_checkpoint(self, path: str | Path) -> Path:
        target = ensure_parent(path)
        torch.save(
            {
                "model_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "metrics": self.metrics,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.network.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint {path} is incompatible with the current game/network config. "
                "Retrain the model or pass a checkpoint created with the same observation shape."
            ) from exc
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "metrics" in checkpoint:
            self.metrics = checkpoint["metrics"]

    def save_metrics(self, path: str | Path) -> Path:
        target = ensure_parent(path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)
        return target

    def _choose_action(
        self,
        mode: PolicyMode,
        observation_stack: np.ndarray,
        training: bool,
    ) -> tuple[int, np.ndarray, float, bool]:
        if mode == "random":
            policy = self._legal_uniform_policy()
            action = int(np.random.choice(self.game.action_space_size, p=policy))
            action = self._coerce_action(action, policy)
            return action, policy, 0.0, False
        if mode == "heuristic":
            action = self.game.heuristic_action()
            policy = np.zeros(self.game.action_space_size, dtype=np.float32)
            policy[action] = 1.0
            return action, policy, 0.0, False
        if mode == "mcts":
            result = self.mcts.search(
                observation_stack,
                add_exploration_noise=training,
                legal_actions=self.game.legal_actions(),
            )
            policy = self._mask_policy_to_legal(result.policy)
            action = int(np.random.choice(self.game.action_space_size, p=policy)) if training else int(policy.argmax())
            action = self._coerce_action(action, policy)
            return action, policy, result.value, True
        if mode == "actor":
            policy, value = self._actor_policy(observation_stack)
            masked_policy = self._mask_policy_to_legal(policy)
            action = self._coerce_action(int(masked_policy.argmax()), masked_policy)
            return action, masked_policy, value, False
        raise ValueError(f"unknown policy mode {mode}")

    def _coerce_action(self, action: int, policy: np.ndarray | None = None) -> int:
        coerce = getattr(self.game, "coerce_action", None)
        if coerce is None:
            return action
        return int(coerce(action, policy))

    def _legal_uniform_policy(self) -> np.ndarray:
        legal_actions = self.game.legal_actions()
        policy = np.zeros(self.game.action_space_size, dtype=np.float32)
        if not legal_actions:
            policy[:] = 1.0 / self.game.action_space_size
            return policy
        for action in legal_actions:
            policy[action] = 1.0 / len(legal_actions)
        return policy

    def _mask_policy_to_legal(self, policy: np.ndarray) -> np.ndarray:
        legal_actions = self.game.legal_actions()
        if not legal_actions:
            return policy.astype(np.float32) / max(float(policy.sum()), 1e-8)
        masked = np.zeros_like(policy, dtype=np.float32)
        for action in legal_actions:
            masked[action] = max(float(policy[action]), 0.0)
        total = float(masked.sum())
        if total <= 1e-8:
            for action in legal_actions:
                masked[action] = 1.0 / len(legal_actions)
            return masked
        return masked / total

    def _actor_policy(self, observation_stack: np.ndarray) -> tuple[np.ndarray, float]:
        self.network.eval()
        with torch.no_grad():
            observation = torch.as_tensor(observation_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
            output = self.network.initial_inference(observation)
            policy = torch.softmax(output.policy_logits, dim=-1)[0].cpu().numpy().astype(np.float32)
            return policy / policy.sum(), float(output.value.item())

    def _run_gradient_steps(self) -> dict[str, float]:
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0}
        for _ in range(self.config.training.gradient_steps):
            loss_info = self._train_step()
            for key in totals:
                totals[key] += loss_info[key]
        return {key: value / self.config.training.gradient_steps for key, value in totals.items()}

    def _run_fixed_gradient_steps(self, steps: int) -> dict[str, float]:
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0}
        for _ in range(steps):
            loss_info = self._train_step()
            for key in totals:
                totals[key] += loss_info[key]
        denominator = max(1, steps)
        return {key: value / denominator for key, value in totals.items()}

    def _bootstrap_replay_buffer(self) -> None:
        episodes = self.config.training.bootstrap_episodes
        if episodes <= 0:
            return
        mode = self.config.training.bootstrap_policy
        if mode not in {"heuristic", "random"}:
            raise ValueError("bootstrap_policy must be 'heuristic' or 'random'")
        for _ in range(episodes):
            episode, _ = self.run_episode(mode=mode, training=False)
            self.buffer.add(episode)
        if (
            self.config.training.bootstrap_imitation_steps > 0
            and self.buffer.can_sample(self.config.training.batch_size)
        ):
            self._run_imitation_steps(self.config.training.bootstrap_imitation_steps)
        if (
            self.config.training.bootstrap_gradient_steps > 0
            and self.buffer.can_sample(self.config.training.batch_size)
        ):
            self._run_fixed_gradient_steps(self.config.training.bootstrap_gradient_steps)
        # Do not save a "best" checkpoint before self-play has run. Otherwise
        # heuristic bootstrap can be mistaken for a learned MuZero result.

    def _save_if_best(self, actor_stats: dict[str, float | None] | None) -> None:
        if actor_stats is None or actor_stats["reward"] is None:
            return
        if actor_stats["reward"] > self.best_actor_eval:
            self.best_actor_eval = actor_stats["reward"]
            self.best_actor_stats = dict(actor_stats)
            self.save_checkpoint(self.config.training.checkpoint_path)

    def _run_imitation_steps(self, steps: int) -> dict[str, float]:
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        for _ in range(steps):
            loss_info = self._imitation_step()
            for key in totals:
                totals[key] += loss_info[key]
        denominator = max(1, steps)
        return {key: value / denominator for key, value in totals.items()}

    def _imitation_step(self) -> dict[str, float]:
        self.network.train()
        observations: list[np.ndarray] = []
        target_policies: list[np.ndarray] = []
        target_values: list[float] = []
        episodes = self.buffer.episodes
        for _ in range(self.config.training.batch_size):
            episode = random.choice(episodes)
            index = random.randrange(len(episode))
            observations.append(self.game.stack_observations(episode.observations, index))
            target_policies.append(episode.policies[index])
            target_values.append(episode.returns[index])

        observation_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)
        policy_tensor = torch.as_tensor(np.stack(target_policies), dtype=torch.float32, device=self.device)
        value_tensor = torch.as_tensor(np.asarray(target_values), dtype=torch.float32, device=self.device)
        mask = torch.ones((len(observations),), dtype=torch.float32, device=self.device)

        output = self.network.initial_inference(observation_tensor)
        imitation_policy_loss = policy_loss(output.policy_logits, policy_tensor, mask)
        imitation_value_loss = scalar_loss(output.value, value_tensor, mask)
        loss = imitation_policy_loss + 0.1 * imitation_value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(imitation_policy_loss.item()),
            "value_loss": float(imitation_value_loss.item()),
        }

    def _train_step(self) -> dict[str, float]:
        self.network.train()
        batch = self.buffer.sample(
            batch_size=self.config.training.batch_size,
            unroll_steps=self.config.training.unroll_steps,
            game=self.game,
        )
        observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        target_rewards = torch.as_tensor(batch.target_rewards, dtype=torch.float32, device=self.device)
        target_policies = torch.as_tensor(batch.target_policies, dtype=torch.float32, device=self.device)
        target_values = torch.as_tensor(batch.target_values, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=self.device)

        initial = self.network.initial_inference(observations)
        latent = initial.latent
        total_policy_loss = policy_loss(initial.policy_logits, target_policies[:, 0], masks[:, 0])
        total_value_loss = scalar_loss(initial.value, target_values[:, 0], masks[:, 0])
        total_reward_loss = torch.zeros((), dtype=torch.float32, device=self.device)

        for step in range(self.config.training.unroll_steps):
            recurrent = self.network.recurrent_inference(latent, actions[:, step])
            latent = recurrent.latent
            reward_mask = masks[:, step]
            state_mask = masks[:, step + 1]
            total_reward_loss = total_reward_loss + scalar_loss(
                recurrent.reward,
                target_rewards[:, step],
                reward_mask,
            )
            total_policy_loss = total_policy_loss + policy_loss(
                recurrent.policy_logits,
                target_policies[:, step + 1],
                state_mask,
            )
            total_value_loss = total_value_loss + scalar_loss(
                recurrent.value,
                target_values[:, step + 1],
                state_mask,
            )

        scale = float(self.config.training.unroll_steps + 1)
        total_policy_loss = total_policy_loss / scale
        total_value_loss = total_value_loss / scale
        total_reward_loss = total_reward_loss / max(1.0, float(self.config.training.unroll_steps))
        loss = (
            self.config.training.policy_loss_weight * total_policy_loss
            + self.config.training.value_loss_weight * total_value_loss
            + self.config.training.reward_loss_weight * total_reward_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(total_policy_loss.item()),
            "value_loss": float(total_value_loss.item()),
            "reward_loss": float(total_reward_loss.item()),
        }

    def _append_metrics(
        self,
        episode_index: int,
        train_stats: dict[str, float | None],
        loss_info: dict[str, float | None],
        actor_eval_stats: dict[str, float | None] | None,
        random_eval_stats: dict[str, float | None] | None,
        mcts_eval_stats: dict[str, float | None] | None,
        heuristic_eval_stats: dict[str, float | None] | None,
    ) -> None:
        self.metrics["episode"].append(episode_index)
        self._append_stat_group("train", train_stats)
        self.metrics["loss"].append(loss_info["loss"])
        self.metrics["policy_loss"].append(loss_info["policy_loss"])
        self.metrics["value_loss"].append(loss_info["value_loss"])
        self.metrics["reward_loss"].append(loss_info["reward_loss"])
        self._append_stat_group("actor_eval", actor_eval_stats)
        self._append_stat_group("random_eval", random_eval_stats)
        self._append_stat_group("mcts_eval", mcts_eval_stats)
        self._append_stat_group("heuristic_eval", heuristic_eval_stats)
        self._append_stat_group("best_actor_eval", self.best_actor_stats)

    def _append_stat_group(self, prefix: str, stats: dict[str, float | None] | None) -> None:
        for key in STAT_KEYS:
            metric_key = f"{prefix}_{key}"
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []
            self.metrics[metric_key].append(None if stats is None else stats[key])

    def _episode_stats(self, episode: Episode, reward: float) -> dict[str, float | None]:
        state = self.game.state
        score = float(getattr(state, "score", 0.0)) if hasattr(state, "score") else None
        max_tile: float | None = None
        if hasattr(state, "board"):
            board = np.asarray(state.board)
            max_exponent = int(board.max()) if board.size else 0
            max_tile = float(2**max_exponent) if max_exponent > 0 else 0.0
        catches = float(getattr(state, "catches")) if hasattr(state, "catches") else None
        misses = float(getattr(state, "misses")) if hasattr(state, "misses") else None
        scored_objects = None if catches is None or misses is None else catches + misses
        miss_percentage = (
            None
            if scored_objects is None or scored_objects <= 0
            else float(100.0 * misses / scored_objects)
        )
        heuristic_agreement_percentage = (
            None
            if not episode.heuristic_matches
            else float(100.0 * np.mean(episode.heuristic_matches))
        )
        return {
            "reward": float(reward),
            "score": score,
            "max_tile": max_tile,
            "steps": float(len(episode.actions)),
            "1024_reached": None if max_tile is None else float(max_tile >= 1024),
            "catches": catches,
            "misses": misses,
            "miss_percentage": miss_percentage,
            "heuristic_agreement_percentage": heuristic_agreement_percentage,
        }

    def _mean_stats(self, stats: list[dict[str, float | None]]) -> dict[str, float | None]:
        means: dict[str, float | None] = {}
        for key in STAT_KEYS:
            values = [stat[key] for stat in stats if stat[key] is not None]
            means[key] = None if not values else float(np.mean(values))
        return means

    @staticmethod
    def _empty_metrics() -> dict[str, list[float | int | None]]:
        metrics: dict[str, list[float | int | None]] = {
            "episode": [],
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
        }
        for prefix in STAT_PREFIXES:
            for key in STAT_KEYS:
                metrics[f"{prefix}_{key}"] = []
        return metrics

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def load_trainer_with_checkpoint(config: AppConfig, checkpoint_path: str | Path | None = None) -> Trainer:
    trainer = Trainer(config)
    path = checkpoint_path or config.training.checkpoint_path
    if Path(path).exists():
        trainer.load_checkpoint(path)
    return trainer
