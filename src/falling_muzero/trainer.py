from __future__ import annotations

from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from falling_muzero.config import AppConfig, ensure_parent
from falling_muzero.game import FallingCatchGame
from falling_muzero.mcts import MuZeroMCTS, SearchResult
from falling_muzero.networks import MuZeroNetwork, policy_loss, scalar_loss
from falling_muzero.replay_buffer import Episode, EpisodeBuffer

PolicyMode = Literal["actor", "mcts", "random", "heuristic"]


class Trainer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cpu")
        self._set_seeds(config.training.seed)

        self.game = FallingCatchGame(config.game)
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
        )
        self.mcts = MuZeroMCTS(
            network=self.network,
            action_space_size=self.game.action_space_size,
            config=config.mcts,
            discount=config.training.discount,
            device=self.device,
            seed=config.training.seed,
        )
        self.metrics: dict[str, list[float | int | None]] = {
            "episode": [],
            "train_reward": [],
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
            "actor_eval_reward": [],
            "random_eval_reward": [],
            "heuristic_eval_reward": [],
            "best_actor_eval_reward": [],
        }
        self.best_actor_eval = float("-inf")

    def train(self) -> dict[str, list[float | int | None]]:
        self._bootstrap_replay_buffer()
        for episode_index in range(1, self.config.training.episodes + 1):
            episode, reward = self.run_episode(mode="mcts", training=True)
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

            actor_eval = random_eval = heuristic_eval = None
            if episode_index % self.config.training.eval_every == 0 or episode_index == self.config.training.episodes:
                actor_eval = self.evaluate("actor", self.config.training.eval_episodes)
                random_eval = self.evaluate("random", self.config.training.eval_episodes)
                heuristic_eval = self.evaluate("heuristic", self.config.training.eval_episodes)
                self._save_if_best(actor_eval)
                self.save_metrics(self.config.training.metrics_path)

            self._append_metrics(episode_index, reward, loss_info, actor_eval, random_eval, heuristic_eval)

        self.save_checkpoint(self.config.training.final_checkpoint_path)
        self.save_metrics(self.config.training.metrics_path)
        return self.metrics

    def run_episode(self, mode: PolicyMode = "actor", training: bool = False) -> tuple[Episode, float]:
        observation = self.game.reset()
        observations = [observation]
        episode = Episode()
        total_reward = 0.0

        while True:
            current_observation = observations[-1]
            observation_stack = self.game.stack_observations(observations, len(observations) - 1)
            action, policy, search_value = self._choose_action(mode, observation_stack, training)
            result = self.game.step(action)
            episode.append(
                observation=current_observation,
                action=action,
                reward=result.reward,
                policy=policy,
                search_value=search_value,
            )
            total_reward += result.reward
            observations.append(result.observation)
            if result.done:
                break

        episode.finalize_returns(self.config.training.discount)
        return episode, float(total_reward)

    def evaluate(self, mode: PolicyMode = "actor", episodes: int = 20) -> float:
        rewards = [self.run_episode(mode=mode, training=False)[1] for _ in range(episodes)]
        return float(np.mean(rewards))

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
        self.network.load_state_dict(checkpoint["model_state"])
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
    ) -> tuple[int, np.ndarray, float]:
        if mode == "random":
            policy = np.ones(self.game.action_space_size, dtype=np.float32) / self.game.action_space_size
            return int(np.random.choice(self.game.action_space_size, p=policy)), policy, 0.0
        if mode == "heuristic":
            action = self.game.heuristic_action()
            policy = np.zeros(self.game.action_space_size, dtype=np.float32)
            policy[action] = 1.0
            return action, policy, 0.0
        if mode == "mcts":
            result = self.mcts.search(observation_stack, add_exploration_noise=training)
            action = int(np.random.choice(self.game.action_space_size, p=result.policy)) if training else int(result.policy.argmax())
            return action, result.policy, result.value
        if mode == "actor":
            policy, value = self._actor_policy(observation_stack)
            return int(policy.argmax()), policy, value
        raise ValueError(f"unknown policy mode {mode}")

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
            self.config.training.bootstrap_gradient_steps > 0
            and self.buffer.can_sample(self.config.training.batch_size)
        ):
            self._run_fixed_gradient_steps(self.config.training.bootstrap_gradient_steps)
        actor_eval = self.evaluate("actor", self.config.training.eval_episodes)
        self._save_if_best(actor_eval)

    def _save_if_best(self, actor_eval: float | None) -> None:
        if actor_eval is None:
            return
        if actor_eval > self.best_actor_eval:
            self.best_actor_eval = actor_eval
            self.save_checkpoint(self.config.training.checkpoint_path)

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
        reward: float,
        loss_info: dict[str, float | None],
        actor_eval: float | None,
        random_eval: float | None,
        heuristic_eval: float | None,
    ) -> None:
        self.metrics["episode"].append(episode_index)
        self.metrics["train_reward"].append(float(reward))
        self.metrics["loss"].append(loss_info["loss"])
        self.metrics["policy_loss"].append(loss_info["policy_loss"])
        self.metrics["value_loss"].append(loss_info["value_loss"])
        self.metrics["reward_loss"].append(loss_info["reward_loss"])
        self.metrics["actor_eval_reward"].append(actor_eval)
        self.metrics["random_eval_reward"].append(random_eval)
        self.metrics["heuristic_eval_reward"].append(heuristic_eval)
        self.metrics["best_actor_eval_reward"].append(self.best_actor_eval if self.best_actor_eval > float("-inf") else None)

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
