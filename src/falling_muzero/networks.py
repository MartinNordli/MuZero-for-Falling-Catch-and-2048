from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from falling_muzero.config import NetworkConfig


@dataclass(slots=True)
class NetworkOutput:
    latent: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor | None = None


class RepresentationNetwork(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], config: NetworkConfig):
        super().__init__()
        channels, height, width = input_shape
        input_size = channels * height * width
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


class DynamicsNetwork(nn.Module):
    def __init__(self, action_space_size: int, config: NetworkConfig):
        super().__init__()
        self.action_space_size = action_space_size
        self.trunk = nn.Sequential(
            nn.Linear(config.latent_dim + action_space_size, config.hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Tanh(),
        )
        self.reward_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_one_hot = F.one_hot(actions.long(), num_classes=self.action_space_size).float()
        hidden = self.trunk(torch.cat([latent, action_one_hot], dim=-1))
        next_latent = self.state_head(hidden)
        reward = self.reward_head(hidden).squeeze(-1)
        return next_latent, reward


class PredictionNetwork(nn.Module):
    def __init__(self, action_space_size: int, config: NetworkConfig):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(config.hidden_dim, action_space_size)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(latent)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class MuZeroNetwork(nn.Module):
    """The assignment's Trinet: representation, dynamics, and prediction."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_space_size: int,
        config: NetworkConfig,
    ):
        super().__init__()
        self.representation = RepresentationNetwork(observation_shape, config)
        self.dynamics = DynamicsNetwork(action_space_size, config)
        self.prediction = PredictionNetwork(action_space_size, config)

    def initial_inference(self, observations: torch.Tensor) -> NetworkOutput:
        latent = self.representation(observations)
        policy_logits, value = self.prediction(latent)
        return NetworkOutput(latent=latent, policy_logits=policy_logits, value=value)

    def recurrent_inference(self, latent: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        next_latent, reward = self.dynamics(latent, actions)
        policy_logits, value = self.prediction(next_latent)
        return NetworkOutput(latent=next_latent, policy_logits=policy_logits, value=value, reward=reward)


def policy_loss(logits: torch.Tensor, target_policy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return _masked_mean(losses, mask)


def scalar_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(F.mse_loss(prediction, target, reduction="none"), mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    denominator = mask.sum().clamp_min(1.0)
    return masked.sum() / denominator
