from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch.nn import functional as F

from falling_muzero.config import MCTSConfig
from falling_muzero.networks import MuZeroNetwork


@dataclass(slots=True)
class Node:
    prior: float
    latent: torch.Tensor
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class SearchResult:
    policy: np.ndarray
    value: float
    root: Node


class MuZeroMCTS:
    """u-MCTS over learned abstract states, not over real game states."""

    def __init__(
        self,
        network: MuZeroNetwork,
        action_space_size: int,
        config: MCTSConfig,
        discount: float,
        device: torch.device,
        seed: int = 0,
    ):
        self.network = network
        self.action_space_size = action_space_size
        self.config = config
        self.discount = discount
        self.device = device
        self.rng = np.random.default_rng(seed)

    def search(self, observation_stack: np.ndarray, add_exploration_noise: bool = True) -> SearchResult:
        self.network.eval()
        with torch.no_grad():
            observation = torch.as_tensor(observation_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
            initial = self.network.initial_inference(observation)
            root = Node(prior=1.0, latent=initial.latent[0].detach(), reward=0.0)
            self._expand(root, initial.policy_logits[0], add_exploration_noise=add_exploration_noise)
            root.value_sum = float(initial.value.item())
            root.visit_count = 1

            for _ in range(self.config.simulations):
                node = root
                search_path = [node]
                depth = 0

                while node.expanded and depth < self.config.max_depth:
                    _, node = self._select_child(node)
                    search_path.append(node)
                    depth += 1

                output = self.network.prediction(node.latent.unsqueeze(0))
                policy_logits, value = output[0].squeeze(0), output[1].squeeze(0)
                if depth < self.config.max_depth:
                    self._expand(node, policy_logits, add_exploration_noise=False)
                self._backpropagate(search_path, float(value.item()))

        visits = np.array([root.children[action].visit_count for action in range(self.action_space_size)], dtype=np.float32)
        if visits.sum() == 0:
            policy = np.ones(self.action_space_size, dtype=np.float32) / self.action_space_size
        else:
            if self.config.temperature <= 1e-6:
                policy = np.zeros(self.action_space_size, dtype=np.float32)
                policy[int(visits.argmax())] = 1.0
            else:
                adjusted = visits ** (1.0 / self.config.temperature)
                policy = adjusted / adjusted.sum()
        return SearchResult(policy=policy.astype(np.float32), value=root.value, root=root)

    def _expand(self, node: Node, policy_logits: torch.Tensor, add_exploration_noise: bool) -> None:
        priors = F.softmax(policy_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        if add_exploration_noise:
            noise = self.rng.dirichlet([self.config.dirichlet_alpha] * self.action_space_size)
            priors = (1.0 - self.config.exploration_fraction) * priors + self.config.exploration_fraction * noise
        priors = priors / priors.sum()

        for action in range(self.action_space_size):
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            recurrent = self.network.recurrent_inference(node.latent.unsqueeze(0), action_tensor)
            node.children[action] = Node(
                prior=float(priors[action]),
                latent=recurrent.latent[0].detach(),
                reward=float(recurrent.reward.item()) if recurrent.reward is not None else 0.0,
            )

    def _select_child(self, node: Node) -> tuple[int, Node]:
        _, action, child = max(
            (self._ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def _ucb_score(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return child.value + pb_c * child.prior

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.discount * value
