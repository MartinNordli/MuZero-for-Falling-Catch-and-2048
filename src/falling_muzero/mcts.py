"""u-MCTS over learned latent states.

This is the assignment's "u-MCTS" — Monte-Carlo Tree Search that never touches
the real simulator after the root. The root latent comes from the
representation network on the actual observation, and every descent into the
tree is driven by the dynamics + prediction networks. Selection uses the
PUCT formula from MuZero/AlphaZero with Dirichlet noise mixed into the root
priors for exploration.

The output of one ``search`` call is a *visit-count-derived policy* over the
real action space, which the trainer either samples (during self-play) or
takes the argmax of (during evaluation).
"""

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
    """One MCTS node, holding the latent it was reached at and its running statistics."""

    prior: float
    latent: torch.Tensor
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        """``True`` once at least one child has been added by :meth:`MuZeroMCTS._expand`."""

        return bool(self.children)

    @property
    def value(self) -> float:
        """Mean backed-up value (zero before the first backpropagation)."""

        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class SearchResult:
    """Output of one ``MuZeroMCTS.search`` call: improved policy, root value, and root node."""

    policy: np.ndarray
    value: float
    root: Node


class MuZeroMCTS:
    """Latent-space MCTS that drives self-play and (optionally) evaluation."""

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

    def search(
        self,
        observation_stack: np.ndarray,
        add_exploration_noise: bool = True,
        legal_actions: tuple[int, ...] | None = None,
    ) -> SearchResult:
        """Run ``simulations`` MCTS rollouts from the given observation stack.

        Returns a :class:`SearchResult` whose ``policy`` is a normalised
        distribution derived from root visit counts (with optional temperature
        sharpening). When ``add_exploration_noise`` is ``True`` (default at
        training time) Dirichlet noise is mixed into the root priors. The
        optional ``legal_actions`` mask is honoured when expanding the root so
        illegal actions never receive search budget.
        """

        self.network.eval()
        with torch.no_grad():
            observation = torch.as_tensor(observation_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
            initial = self.network.initial_inference(observation)
            root = Node(prior=1.0, latent=initial.latent[0].detach(), reward=0.0)
            self._expand(
                root,
                initial.policy_logits[0],
                add_exploration_noise=add_exploration_noise,
                actions=legal_actions,
            )
            root.value_sum = float(initial.value.item())
            root.visit_count = 1

            for _ in range(self.config.simulations):
                if not root.children:
                    break
                node = root
                search_path = [node]
                depth = 0

                # Descend: PUCT-greedy until we hit an unexpanded leaf or the depth bound.
                while node.expanded and depth < self.config.max_depth:
                    _, node = self._select_child(node)
                    search_path.append(node)
                    depth += 1

                # Evaluate: prediction network on the leaf's latent.
                output = self.network.prediction(node.latent.unsqueeze(0))
                policy_logits, value = output[0].squeeze(0), output[1].squeeze(0)
                if depth < self.config.max_depth:
                    self._expand(node, policy_logits, add_exploration_noise=False)
                # Backup: discount predicted rewards along the path back to the root.
                self._backpropagate(search_path, float(value.item()))

        # Convert root visit counts into a search-improved policy. Temperature
        # near zero collapses to argmax (used at evaluation); larger values keep
        # the distribution stochastic, which is what feeds into the BPTT policy
        # target during training.
        visits = np.array(
            [root.children[action].visit_count if action in root.children else 0 for action in range(self.action_space_size)],
            dtype=np.float32,
        )
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

    def _expand(
        self,
        node: Node,
        policy_logits: torch.Tensor,
        add_exploration_noise: bool,
        actions: tuple[int, ...] | None = None,
    ) -> None:
        """Create one child per (legal) action, calling the dynamics network for each."""

        active_actions = tuple(range(self.action_space_size)) if actions is None else tuple(actions)
        if not active_actions:
            return

        priors = F.softmax(policy_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        masked_priors = np.zeros_like(priors)
        for action in active_actions:
            masked_priors[action] = priors[action]
        if masked_priors.sum() <= 1e-12:
            # The prediction net put zero mass on every legal action; fall back
            # to uniform so the search can still make progress.
            for action in active_actions:
                masked_priors[action] = 1.0
        priors = masked_priors / masked_priors.sum()

        if add_exploration_noise:
            # Standard MuZero/AlphaZero root-noise trick: blend in Dirichlet
            # noise so the search occasionally tries low-prior actions.
            noise_values = self.rng.dirichlet([self.config.dirichlet_alpha] * len(active_actions))
            noise = np.zeros_like(priors)
            for action, value in zip(active_actions, noise_values, strict=False):
                noise[action] = value
            priors = (1.0 - self.config.exploration_fraction) * priors + self.config.exploration_fraction * noise
            priors = priors / priors.sum()

        for action in active_actions:
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            recurrent = self.network.recurrent_inference(node.latent.unsqueeze(0), action_tensor)
            node.children[action] = Node(
                prior=float(priors[action]),
                latent=recurrent.latent[0].detach(),
                reward=float(recurrent.reward.item()) if recurrent.reward is not None else 0.0,
            )

    def _select_child(self, node: Node) -> tuple[int, Node]:
        """Pick the child with the largest PUCT score."""

        _, action, child = max(
            (self._ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """PUCT score = Q(s,a) + c(N) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))."""

        # The ``pb_c`` exploration coefficient grows slowly with the parent's
        # visit count (``pb_c_base`` controls how slowly, ``pb_c_init`` is the
        # constant baseline). This is the standard MuZero PUCT scaling.
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        # Q(s,a) folds in the predicted reward for the transition itself plus
        # the discounted child value backed up from the subtree.
        q_value = child.reward + self.discount * child.value
        return q_value + pb_c * child.prior

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        """Propagate ``value`` up the path, accumulating discounted rewards along the way."""

        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Move one step up in the tree: the parent's bootstrap value is the
            # current node's reward plus the discounted child value.
            value = node.reward + self.discount * value
