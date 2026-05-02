import numpy as np
import torch

from falling_muzero.config import GameConfig, MCTSConfig, NetworkConfig
from falling_muzero.game import FallingCatchGame
from falling_muzero.mcts import MuZeroMCTS
from falling_muzero.networks import MuZeroNetwork


def test_network_forward_shapes():
    game = FallingCatchGame(GameConfig(width=4, height=4, history_length=2))
    network = MuZeroNetwork(
        observation_shape=game.stacked_observation_shape,
        action_space_size=game.action_space_size,
        config=NetworkConfig(latent_dim=16, hidden_dim=32),
    )
    observations = torch.zeros((3, *game.stacked_observation_shape), dtype=torch.float32)
    output = network.initial_inference(observations)

    assert output.latent.shape == (3, 16)
    assert output.policy_logits.shape == (3, 3)
    assert output.value.shape == (3,)

    recurrent = network.recurrent_inference(output.latent, torch.tensor([0, 1, 2]))
    assert recurrent.latent.shape == (3, 16)
    assert recurrent.reward.shape == (3,)


def test_mcts_returns_valid_policy_distribution():
    game = FallingCatchGame(GameConfig(width=4, height=4, history_length=2))
    network = MuZeroNetwork(
        observation_shape=game.stacked_observation_shape,
        action_space_size=game.action_space_size,
        config=NetworkConfig(latent_dim=16, hidden_dim=32),
    )
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=game.action_space_size,
        config=MCTSConfig(simulations=3, max_depth=2),
        discount=0.95,
        device=torch.device("cpu"),
        seed=3,
    )
    obs = game.reset()
    stack = game.stack_observations([obs], 0)
    result = mcts.search(stack, add_exploration_noise=False)

    assert result.policy.shape == (3,)
    assert np.isclose(result.policy.sum(), 1.0)
    assert np.all(result.policy >= 0.0)
