import numpy as np
import torch

from falling_muzero.config import GameConfig, MCTSConfig, NetworkConfig
from falling_muzero.games.falling_catch import FallingCatchGame
from falling_muzero.games.game_2048 import Game2048
from falling_muzero.mcts import MuZeroMCTS, Node
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


def test_conv_network_forward_shapes_for_2048():
    game = Game2048(GameConfig(kind="2048", size=4, history_length=2, max_tile_exponent=8))
    network = MuZeroNetwork(
        observation_shape=game.stacked_observation_shape,
        action_space_size=game.action_space_size,
        config=NetworkConfig(architecture="conv", latent_dim=16, hidden_dim=32),
    )
    observations = torch.zeros((2, *game.stacked_observation_shape), dtype=torch.float32)
    output = network.initial_inference(observations)

    assert output.latent.shape == (2, 16)
    assert output.policy_logits.shape == (2, 4)
    assert output.value.shape == (2,)


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


def test_mcts_ucb_includes_predicted_reward():
    game = FallingCatchGame(GameConfig(width=4, height=4, history_length=2))
    network = MuZeroNetwork(
        observation_shape=game.stacked_observation_shape,
        action_space_size=game.action_space_size,
        config=NetworkConfig(latent_dim=16, hidden_dim=32),
    )
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=game.action_space_size,
        config=MCTSConfig(pb_c_base=19652, pb_c_init=0.0),
        discount=0.5,
        device=torch.device("cpu"),
        seed=3,
    )
    latent = torch.zeros(16)
    parent = Node(prior=1.0, latent=latent, visit_count=10)
    child = Node(prior=0.0, latent=latent, reward=2.0, visit_count=1, value_sum=6.0)

    assert np.isclose(mcts._ucb_score(parent, child), 2.0 + 0.5 * 6.0)
