import numpy as np

from falling_muzero.config import GameConfig, MCTSConfig, NetworkConfig
from falling_muzero.games.game_2048 import Game2048, Game2048State
from falling_muzero.game_factory import create_game
from falling_muzero.mcts import MuZeroMCTS
from falling_muzero.networks import MuZeroNetwork
import torch


def test_2048_merge_line_merges_each_tile_once():
    game = Game2048(GameConfig(kind="2048", size=4))
    merged, score = game._merge_line(np.asarray([1, 1, 1, 1], dtype=np.int16))

    assert merged.tolist() == [2, 2, 0, 0]
    assert score == 8


def test_2048_left_move_and_deterministic_spawn():
    config = GameConfig(kind="2048", size=4, spawn_seed=3)
    game_a = Game2048(config)
    game_b = Game2048(config)

    assert np.array_equal(game_a.reset(), game_b.reset())

    state = Game2048State(
        board=((1, 1, 0, 0), (2, 0, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        step=0,
        spawn_count=0,
        score=0,
    )
    result = game_a.transition(state, action=2)
    board = np.asarray(result.state.board)

    assert board[0, 0] == 2
    assert board[1, 0] == 3
    assert result.state.score == 12
    assert result.reward == 12 / config.merge_reward_scale
    assert result.state.spawn_count == 1


def test_2048_invalid_move_penalty_and_terminal_detection():
    config = GameConfig(kind="2048", size=4, invalid_move_penalty=-0.25)
    game = Game2048(config)
    state = Game2048State(
        board=((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        step=0,
        spawn_count=0,
        score=0,
    )

    result = game.transition(state, action=2)
    assert result.reward == -0.25
    assert result.state.spawn_count == 0

    terminal = Game2048State(
        board=((1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1)),
        step=0,
        spawn_count=0,
        score=0,
    )
    assert game.legal_actions(terminal) == ()
    assert game.transition(terminal, action=0).done


def test_2048_coerces_illegal_action_to_best_legal_policy_action():
    game = Game2048(GameConfig(kind="2048", size=4))
    state = Game2048State(
        board=((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        step=0,
        spawn_count=0,
        score=0,
    )
    game._state = state
    policy = np.asarray([0.1, 0.8, 0.09, 0.01], dtype=np.float32)

    assert game.legal_actions() == (1, 3)
    assert game.coerce_action(2, policy) == 1
    assert game.coerce_action(3, policy) == 3


def test_2048_observation_shape_and_heuristic_action():
    game = Game2048(GameConfig(kind="2048", size=4, history_length=3, max_tile_exponent=8))
    obs = game.reset()
    tile_channels = game.tile_channel_count

    assert obs.shape == (15, 4, 4)
    assert game.stacked_observation_shape == (45, 4, 4)
    assert np.allclose(obs[:tile_channels].sum(axis=0), 1.0)
    assert np.allclose(obs[tile_channels], 0.0)
    assert np.allclose(obs[tile_channels + 1], 2 / (game.config.episode_length + 2))
    assert np.allclose(obs[tile_channels + 2], 0.0)
    assert np.allclose(obs[tile_channels + 3], 2 / 9.0)
    assert np.array_equal(game.board_from_observation(obs), game.board_from_observation(game.stack_observations([obs], 0)))
    assert game.heuristic_action() in game.legal_actions()


def test_factory_and_mcts_support_2048():
    game = create_game(GameConfig(kind="2048", size=4, history_length=2, max_tile_exponent=8))
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
    result = mcts.search(stack, add_exploration_noise=False, legal_actions=(1, 3))

    assert result.policy.shape == (4,)
    assert np.isclose(result.policy.sum(), 1.0)
    assert np.all(result.policy >= 0.0)
    assert result.policy[0] == 0.0
    assert result.policy[2] == 0.0
