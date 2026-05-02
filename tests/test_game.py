import numpy as np

from falling_muzero.config import GameConfig
from falling_muzero.game import FallingCatchGame, GameState


def test_reset_is_deterministic_and_observation_shape_is_gridded():
    config = GameConfig(width=5, height=5, paddle_width=1, history_length=3)
    game_a = FallingCatchGame(config)
    game_b = FallingCatchGame(config)

    obs_a = game_a.reset()
    obs_b = game_b.reset()

    assert obs_a.shape == (2, 5, 5)
    assert np.array_equal(obs_a, obs_b)
    assert game_a.stacked_observation_shape == (6, 5, 5)


def test_bottom_row_catch_and_miss_rewards():
    config = GameConfig(width=5, height=5, paddle_width=1, distance_shaping=0.0)
    game = FallingCatchGame(config)

    catch_state = GameState(ball_row=4, ball_col=2, paddle_left=2, step=0, spawn_count=0)
    catch = game.transition(catch_state, action=1)
    assert catch.reward == 1.0
    assert catch.state.ball_row == 0

    miss_state = GameState(ball_row=4, ball_col=4, paddle_left=0, step=0, spawn_count=0)
    miss = game.transition(miss_state, action=1)
    assert miss.reward == -1.0


def test_heuristic_moves_toward_ball():
    config = GameConfig(width=5, height=5, paddle_width=1)
    game = FallingCatchGame(config)
    state = GameState(ball_row=2, ball_col=0, paddle_left=3, step=0, spawn_count=0)
    assert game.heuristic_action(state) == 0
    state = GameState(ball_row=2, ball_col=4, paddle_left=1, step=0, spawn_count=0)
    assert game.heuristic_action(state) == 2
