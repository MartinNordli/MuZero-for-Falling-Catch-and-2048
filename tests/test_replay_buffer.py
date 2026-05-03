import numpy as np

from falling_muzero.config import GameConfig
from falling_muzero.games.falling_catch import FallingCatchGame
from falling_muzero.replay_buffer import Episode, EpisodeBuffer


def test_episode_buffer_samples_bptt_targets():
    config = GameConfig(width=4, height=4, history_length=2, episode_length=6)
    game = FallingCatchGame(config)
    obs = game.reset()
    episode = Episode()
    policy = np.ones(game.action_space_size, dtype=np.float32) / game.action_space_size

    for _ in range(4):
        result = game.step(1)
        episode.append(obs, action=1, reward=result.reward, policy=policy, search_value=0.0)
        obs = result.observation

    buffer = EpisodeBuffer(capacity=10, discount=0.95, seed=1)
    buffer.add(episode)
    batch = buffer.sample(batch_size=2, unroll_steps=2, game=game)

    assert batch.observations.shape == (2, 4, 4, 4)
    assert batch.actions.shape == (2, 2)
    assert batch.target_rewards.shape == (2, 2)
    assert batch.target_policies.shape == (2, 3, 3)
    assert batch.target_values.shape == (2, 3)
    assert batch.masks.shape == (2, 3)


def test_episode_buffer_prioritizes_high_reward_episodes():
    low = Episode()
    high = Episode()
    observation = np.zeros((2, 2, 2), dtype=np.float32)
    policy = np.ones(3, dtype=np.float32) / 3
    low.append(observation, action=0, reward=1.0, policy=policy, search_value=0.0)
    high.append(observation, action=0, reward=10.0, policy=policy, search_value=0.0)

    buffer = EpisodeBuffer(capacity=10, discount=0.95, seed=1, priority_alpha=1.0)
    buffer.add(low)
    buffer.add(high)

    assert buffer._priorities[1] > buffer._priorities[0]


def test_episode_buffer_blends_returns_with_real_mcts_search_values():
    observation = np.zeros((2, 2, 2), dtype=np.float32)
    policy = np.ones(3, dtype=np.float32) / 3
    episode = Episode()
    episode.append(
        observation,
        action=0,
        reward=2.0,
        policy=policy,
        search_value=10.0,
        has_search_value=True,
    )
    buffer = EpisodeBuffer(capacity=10, discount=1.0, search_value_target_weight=0.25)
    buffer.add(episode)

    assert np.isclose(buffer._target_value(buffer.episodes[0], 0), 4.0)

    no_search = Episode()
    no_search.append(
        observation,
        action=0,
        reward=2.0,
        policy=policy,
        search_value=10.0,
        has_search_value=False,
    )
    buffer.add(no_search)
    assert np.isclose(buffer._target_value(buffer.episodes[1], 0), 2.0)
