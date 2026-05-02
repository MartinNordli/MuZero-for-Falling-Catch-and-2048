import numpy as np

from falling_muzero.config import GameConfig
from falling_muzero.game import FallingCatchGame
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
