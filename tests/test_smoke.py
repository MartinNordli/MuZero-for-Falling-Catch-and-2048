from pathlib import Path

from falling_muzero.config import load_config
from falling_muzero.trainer import Trainer, load_trainer_with_checkpoint


def test_tiny_training_checkpoint_and_evaluation(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    metrics = tmp_path / "metrics.json"
    config = load_config(
        "configs/default.yaml",
        overrides={
            "game": {
                "width": 4,
                "height": 4,
                "episode_length": 8,
                "history_length": 2,
                "distance_shaping": 0.02,
            },
            "network": {"latent_dim": 16, "hidden_dim": 32},
            "mcts": {"simulations": 2, "max_depth": 2},
            "training": {
                "episodes": 3,
                "bootstrap_episodes": 0,
                "bootstrap_gradient_steps": 0,
                "train_after_episodes": 1,
                "gradient_steps": 1,
                "batch_size": 2,
                "unroll_steps": 2,
                "eval_every": 3,
                "eval_episodes": 1,
                "checkpoint_path": str(checkpoint),
                "final_checkpoint_path": str(tmp_path / "final.pt"),
                "metrics_path": str(metrics),
            },
        },
    )

    trainer = Trainer(config)
    trainer.train()
    assert checkpoint.exists()
    assert metrics.exists()

    reloaded = load_trainer_with_checkpoint(config, checkpoint)
    reward = reloaded.evaluate("actor", episodes=1)
    assert isinstance(reward, float)
