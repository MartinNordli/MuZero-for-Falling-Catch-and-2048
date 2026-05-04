"""End-to-end smoke runs for both games — train, checkpoint, evaluate, and plot in under a minute."""

from pathlib import Path

from falling_muzero.config import load_config
from falling_muzero.trainer import Trainer, load_trainer_with_checkpoint
from falling_muzero.visualization import save_training_plots


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
                "bootstrap_imitation_steps": 0,
                "bootstrap_gradient_steps": 0,
                "train_after_episodes": 1,
                "gradient_steps": 1,
                "batch_size": 2,
                "unroll_steps": 2,
                "eval_every": 3,
                "eval_episodes": 1,
                "mcts_eval_episodes": 1,
                "checkpoint_path": str(checkpoint),
                "final_checkpoint_path": str(tmp_path / "final.pt"),
                "metrics_path": str(metrics),
            },
        },
    )

    trainer = Trainer(config)
    training_metrics = trainer.train()
    assert checkpoint.exists()
    assert metrics.exists()
    assert "train_miss_percentage" in training_metrics
    assert "train_heuristic_agreement_percentage" in training_metrics
    assert any(value is not None for value in training_metrics["train_miss_percentage"])
    assert any(value is not None for value in training_metrics["train_heuristic_agreement_percentage"])
    plot_outputs = save_training_plots(training_metrics, tmp_path / "plots")
    assert tmp_path / "plots" / "falling_catch_summary.png" in plot_outputs
    assert tmp_path / "plots" / "heuristic_agreement_eval_only.png" in plot_outputs

    reloaded = load_trainer_with_checkpoint(config, checkpoint)
    stats = reloaded.evaluate_stats("actor", episodes=1)
    reward = stats["reward"]
    assert isinstance(reward, float)
    assert stats["miss_percentage"] is not None
    assert stats["heuristic_agreement_percentage"] is not None


def test_tiny_2048_training_checkpoint_and_evaluation(tmp_path: Path):
    checkpoint = tmp_path / "2048_checkpoint.pt"
    metrics = tmp_path / "2048_metrics.json"
    config = load_config(
        "configs/2048.yaml",
        overrides={
            "game": {
                "kind": "2048",
                "size": 4,
                "episode_length": 10,
                "history_length": 2,
                "max_tile_exponent": 8,
                "invalid_move_penalty": -0.1,
                "merge_reward_scale": 64.0,
            },
            "network": {"latent_dim": 16, "hidden_dim": 32},
            "mcts": {"simulations": 2, "max_depth": 2},
            "training": {
                "episodes": 2,
                "bootstrap_episodes": 1,
                "bootstrap_imitation_steps": 1,
                "bootstrap_gradient_steps": 1,
                "train_after_episodes": 1,
                "gradient_steps": 1,
                "batch_size": 2,
                "unroll_steps": 2,
                "eval_every": 2,
                "eval_episodes": 1,
                "checkpoint_path": str(checkpoint),
                "final_checkpoint_path": str(tmp_path / "2048_final.pt"),
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
