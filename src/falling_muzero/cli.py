from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from falling_muzero.config import load_config
from falling_muzero.trainer import PolicyMode, Trainer, load_trainer_with_checkpoint
from falling_muzero.visualization import (
    load_metrics,
    save_demo_frames,
    save_demo_gif,
    save_mcts_diagram,
    save_training_plots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MuZero Falling Catch")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train MuZero by self-play.")
    train_parser.add_argument("--episodes", type=int, default=None)
    train_parser.add_argument("--simulations", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--mode", choices=["actor", "mcts", "random", "heuristic"], default="actor")
    eval_parser.add_argument("--episodes", type=int, default=20)

    demo_parser = subparsers.add_parser("demo", help="Render gameplay frames and an MCTS diagram.")
    demo_parser.add_argument("--checkpoint", default=None)
    demo_parser.add_argument("--mode", choices=["actor", "mcts", "random", "heuristic"], default="actor")
    demo_parser.add_argument("--frames", type=int, default=16)
    demo_parser.add_argument("--gif-frames", type=int, default=80)
    demo_parser.add_argument("--gif-duration-ms", type=int, default=180)

    plot_parser = subparsers.add_parser("plot", help="Create plots from saved training metrics.")
    plot_parser.add_argument("--metrics", default=None)

    subparsers.add_parser("smoke-test", help="Run a tiny end-to-end training/evaluation pass.")

    args = parser.parse_args(argv)
    config = load_config(args.config, overrides=_overrides_from_args(args))

    if args.command == "train":
        trainer = Trainer(config)
        metrics = trainer.train()
        outputs = save_training_plots(metrics, config.visualization.plots_dir)
        print(f"saved checkpoint: {config.training.checkpoint_path}")
        print("saved plots:")
        for output in outputs:
            print(f"  {output}")
        return 0

    if args.command == "evaluate":
        trainer = load_trainer_with_checkpoint(config, args.checkpoint)
        reward = trainer.evaluate(mode=args.mode, episodes=args.episodes)
        print(f"{args.mode} average reward over {args.episodes} episodes: {reward:.3f}")
        return 0

    if args.command == "demo":
        trainer = load_trainer_with_checkpoint(config, args.checkpoint)
        episode, reward = trainer.run_episode(mode=args.mode, training=False)
        frame_paths = save_demo_frames(episode, trainer.game, config.visualization.demo_dir, max_frames=args.frames)
        gif_observations = _collect_demo_observations(trainer, args.mode, args.gif_frames)
        gif_path = save_demo_gif(
            gif_observations,
            trainer.game,
            Path(config.visualization.demo_dir) / f"{args.mode}_performance.gif",
            duration_ms=args.gif_duration_ms,
        )
        observation_stack = trainer.game.stack_observations(episode.observations, 0)
        result = trainer.mcts.search(observation_stack, add_exploration_noise=False)
        diagram = save_mcts_diagram(result.policy, Path(config.visualization.demo_dir) / "mcts_root_policy.png")
        print(f"{args.mode} demo reward: {reward:.3f}")
        print(f"saved {len(frame_paths)} demo frame/contact-sheet files under {config.visualization.demo_dir}")
        print(f"saved GIF: {gif_path}")
        print(f"saved MCTS diagram: {diagram}")
        return 0

    if args.command == "plot":
        metrics_path = args.metrics or config.training.metrics_path
        metrics = load_metrics(metrics_path)
        outputs = save_training_plots(metrics, config.visualization.plots_dir)
        for output in outputs:
            print(f"saved plot: {output}")
        return 0

    if args.command == "smoke-test":
        return _smoke_test(args.config)

    raise ValueError(f"unknown command {args.command}")


def _collect_demo_observations(trainer: Trainer, mode: PolicyMode, frame_count: int) -> list[np.ndarray]:
    if frame_count < 1:
        raise ValueError("gif frame count must be positive")
    observations = []
    while len(observations) < frame_count:
        episode, _ = trainer.run_episode(mode=mode, training=False)
        observations.extend(episode.observations)
    return observations[:frame_count]


def _overrides_from_args(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "episodes", None) is not None and args.command == "train":
        overrides.setdefault("training", {})["episodes"] = args.episodes
    if getattr(args, "simulations", None) is not None:
        overrides.setdefault("mcts", {})["simulations"] = args.simulations
    return overrides


def _smoke_test(config_path: str) -> int:
    overrides = {
        "game": {
            "width": 4,
            "height": 4,
            "episode_length": 8,
            "history_length": 2,
            "distance_shaping": 0.02,
        },
        "network": {"latent_dim": 16, "hidden_dim": 32},
        "mcts": {"simulations": 2, "max_depth": 2, "temperature": 1.0},
        "training": {
            "episodes": 3,
            "bootstrap_episodes": 0,
            "bootstrap_gradient_steps": 0,
            "train_after_episodes": 1,
            "gradient_steps": 1,
            "batch_size": 2,
            "unroll_steps": 2,
            "eval_every": 3,
            "eval_episodes": 2,
            "checkpoint_path": "artifacts/smoke/best.pt",
            "final_checkpoint_path": "artifacts/smoke/final.pt",
            "metrics_path": "artifacts/smoke/metrics.json",
        },
        "visualization": {
            "plots_dir": "artifacts/smoke/plots",
            "demo_dir": "artifacts/smoke/demo",
        },
    }
    config = load_config(config_path, overrides=overrides)
    trainer = Trainer(config)
    metrics = trainer.train()
    reloaded = load_trainer_with_checkpoint(config, config.training.checkpoint_path)
    actor_reward = reloaded.evaluate("actor", episodes=1)
    random_reward = reloaded.evaluate("random", episodes=1)
    outputs = save_training_plots(metrics, config.visualization.plots_dir)
    episode, demo_reward = reloaded.run_episode(mode="actor", training=False)
    frame_paths = save_demo_frames(episode, reloaded.game, config.visualization.demo_dir, max_frames=4)
    gif_path = save_demo_gif(
        episode.observations[:4],
        reloaded.game,
        Path(config.visualization.demo_dir) / "actor_performance.gif",
        duration_ms=120,
    )
    policy = np.ones(reloaded.game.action_space_size, dtype=np.float32) / reloaded.game.action_space_size
    diagram = save_mcts_diagram(policy, Path(config.visualization.demo_dir) / "mcts_root_policy.png")
    print("smoke test completed")
    print(f"actor reward: {actor_reward:.3f}")
    print(f"random reward: {random_reward:.3f}")
    print(f"demo reward: {demo_reward:.3f}")
    print(f"checkpoint: {config.training.checkpoint_path}")
    for output in [*outputs, *frame_paths, gif_path, diagram]:
        print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
