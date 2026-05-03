from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from falling_muzero.config import load_config
from falling_muzero.trainer import PolicyMode, Trainer, load_trainer_with_checkpoint
from falling_muzero.visualization import (
    display_observations_for_episode,
    load_metrics,
    save_demo_frames,
    save_demo_gif,
    save_mcts_diagram,
    save_training_plots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MuZero grid games")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--game", choices=["catch", "2048"], default=None, help="Game to run. Defaults to config value.")
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
    demo_parser.add_argument("--gif-frames", type=int, default=None)
    demo_parser.add_argument("--gif-duration-ms", type=int, default=None)
    demo_parser.add_argument(
        "--gif-final-pause-ms",
        type=int,
        default=None,
        help="How long the GIF holds on the last frame after game over.",
    )
    demo_parser.add_argument(
        "--demo-steps",
        type=int,
        default=None,
        help="Demo-only episode length override. Useful for watching longer 2048 games.",
    )

    plot_parser = subparsers.add_parser("plot", help="Create plots from saved training metrics.")
    plot_parser.add_argument("--metrics", default=None)

    subparsers.add_parser("smoke-test", help="Run a tiny end-to-end training/evaluation pass.")

    args = parser.parse_args(argv)
    config_path = _config_path_for_args(args)
    config = load_config(config_path, overrides=_overrides_from_args(args))
    _apply_command_defaults(args, config)

    if args.command == "train":
        trainer = Trainer(config)
        metrics = trainer.train()
        outputs = save_training_plots(metrics, config.visualization.plots_dir)
        print(f"game: {config.game.kind}")
        print(f"saved checkpoint: {config.training.checkpoint_path}")
        print("saved plots:")
        for output in outputs:
            print(f"  {output}")
        return 0

    if args.command == "evaluate":
        _require_checkpoint_for_learned_mode(config, args.checkpoint, args.mode)
        trainer = _trainer_for_mode(config, args.checkpoint, args.mode)
        stats = trainer.evaluate_stats(mode=args.mode, episodes=args.episodes)
        print(f"{config.game.kind} {args.mode} average reward over {args.episodes} episodes: {stats['reward']:.3f}")
        if stats["score"] is not None:
            print(f"average raw score: {stats['score']:.1f}")
            print(f"average max tile: {stats['max_tile']:.1f}")
            print(f"average survival steps: {stats['steps']:.1f}")
            print(f"1024 reached rate: {100.0 * stats['1024_reached']:.1f}%")
        if stats["miss_percentage"] is not None:
            print(f"average catches: {stats['catches']:.1f}")
            print(f"average misses: {stats['misses']:.1f}")
            print(f"average miss percentage: {stats['miss_percentage']:.1f}%")
        if stats["heuristic_agreement_percentage"] is not None:
            print(f"average heuristic agreement: {stats['heuristic_agreement_percentage']:.1f}%")
        return 0

    if args.command == "demo":
        _require_checkpoint_for_learned_mode(config, args.checkpoint, args.mode)
        trainer = _trainer_for_mode(config, args.checkpoint, args.mode)
        episode, reward = trainer.run_episode(
            mode=args.mode,
            training=False,
            include_terminal_observation=True,
        )
        frame_paths = save_demo_frames(episode, trainer.game, config.visualization.demo_dir, max_frames=args.frames)
        gif_observations = display_observations_for_episode(episode, trainer.game, args.gif_frames)
        gif_path = save_demo_gif(
            gif_observations,
            trainer.game,
            Path(config.visualization.demo_dir) / f"{args.mode}_performance.gif",
            duration_ms=args.gif_duration_ms,
            final_pause_ms=args.gif_final_pause_ms,
        )
        observation_stack = trainer.game.stack_observations(episode.observations, 0)
        result = trainer.mcts.search(observation_stack, add_exploration_noise=False)
        diagram = save_mcts_diagram(
            result.policy,
            Path(config.visualization.demo_dir) / "mcts_root_policy.png",
            action_names=trainer.game.ACTION_NAMES,
        )
        print(f"{config.game.kind} {args.mode} demo reward: {reward:.3f}")
        print(f"saved {len(frame_paths)} demo frame/contact-sheet files under {config.visualization.demo_dir}")
        print(f"saved GIF: {gif_path} ({len(gif_observations)} frames)")
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
        return _smoke_test(config_path, config.game.kind)

    raise ValueError(f"unknown command {args.command}")


def _apply_command_defaults(args: argparse.Namespace, config) -> None:
    if args.command != "demo":
        return
    game_kind = config.game.kind
    if args.gif_frames is None:
        args.gif_frames = 600 if game_kind == "2048" else 80
    if args.gif_duration_ms is None:
        args.gif_duration_ms = 45 if game_kind == "2048" else 180
    if args.gif_final_pause_ms is None:
        args.gif_final_pause_ms = 1800 if game_kind == "2048" else 1200
    if args.demo_steps is None and game_kind == "2048":
        args.demo_steps = args.gif_frames
    if args.demo_steps is not None:
        config.game.episode_length = max(config.game.episode_length, args.demo_steps)


def _config_path_for_args(args: argparse.Namespace) -> str:
    if args.game == "2048" and args.config == "configs/default.yaml":
        return "configs/2048.yaml"
    return args.config


def _require_checkpoint_for_learned_mode(config, checkpoint: str | None, mode: PolicyMode) -> None:
    if mode in {"random", "heuristic"}:
        return
    path = Path(checkpoint or config.training.checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found for {config.game.kind} {mode} mode: {path}. "
            "Run training first, pass --checkpoint, or use --mode random/heuristic."
        )


def _trainer_for_mode(config, checkpoint: str | None, mode: PolicyMode) -> Trainer:
    if mode in {"random", "heuristic"}:
        return Trainer(config)
    return load_trainer_with_checkpoint(config, checkpoint)


def _overrides_from_args(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "game", None) is not None:
        overrides.setdefault("game", {})["kind"] = args.game
    if getattr(args, "episodes", None) is not None and args.command == "train":
        overrides.setdefault("training", {})["episodes"] = args.episodes
    if getattr(args, "simulations", None) is not None:
        overrides.setdefault("mcts", {})["simulations"] = args.simulations
    return overrides


def _smoke_test(config_path: str, game_kind: str) -> int:
    game_overrides = {
        "catch": {
            "kind": "catch",
            "width": 4,
            "height": 4,
            "episode_length": 8,
            "history_length": 2,
            "distance_shaping": 0.02,
        },
        "2048": {
            "kind": "2048",
            "size": 4,
            "episode_length": 10,
            "history_length": 2,
            "max_tile_exponent": 8,
            "invalid_move_penalty": -0.1,
            "merge_reward_scale": 64.0,
        },
    }[game_kind]
    overrides = {
        "game": game_overrides,
        "network": {"latent_dim": 16, "hidden_dim": 32},
        "mcts": {"simulations": 2, "max_depth": 2, "temperature": 1.0},
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
            "eval_episodes": 2,
            "checkpoint_path": f"artifacts/smoke/{game_kind}/best.pt",
            "final_checkpoint_path": f"artifacts/smoke/{game_kind}/final.pt",
            "metrics_path": f"artifacts/smoke/{game_kind}/metrics.json",
        },
        "visualization": {
            "plots_dir": f"artifacts/smoke/{game_kind}/plots",
            "demo_dir": f"artifacts/smoke/{game_kind}/demo",
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
        final_pause_ms=600,
    )
    policy = np.ones(reloaded.game.action_space_size, dtype=np.float32) / reloaded.game.action_space_size
    diagram = save_mcts_diagram(
        policy,
        Path(config.visualization.demo_dir) / "mcts_root_policy.png",
        action_names=reloaded.game.ACTION_NAMES,
    )
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
