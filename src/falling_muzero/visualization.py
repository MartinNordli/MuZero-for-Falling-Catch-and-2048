from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_MPL_DIR = Path.cwd() / "artifacts" / ".mplconfig"
_CACHE_DIR = Path.cwd() / "artifacts" / ".cache"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from falling_muzero.game import FallingCatchGame
from falling_muzero.replay_buffer import Episode


def load_metrics(path: str | Path) -> dict[str, list[float | int | None]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_training_plots(metrics: dict[str, list[float | int | None]], plots_dir: str | Path) -> list[Path]:
    target_dir = Path(plots_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _save_reward_plot(metrics, target_dir / "training_rewards.png"),
        _save_loss_plot(metrics, target_dir / "loss_curves.png"),
    ]
    return outputs


def save_demo_frames(
    episode: Episode,
    game: FallingCatchGame,
    demo_dir: str | Path,
    max_frames: int = 16,
) -> list[Path]:
    target_dir = Path(demo_dir)
    frames_dir = target_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for index, observation in enumerate(episode.observations[:max_frames]):
        path = frames_dir / f"frame_{index:03d}.png"
        _save_grid_image(observation, game, path)
        frame_paths.append(path)

    if frame_paths:
        contact_sheet = target_dir / "demo_contact_sheet.png"
        _save_contact_sheet(episode.observations[:max_frames], game, contact_sheet)
        frame_paths.append(contact_sheet)
    return frame_paths


def save_demo_gif(
    observations: list[np.ndarray],
    game: FallingCatchGame,
    path: str | Path,
    duration_ms: int = 180,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not observations:
        raise ValueError("cannot create GIF without observations")

    frames = [
        Image.fromarray((_observation_to_rgb(observation) * 255).astype(np.uint8)).resize(
            (game.config.width * 72, game.config.height * 72),
            resample=Image.Resampling.NEAREST,
        )
        for observation in observations
    ]
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return target


def save_mcts_diagram(policy: np.ndarray, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    actions = ["left", "stay", "right"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    root = (0.5, 0.82)
    children = [(0.2, 0.25), (0.5, 0.25), (0.8, 0.25)]
    ax.scatter([root[0]], [root[1]], s=1500, color="#2f80ed")
    ax.text(root[0], root[1], "root\nlatent", ha="center", va="center", color="white", fontsize=11)
    for child, action, probability in zip(children, actions, policy):
        ax.annotate(
            "",
            xy=child,
            xytext=root,
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#333333"},
        )
        ax.scatter([child[0]], [child[1]], s=1100, color="#27ae60")
        ax.text(
            child[0],
            child[1],
            f"{action}\n{probability:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def _save_reward_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    train_reward = np.asarray(metrics["train_reward"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, train_reward, color="#2f80ed", alpha=0.35, label="self-play episode")
    if len(train_reward) >= 5:
        window = min(10, len(train_reward))
        smooth = np.convolve(train_reward, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1 :], smooth, color="#2f80ed", label=f"{window}-episode average")
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_reward", []), "#27ae60", "actor eval")
    _plot_sparse_metric(ax, episodes, metrics.get("random_eval_reward", []), "#eb5757", "random baseline")
    _plot_sparse_metric(ax, episodes, metrics.get("heuristic_eval_reward", []), "#8e44ad", "heuristic baseline")
    ax.set_title("Falling Catch reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_loss_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in [
        ("loss", "#333333", "total"),
        ("policy_loss", "#2f80ed", "policy"),
        ("value_loss", "#27ae60", "value"),
        ("reward_loss", "#eb5757", "reward"),
    ]:
        _plot_sparse_metric(ax, episodes, metrics.get(key, []), color, label)
    ax.set_title("MuZero training losses")
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_sparse_metric(ax: plt.Axes, episodes: np.ndarray, values: list[float | int | None], color: str, label: str) -> None:
    points = [(episode, value) for episode, value in zip(episodes, values) if value is not None]
    if not points:
        return
    x, y = zip(*points)
    ax.plot(x, y, marker="o", color=color, label=label)


def _legend_if_present(ax: plt.Axes) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _save_grid_image(observation: np.ndarray, game: FallingCatchGame, path: Path) -> None:
    rgb = _observation_to_rgb(observation)
    fig, ax = plt.subplots(figsize=(game.config.width, game.config.height))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, game.config.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, game.config.height, 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_contact_sheet(observations: list[np.ndarray], game: FallingCatchGame, path: Path) -> None:
    cols = min(4, len(observations))
    rows = int(np.ceil(len(observations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for index, ax in enumerate(axes_array.flat):
        ax.axis("off")
        if index >= len(observations):
            continue
        ax.imshow(_observation_to_rgb(observations[index]), interpolation="nearest")
        ax.set_title(f"t={index}", fontsize=9)
        ax.set_xticks(np.arange(-0.5, game.config.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, game.config.height, 1), minor=True)
        ax.grid(which="minor", color="#444444", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _observation_to_rgb(observation: np.ndarray) -> np.ndarray:
    ball = observation[0]
    paddle = observation[1]
    height, width = ball.shape
    rgb = np.ones((height, width, 3), dtype=np.float32) * np.array([0.96, 0.96, 0.93], dtype=np.float32)
    rgb[paddle > 0] = np.array([0.15, 0.49, 0.93], dtype=np.float32)
    rgb[ball > 0] = np.array([0.9, 0.18, 0.16], dtype=np.float32)
    overlap = (ball > 0) & (paddle > 0)
    rgb[overlap] = np.array([0.12, 0.65, 0.33], dtype=np.float32)
    return rgb
