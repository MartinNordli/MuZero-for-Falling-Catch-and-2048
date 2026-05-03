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
from PIL import Image, ImageDraw, ImageFont

from falling_muzero.games.types import MuZeroGame
from falling_muzero.replay_buffer import Episode

_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


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
    if _has_metric_values(metrics, "train_miss_percentage") or _has_metric_values(metrics, "actor_eval_miss_percentage"):
        outputs.extend(
            [
                _save_miss_percentage_plot(metrics, target_dir / "miss_percentage_progress.png"),
                _save_miss_percentage_plot(
                    metrics,
                    target_dir / "miss_percentage_eval_only.png",
                    include_self_play=False,
                ),
                _save_falling_catch_summary_plot(metrics, target_dir / "falling_catch_summary.png"),
            ]
        )
    if _has_metric_values(metrics, "train_heuristic_agreement_percentage") or _has_metric_values(
        metrics,
        "actor_eval_heuristic_agreement_percentage",
    ):
        outputs.extend(
            [
                _save_heuristic_agreement_plot(metrics, target_dir / "heuristic_agreement_progress.png"),
                _save_heuristic_agreement_plot(
                    metrics,
                    target_dir / "heuristic_agreement_eval_only.png",
                    include_self_play=False,
                ),
            ]
        )
    if _has_metric_values(metrics, "train_score"):
        outputs.extend(
            [
                _save_score_plot(metrics, target_dir / "raw_score_progress.png"),
                _save_max_tile_plot(metrics, target_dir / "max_tile_progress.png"),
                _save_survival_plot(metrics, target_dir / "survival_steps_progress.png"),
                _save_2048_summary_plot(metrics, target_dir / "performance_summary.png"),
            ]
        )
    return outputs


def save_demo_frames(
    episode: Episode,
    game: MuZeroGame,
    demo_dir: str | Path,
    max_frames: int = 16,
) -> list[Path]:
    target_dir = Path(demo_dir)
    frames_dir = target_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    display_observations = display_observations_for_episode(episode, game, max_frames)

    frame_paths: list[Path] = []
    for index, observation in enumerate(display_observations):
        path = frames_dir / f"frame_{index:03d}.png"
        _save_grid_image(observation, game, path)
        frame_paths.append(path)

    if frame_paths:
        contact_sheet = target_dir / "demo_contact_sheet.png"
        _save_contact_sheet(display_observations, game, contact_sheet)
        frame_paths.append(contact_sheet)
    return frame_paths


def display_observations_for_episode(
    episode: Episode,
    game: MuZeroGame,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Return demo frames that match the action timing used by the simulator.

    Falling Catch applies the chosen action before scoring a bottom-row object.
    Raw observations are the decision states before that action, so a successful
    catch can look like a miss in a GIF. For display only, move the paddle by
    the recorded action before rendering the frame.
    """

    observations = episode.observations if max_frames is None else episode.observations[:max_frames]
    if game.config.kind != "catch":
        return [observation.astype(np.float32, copy=True) for observation in observations]

    display_observations: list[np.ndarray] = []
    for index, observation in enumerate(observations):
        action = episode.actions[index] if index < len(episode.actions) else None
        display_observations.append(_catch_observation_after_action(observation, game, action))
    return display_observations


def save_demo_gif(
    observations: list[np.ndarray],
    game: MuZeroGame,
    path: str | Path,
    duration_ms: int = 180,
    final_pause_ms: int = 1400,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not observations:
        raise ValueError("cannot create GIF without observations")
    if duration_ms < 1:
        raise ValueError("GIF frame duration must be positive")
    if final_pause_ms < 0:
        raise ValueError("GIF final pause must be non-negative")

    frames = [_stamp_frame_number(_observation_to_pil(observation, game), index) for index, observation in enumerate(observations)]
    durations = [duration_ms] * len(frames)
    if final_pause_ms > 0:
        durations[-1] = final_pause_ms
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return target


def save_mcts_diagram(policy: np.ndarray, path: str | Path, action_names: tuple[str, ...] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    actions = action_names or tuple(f"a{i}" for i in range(len(policy)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    root = (0.5, 0.82)
    children = [(x, 0.25) for x in np.linspace(0.15, 0.85, len(policy))]
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
    is_2048 = _has_metric_values(metrics, "train_score")
    train_color = "#9aa0a6" if is_2048 else "#2f80ed"
    train_label = "self-play episode after updates" if is_2048 else "self-play episode"
    ax.plot(episodes, train_reward, color=train_color, alpha=0.35, linewidth=1.3, label=train_label)
    if len(train_reward) >= 5:
        window = min(10, len(train_reward))
        smooth = np.convolve(train_reward, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1 :], smooth, color=train_color, linestyle="--", alpha=0.85, label=f"self-play {window}-episode avg")
    if is_2048:
        _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_reward", []), "#f2994a", "current actor eval")
        _plot_sparse_metric(ax, episodes, metrics.get("best_actor_eval_reward", []), "#1f8f4d", "best checkpoint used by demo/evaluate", linewidth=2.8)
        _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_reward", []), "#2f80ed", "MCTS eval")
        _plot_horizontal_reference(ax, episodes, metrics.get("random_eval_reward", []), "#eb5757", "random baseline")
        _plot_horizontal_reference(ax, episodes, metrics.get("heuristic_eval_reward", []), "#8e44ad", "heuristic guide")
    else:
        _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_reward", []), "#27ae60", "actor eval")
        _plot_sparse_metric(ax, episodes, metrics.get("best_actor_eval_reward", []), "#1f8f4d", "best actor checkpoint")
        _plot_sparse_metric(ax, episodes, metrics.get("random_eval_reward", []), "#eb5757", "random baseline")
        _plot_sparse_metric(ax, episodes, metrics.get("heuristic_eval_reward", []), "#8e44ad", "heuristic baseline")
    title = "2048 reward progression" if is_2048 else "Falling Catch reward"
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_score_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_sparse_metric(ax, episodes, metrics.get("train_score", []), "#9aa0a6", "self-play episode after updates", alpha=0.35, linewidth=1.3)
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_score", []), "#f2994a", "current actor eval")
    _plot_sparse_metric(ax, episodes, metrics.get("best_actor_eval_score", []), "#1f8f4d", "best checkpoint used by demo/evaluate", linewidth=2.8)
    _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_score", []), "#2f80ed", "MCTS eval")
    _plot_horizontal_reference(ax, episodes, metrics.get("random_eval_score", []), "#eb5757", "random baseline")
    _plot_horizontal_reference(ax, episodes, metrics.get("heuristic_eval_score", []), "#8e44ad", "heuristic guide")
    ax.set_title("2048 raw score")
    ax.set_xlabel("episode")
    ax.set_ylabel("raw 2048 score")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_max_tile_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_sparse_metric(ax, episodes, metrics.get("train_max_tile", []), "#9aa0a6", "self-play episode after updates", alpha=0.35, linewidth=1.3)
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_max_tile", []), "#f2994a", "current actor eval")
    _plot_sparse_metric(ax, episodes, metrics.get("best_actor_eval_max_tile", []), "#1f8f4d", "best checkpoint used by demo/evaluate", linewidth=2.8)
    _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_max_tile", []), "#2f80ed", "MCTS eval")
    _plot_horizontal_reference(ax, episodes, metrics.get("random_eval_max_tile", []), "#eb5757", "random baseline")
    _plot_horizontal_reference(ax, episodes, metrics.get("heuristic_eval_max_tile", []), "#8e44ad", "heuristic guide")
    ax.axhline(1024, color="#333333", linestyle="--", linewidth=1.2, label="1024 target")
    ax.set_title("2048 max tile reached")
    ax.set_xlabel("episode")
    ax.set_ylabel("max tile")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_survival_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_sparse_metric(ax, episodes, metrics.get("train_steps", []), "#9aa0a6", "self-play episode after updates", alpha=0.35, linewidth=1.3)
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_steps", []), "#f2994a", "current actor eval")
    _plot_sparse_metric(ax, episodes, metrics.get("best_actor_eval_steps", []), "#1f8f4d", "best checkpoint used by demo/evaluate", linewidth=2.8)
    _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_steps", []), "#2f80ed", "MCTS eval")
    _plot_horizontal_reference(ax, episodes, metrics.get("random_eval_steps", []), "#eb5757", "random baseline")
    _plot_horizontal_reference(ax, episodes, metrics.get("heuristic_eval_steps", []), "#8e44ad", "heuristic guide")
    ax.set_title("2048 survival length")
    ax.set_xlabel("episode")
    ax.set_ylabel("moves survived")
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_2048_summary_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    groups = [
        ("Best actor", "best_actor_eval", "#1f8f4d"),
        ("Current actor", "actor_eval", "#f2994a"),
        ("MCTS search", "mcts_eval", "#2f80ed"),
        ("Random", "random_eval", "#eb5757"),
        ("Heuristic guide", "heuristic_eval", "#8e44ad"),
    ]
    series = [
        ("Reward", "reward"),
        ("Raw score", "score"),
        ("Max tile", "max_tile"),
        ("Moves survived", "steps"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (title, suffix) in zip(axes.flat, series):
        labels: list[str] = []
        colors: list[str] = []
        values: list[float] = []
        for label, prefix, color in groups:
            value = _last_value(metrics.get(f"{prefix}_{suffix}", []))
            if value is None:
                continue
            labels.append(label)
            colors.append(color)
            values.append(float(value))
        ax.bar(labels, values, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
        for index, value in enumerate(values):
            label = f"{value:.0f}" if abs(float(value)) >= 10 else f"{value:.2f}"
            ax.text(index, float(value), label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("2048 performance summary")
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


def _save_miss_percentage_plot(
    metrics: dict[str, list[float | int | None]],
    path: Path,
    include_self_play: bool = True,
) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if include_self_play:
        _plot_sparse_metric(
            ax,
            episodes,
            metrics.get("train_miss_percentage", []),
            "#9aa0a6",
            "self-play episode",
            alpha=0.35,
            linewidth=1.3,
        )
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_miss_percentage", []), "#27ae60", "actor eval")
    _plot_sparse_metric(
        ax,
        episodes,
        metrics.get("best_actor_eval_miss_percentage", []),
        "#1f8f4d",
        "best actor checkpoint",
        linewidth=2.8,
    )
    _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_miss_percentage", []), "#2f80ed", "MCTS eval")
    _plot_horizontal_reference(ax, episodes, metrics.get("random_eval_miss_percentage", []), "#eb5757", "random baseline")
    _plot_horizontal_reference(ax, episodes, metrics.get("heuristic_eval_miss_percentage", []), "#8e44ad", "heuristic baseline")
    title = "Falling Catch miss percentage" if include_self_play else "Falling Catch evaluation miss percentage"
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("misses / catch opportunities (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_heuristic_agreement_plot(
    metrics: dict[str, list[float | int | None]],
    path: Path,
    include_self_play: bool = True,
) -> Path:
    episodes = np.asarray(metrics["episode"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if include_self_play:
        _plot_sparse_metric(
            ax,
            episodes,
            metrics.get("train_heuristic_agreement_percentage", []),
            "#9aa0a6",
            "self-play episode",
            alpha=0.35,
            linewidth=1.3,
        )
    _plot_sparse_metric(ax, episodes, metrics.get("actor_eval_heuristic_agreement_percentage", []), "#27ae60", "actor eval")
    _plot_sparse_metric(
        ax,
        episodes,
        metrics.get("best_actor_eval_heuristic_agreement_percentage", []),
        "#1f8f4d",
        "best actor checkpoint",
        linewidth=2.8,
    )
    _plot_sparse_metric(ax, episodes, metrics.get("mcts_eval_heuristic_agreement_percentage", []), "#2f80ed", "MCTS eval")
    _plot_horizontal_reference(
        ax,
        episodes,
        metrics.get("random_eval_heuristic_agreement_percentage", []),
        "#eb5757",
        "random baseline",
    )
    _plot_horizontal_reference(
        ax,
        episodes,
        metrics.get("heuristic_eval_heuristic_agreement_percentage", []),
        "#8e44ad",
        "heuristic baseline",
    )
    title = "Falling Catch heuristic agreement" if include_self_play else "Falling Catch evaluation heuristic agreement"
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("actions matching heuristic (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    _legend_if_present(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_falling_catch_summary_plot(metrics: dict[str, list[float | int | None]], path: Path) -> Path:
    groups = [
        ("Best actor", "best_actor_eval", "#1f8f4d"),
        ("Current actor", "actor_eval", "#27ae60"),
        ("MCTS search", "mcts_eval", "#2f80ed"),
        ("Random", "random_eval", "#eb5757"),
        ("Heuristic", "heuristic_eval", "#8e44ad"),
    ]
    series = [
        ("Reward", "reward"),
        ("Miss %", "miss_percentage"),
        ("Catches", "catches"),
        ("Heuristic agreement %", "heuristic_agreement_percentage"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (title, suffix) in zip(axes.flat, series):
        labels: list[str] = []
        colors: list[str] = []
        values: list[float] = []
        for label, prefix, color in groups:
            value = _last_value(metrics.get(f"{prefix}_{suffix}", []))
            if value is None:
                continue
            labels.append(label)
            colors.append(color)
            values.append(float(value))
        ax.bar(labels, values, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
        for index, value in enumerate(values):
            label = f"{value:.0f}" if abs(float(value)) >= 10 else f"{value:.2f}"
            ax.text(index, float(value), label, ha="center", va="bottom", fontsize=8)
        if suffix in {"miss_percentage", "heuristic_agreement_percentage"}:
            ax.set_ylim(0, 105)
    fig.suptitle("Falling Catch performance summary")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_sparse_metric(
    ax: plt.Axes,
    episodes: np.ndarray,
    values: list[float | int | None],
    color: str,
    label: str,
    linewidth: float = 1.8,
    alpha: float = 1.0,
) -> None:
    points = [(episode, value) for episode, value in zip(episodes, values) if value is not None]
    if not points:
        return
    x, y = zip(*points)
    ax.plot(x, y, marker="o", color=color, linewidth=linewidth, alpha=alpha, label=label)


def _plot_horizontal_reference(
    ax: plt.Axes,
    episodes: np.ndarray,
    values: list[float | int | None],
    color: str,
    label: str,
) -> None:
    value = _last_value(values)
    if value is None or len(episodes) == 0:
        return
    ax.axhline(float(value), color=color, linestyle=":", linewidth=2.0, label=f"{label} ({value:.1f})")


def _last_value(values: list[float | int | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return float(value)
    return None


def _has_metric_values(metrics: dict[str, list[float | int | None]], key: str) -> bool:
    return any(value is not None for value in metrics.get(key, []))


def _legend_if_present(ax: plt.Axes) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _save_grid_image(observation: np.ndarray, game: MuZeroGame, path: Path) -> None:
    _observation_to_pil(observation, game).save(path)


def _save_contact_sheet(observations: list[np.ndarray], game: MuZeroGame, path: Path) -> None:
    cols = min(4, len(observations))
    rows = int(np.ceil(len(observations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for index, ax in enumerate(axes_array.flat):
        ax.axis("off")
        if index >= len(observations):
            continue
        ax.imshow(np.asarray(_observation_to_pil(observations[index], game)), interpolation="nearest")
        ax.set_title(f"t={index}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _observation_to_pil(observation: np.ndarray, game: MuZeroGame) -> Image.Image:
    if game.config.kind == "2048":
        return _observation_2048_to_pil(observation, game)
    rgb = _observation_to_rgb(observation)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (game.config.width * 72, game.config.height * 72),
        resample=Image.Resampling.NEAREST,
    )


def _stamp_frame_number(image: Image.Image, index: int) -> Image.Image:
    stamped = image.copy()
    draw = ImageDraw.Draw(stamped)
    label = f"t={index:03d}"
    font = _load_font(18)
    bbox = draw.textbbox((0, 0), label, font=font)
    padding = 6
    x0 = 5
    y0 = 5
    x1 = x0 + (bbox[2] - bbox[0]) + 2 * padding
    y1 = y0 + (bbox[3] - bbox[1]) + 2 * padding
    draw.rounded_rectangle([x0, y0, x1, y1], radius=5, fill=(255, 255, 255))
    draw.text((x0 + padding, y0 + padding), label, fill=(30, 30, 30), font=font)
    return stamped


def _observation_to_rgb(observation: np.ndarray) -> np.ndarray:
    ball = observation[0]
    paddle = observation[1]
    height, width = ball.shape
    rgb = np.ones((height, width, 3), dtype=np.float32) * np.array([0.96, 0.96, 0.93], dtype=np.float32)
    rgb[paddle > 0] = np.array([0.15, 0.49, 0.93], dtype=np.float32)
    rgb[ball > 0] = np.array([0.9, 0.18, 0.16], dtype=np.float32)
    overlap = (ball > 0) & (paddle > 0)
    rgb[overlap] = np.array([0.55, 0.22, 0.78], dtype=np.float32)
    return rgb


def _catch_observation_after_action(
    observation: np.ndarray,
    game: MuZeroGame,
    action: int | None,
) -> np.ndarray:
    adjusted = observation.astype(np.float32, copy=True)
    if action is None or action < 0 or action >= len(game.ACTIONS):
        return adjusted
    paddle_cells = np.argwhere(adjusted[1] > 0)
    if paddle_cells.size == 0:
        return adjusted

    width = adjusted.shape[2]
    row = int(paddle_cells[:, 0].max())
    current_left = int(paddle_cells[:, 1].min())
    paddle_width = int(game.config.paddle_width)
    max_left = width - paddle_width
    next_left = int(np.clip(current_left + game.ACTIONS[action], 0, max_left))

    adjusted[1] = 0.0
    adjusted[1, row, next_left : next_left + paddle_width] = 1.0
    return adjusted


def _observation_2048_to_pil(observation: np.ndarray, game: MuZeroGame) -> Image.Image:
    board = game.board_from_observation(observation)
    size = game.config.size
    tile = 108
    gap = 10
    canvas = size * tile + (size + 1) * gap
    image = Image.new("RGB", (canvas, canvas), "#bbada0")
    draw = ImageDraw.Draw(image)
    colors = {
        0: "#cdc1b4",
        1: "#eee4da",
        2: "#ede0c8",
        3: "#f2b179",
        4: "#f59563",
        5: "#f67c5f",
        6: "#f65e3b",
        7: "#edcf72",
        8: "#edcc61",
        9: "#edc850",
        10: "#edc53f",
        11: "#edc22e",
    }
    for row in range(size):
        for col in range(size):
            exponent = int(board[row, col])
            x0 = gap + col * (tile + gap)
            y0 = gap + row * (tile + gap)
            x1 = x0 + tile
            y1 = y0 + tile
            draw.rounded_rectangle([x0, y0, x1, y1], radius=9, fill=colors.get(exponent, "#3c3a32"))
            if exponent > 0:
                label = str(2**exponent)
                font = _fit_font(draw, label, max_width=tile * 0.86, max_height=tile * 0.62, initial_size=46)
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_color = "#776e65" if exponent <= 2 else "#f9f6f2"
                text_x = x0 + (tile - text_w) / 2 - bbox[0]
                text_y = y0 + (tile - text_h) / 2 - bbox[1]
                shadow = "#fdf9ef" if exponent <= 2 else "#6f5f56"
                draw.text(
                    (text_x + 1, text_y + 2),
                    label,
                    fill=shadow,
                    font=font,
                )
                draw.text(
                    (text_x, text_y),
                    label,
                    fill=text_color,
                    font=font,
                )
    return image


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_path in _FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw,
    label: str,
    max_width: float,
    max_height: float,
    initial_size: int,
) -> ImageFont.ImageFont:
    for size in range(initial_size, 15, -2):
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font
    return _load_font(16)
