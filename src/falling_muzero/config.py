from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class GameConfig:
    kind: str = "catch"
    width: int = 5
    height: int = 5
    paddle_width: int = 1
    episode_length: int = 40
    catch_reward: float = 1.0
    miss_reward: float = -1.0
    distance_shaping: float = 0.03
    spawn_seed: int = 2
    history_length: int = 4
    size: int = 4
    max_tile_exponent: int = 11
    invalid_move_penalty: float = -0.2
    merge_reward_scale: float = 100.0
    heuristic_search_depth: int = 1


@dataclass(slots=True)
class NetworkConfig:
    architecture: str = "conv"
    latent_dim: int = 64
    hidden_dim: int = 128


@dataclass(slots=True)
class MCTSConfig:
    simulations: int = 25
    max_depth: int = 5
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    temperature: float = 1.0


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 7
    episodes: int = 120
    bootstrap_episodes: int = 20
    bootstrap_imitation_steps: int = 0
    bootstrap_gradient_steps: int = 80
    bootstrap_policy: str = "heuristic"
    train_after_episodes: int = 4
    train_every: int = 1
    gradient_steps: int = 8
    batch_size: int = 32
    buffer_size: int = 500
    replay_priority_alpha: float = 0.0
    replay_priority_epsilon: float = 0.001
    unroll_steps: int = 4
    discount: float = 0.95
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    value_loss_weight: float = 0.5
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    search_value_target_weight: float = 0.5
    gradient_clip: float = 5.0
    eval_every: int = 20
    eval_episodes: int = 20
    mcts_eval_episodes: int = 0
    checkpoint_path: str = "artifacts/checkpoints/best.pt"
    final_checkpoint_path: str = "artifacts/checkpoints/final.pt"
    metrics_path: str = "artifacts/results/training_metrics.json"


@dataclass(slots=True)
class VisualizationConfig:
    plots_dir: str = "artifacts/plots"
    demo_dir: str = "artifacts/demo"
    render_frames: bool = True


@dataclass(slots=True)
class AppConfig:
    game: GameConfig = field(default_factory=GameConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _merge_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {
        field_name: _dataclass_to_dict(getattr(obj, field_name))
        if hasattr(getattr(obj, field_name), "__dataclass_fields__")
        else getattr(obj, field_name)
        for field_name in obj.__dataclass_fields__
    }


def _build_config(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        game=GameConfig(**data.get("game", {})),
        network=NetworkConfig(**data.get("network", {})),
        mcts=MCTSConfig(**data.get("mcts", {})),
        training=TrainingConfig(**data.get("training", {})),
        visualization=VisualizationConfig(**data.get("visualization", {})),
    )


def load_config(path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load the single project configuration file and optional in-memory overrides."""

    defaults = _dataclass_to_dict(AppConfig())
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            file_data = yaml.safe_load(handle) or {}
        defaults = _merge_dict(defaults, file_data)
    if overrides:
        defaults = _merge_dict(defaults, overrides)
    return _build_config(defaults)


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
