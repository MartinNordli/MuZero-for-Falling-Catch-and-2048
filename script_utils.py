"""Shared dispatch layer for the top-level wrapper scripts.

``train.py``, ``evaluate.py``, ``demo.py``, ``plot.py`` and ``smoke_test.py``
all delegate here. The wrapper scripts exist so users can run
``python train.py --game catch`` from the project root without having to set
``PYTHONPATH=src``.

This module:
1. prepends ``src/`` to ``sys.path`` so ``import falling_muzero`` works,
2. resolves the ``--game`` shortcut into a YAML config path, and
3. forwards the remaining flags to :mod:`falling_muzero.cli`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Shorthand mapping from the user-facing ``--game`` value to a YAML preset.
# ``catch-no-heuristic`` is an alias because both spellings appear in the
# READMEs and slide deck.
GAME_CONFIGS = {
    "catch": "configs/default.yaml",
    "catch_no_heuristic": "configs/catch_no_heuristic.yaml",
    "catch-no-heuristic": "configs/catch_no_heuristic.yaml",
    "2048": "configs/2048.yaml",
}


def run_cli_command(command: str, argv: list[str] | None = None) -> int:
    """Run the package CLI with a short top-level script interface."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser(command)
    args = parser.parse_args(raw_args)

    selected_game = args.game or "catch"
    config_path = args.config or GAME_CONFIGS[selected_game]
    cli_args = ["--config", config_path]

    if args.game in {"catch", "2048"}:
        cli_args.extend(["--game", args.game])

    cli_args.append(command)
    cli_args.extend(_command_args(command, args))

    from falling_muzero.cli import main

    return main(cli_args)


def _build_parser(command: str) -> argparse.ArgumentParser:
    """Build the small wrapper-level argument parser for ``command`` (one parser per top-level script)."""

    script_name = "smoke_test.py" if command == "smoke-test" else f"{command}.py"
    parser = argparse.ArgumentParser(prog=script_name)
    parser.add_argument(
        "--game",
        choices=tuple(GAME_CONFIGS),
        default=None,
        help="Preset to run. Defaults to catch unless --config is supplied.",
    )
    parser.add_argument("--config", default=None, help="Optional explicit YAML config path.")

    if command == "train":
        parser.add_argument("--episodes", type=int, default=None)
        parser.add_argument("--simulations", type=int, default=None)
    elif command == "evaluate":
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--mode", choices=["actor", "mcts", "random", "heuristic"], default="actor")
        parser.add_argument("--episodes", type=int, default=20)
    elif command == "demo":
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--mode", choices=["actor", "mcts", "random", "heuristic"], default="actor")
        parser.add_argument("--frames", type=int, default=16)
        parser.add_argument("--gif-frames", type=int, default=None)
        parser.add_argument("--gif-duration-ms", type=int, default=None)
        parser.add_argument("--gif-final-pause-ms", type=int, default=None)
        parser.add_argument("--demo-steps", type=int, default=None)
    elif command == "plot":
        parser.add_argument("--metrics", default=None)
    elif command != "smoke-test":
        raise ValueError(f"unknown script command {command!r}")

    return parser


def _command_args(command: str, args: argparse.Namespace) -> list[str]:
    """Convert the parsed wrapper args into the argv list expected by the package CLI."""

    if command == "train":
        output: list[str] = []
        if args.episodes is not None:
            output.extend(["--episodes", str(args.episodes)])
        if args.simulations is not None:
            output.extend(["--simulations", str(args.simulations)])
        return output

    if command == "evaluate":
        output = ["--mode", args.mode, "--episodes", str(args.episodes)]
        if args.checkpoint is not None:
            output.extend(["--checkpoint", args.checkpoint])
        return output

    if command == "demo":
        output = ["--mode", args.mode, "--frames", str(args.frames)]
        if args.checkpoint is not None:
            output.extend(["--checkpoint", args.checkpoint])
        if args.gif_frames is not None:
            output.extend(["--gif-frames", str(args.gif_frames)])
        if args.gif_duration_ms is not None:
            output.extend(["--gif-duration-ms", str(args.gif_duration_ms)])
        if args.gif_final_pause_ms is not None:
            output.extend(["--gif-final-pause-ms", str(args.gif_final_pause_ms)])
        if args.demo_steps is not None:
            output.extend(["--demo-steps", str(args.demo_steps)])
        return output

    if command == "plot":
        return [] if args.metrics is None else ["--metrics", args.metrics]

    if command == "smoke-test":
        return []

    raise ValueError(f"unknown script command {command!r}")
