"""Top-level entry point: regenerate plots from a saved ``training_metrics.json`` file."""

from __future__ import annotations

from script_utils import run_cli_command


if __name__ == "__main__":
    raise SystemExit(run_cli_command("plot"))
