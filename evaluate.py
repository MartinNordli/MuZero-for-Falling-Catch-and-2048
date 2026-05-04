"""Top-level entry point: evaluate a trained checkpoint or a non-learning baseline."""

from __future__ import annotations

from script_utils import run_cli_command


if __name__ == "__main__":
    raise SystemExit(run_cli_command("evaluate"))
