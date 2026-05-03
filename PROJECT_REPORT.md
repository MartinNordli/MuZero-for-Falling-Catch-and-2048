# falling-muzero

> A compact MuZero-style implementation for two small deterministic grid games — Falling Catch and 2048 — designed for short CPU training runs and assignment-style demonstration.

| Field | Value |
| --- | --- |
| Version / commit | `0.1.0` / `0488263` |
| Primary language | `Python (>= 3.10)` |
| Status | `Course project, single contributor, active` |
| License | `No LICENSE file present (unverified)` |
| Platform(s) | `CPU only (PyTorch device pinned to CPU in trainer.py:43)` |
| Last updated | `2026-05-03` |

## Executive summary

This project is a small, fully self-contained MuZero-style learner for two deterministic grid games — Falling Catch (5×5, 3 actions) and 2048 (4×4, 4 actions) — sharing a single trainer, MCTS, network and replay buffer. The architecture is the textbook MuZero "Trinet" (representation, dynamics, prediction) with MCTS performed over learned latent states, and three pre-tuned configurations cover the assignment baseline, a heuristic-free Falling Catch ablation, and a harder 2048 extension. On the assignment baseline the actor reaches the heuristic ceiling (`9.17` reward); the no-heuristic ablation still reaches `8.78` from random warm-up only, with `0.0%` miss rate at its best run; the 2048 extension clearly beats random play (`43.0` vs `11.9` reward, max tile `512`) but stays well below the lookahead heuristic (`183.2`) — see [Results](#results) for plots and GIFs. Test coverage exists at the unit level for game rules, network shapes, MCTS policy validity, replay-buffer BPTT targets, and an end-to-end smoke run for both games.

## Table of contents

- [Purpose and audience](#purpose-and-audience)
- [Project overview](#project-overview)
- [Getting started](#getting-started)
- [Architecture](#architecture)
- [Directory structure](#directory-structure)
- [Configuration](#configuration)
- [Usage examples](#usage-examples)
- [API and public interfaces](#api-and-public-interfaces)
- [Quality, tests and validation](#quality-tests-and-validation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [References](#references)

## Purpose and audience

The project implements the MuZero algorithm in a deliberately small form so that the full self-play / MCTS / learned-dynamics loop can be trained, evaluated, and visualized on a CPU within minutes rather than hours. Falling Catch is the assignment-ready baseline; 2048 is included as a harder deterministic extension that exercises the same pipeline on a more complex state space. The deterministic spawn schedule in both games keeps the MuZero search tree a normal action tree without stochastic chance nodes (`src/falling_muzero/games/falling_catch.py:90`, `src/falling_muzero/games/game_2048.py:308`).

Audiences:

- **Course graders / readers** verifying the assignment baseline runs end-to-end and produces the expected plots, GIFs, and metrics.
- **Students learning MuZero** who want a small, readable reference implementation of the Trinet, MCTS over latent states, and BPTT replay targets.
- **Future maintainer (the author)** extending the trainer to additional games via the `MuZeroGame` Protocol and the `game_factory.create_game` dispatch.

## Project overview

| Area | What it does | Where in the repo |
| --- | --- | --- |
| Game environments | Deterministic Falling Catch and 2048 simulators implementing the `MuZeroGame` Protocol | `src/falling_muzero/games/` |
| MuZero networks | Representation / dynamics / prediction Trinet (MLP or conv) | `src/falling_muzero/networks.py` |
| MCTS | u-MCTS over learned abstract latent states with Dirichlet exploration noise | `src/falling_muzero/mcts.py` |
| Replay buffer | Episode storage, BPTT batch construction, optional reward-weighted prioritization | `src/falling_muzero/replay_buffer.py` |
| Trainer | Self-play loop, bootstrap policies, gradient steps, evaluation, checkpointing, metrics | `src/falling_muzero/trainer.py` |
| Visualization | Reward/loss/miss-rate/score/max-tile plots, demo frames, animated GIFs, MCTS root diagram | `src/falling_muzero/visualization.py` |
| CLI | argparse-based subcommands `train`, `evaluate`, `demo`, `plot`, `smoke-test` | `src/falling_muzero/cli.py` |
| Top-level scripts | Thin wrappers so users can run `python train.py`, etc., without setting `PYTHONPATH` | `train.py`, `evaluate.py`, `demo.py`, `plot.py`, `smoke_test.py`, `script_utils.py` |
| Configs | Three YAML presets: default Falling Catch, heuristic-free Falling Catch, 2048 | `configs/` |

## Getting started

### Requirements

- Python 3.10 or newer (`pyproject.toml:9`)
- The packages listed in `requirements.txt`: `numpy`, `torch`, `matplotlib`, `PyYAML`, `pytest`
- `Pillow` is also imported by `visualization.py` for GIF and contact-sheet rendering (transitively pulled in by the matplotlib stack in this environment, but not declared explicitly — `unverified` whether a fresh install gets it without manual install)

### Install

```bash
python3 -m pip install -r requirements.txt
```

### Run

```bash
# Falling Catch (assignment baseline, with heuristic warm-start)
python train.py --game catch
python evaluate.py --game catch --mode actor --episodes 50
python plot.py --game catch
python demo.py --game catch --mode actor --frames 16 --gif-frames 80

# 2048 extension
python train.py --game 2048
python evaluate.py --game 2048 --mode actor --episodes 20
```

### Verify

```bash
# Tiny end-to-end pass (writes to artifacts/smoke/<game>/)
python smoke_test.py --game catch
python smoke_test.py --game 2048

# Unit tests
pytest
```

## Architecture

The system is organized as a clean four-layer pipeline. The CLI parses arguments, loads a YAML config, and constructs a single `Trainer`. The trainer owns a game (selected by `game_factory.create_game`), a `MuZeroNetwork`, an `EpisodeBuffer`, and a `MuZeroMCTS`. Self-play episodes are collected by running MCTS at every step; episodes are stored in the buffer; gradient steps unroll the network over `unroll_steps` actions and compare against BPTT targets blending discounted returns with MCTS root values when available. Architecture below is **inferred from code** unless otherwise noted.

```mermaid
flowchart TD
    User[User] --> Scripts[Top-level scripts<br/>train.py / evaluate.py / demo.py / plot.py / smoke_test.py]
    Scripts --> CLI[falling_muzero.cli.main]
    CLI --> Cfg[load_config<br/>configs/*.yaml]
    CLI --> Trainer[Trainer]
    Trainer --> Factory[game_factory.create_game]
    Factory --> Catch[FallingCatchGame]
    Factory --> G2048[Game2048]
    Trainer --> Net[MuZeroNetwork<br/>Representation+Dynamics+Prediction]
    Trainer --> MCTS[MuZeroMCTS<br/>over learned latents]
    Trainer --> Buf[EpisodeBuffer<br/>BPTT targets]
    Trainer --> Viz[visualization]
    Viz --> Artifacts[(artifacts/...<br/>checkpoints, metrics, plots, GIFs)]
```

### Components

| Component | Responsibility | Key files | Dependencies |
| --- | --- | --- | --- |
| `MuZeroGame` (Protocol) | Common contract for environments: action space, observations, legal actions, heuristic, ASCII render | `src/falling_muzero/games/types.py` | `numpy` |
| `FallingCatchGame` | Deterministic 5×5 falling-object simulator with paddle, distance shaping, deterministic spawn schedule | `src/falling_muzero/games/falling_catch.py` | `numpy` |
| `Game2048` | Deterministic 4×4 2048 with one-hot tile planes + spawn-metadata planes; static and lookahead heuristic baselines | `src/falling_muzero/games/game_2048.py` | `numpy` |
| `create_game` | Dispatches `GameConfig.kind` to one of the two games | `src/falling_muzero/game_factory.py` | both games |
| `MuZeroNetwork` | Trinet: representation (MLP or conv), dynamics, prediction; supports `initial_inference` and `recurrent_inference` | `src/falling_muzero/networks.py` | `torch` |
| `MuZeroMCTS` | u-MCTS over learned latents; PUCT selection, Dirichlet noise at root, optional legal-action mask | `src/falling_muzero/mcts.py` | `torch`, `numpy`, `MuZeroNetwork` |
| `Episode` / `EpisodeBuffer` | Episode storage; samples BPTT batches; blends discounted returns with MCTS root values; optional reward-weighted priority | `src/falling_muzero/replay_buffer.py` | `numpy`, `MuZeroGame` |
| `Trainer` | Bootstraps replay (heuristic or random), runs MCTS self-play, gradient steps, eval (actor/MCTS/random/heuristic), checkpointing | `src/falling_muzero/trainer.py` | all of the above |
| `visualization` | Reward/loss/miss-rate/agreement/score/max-tile plots; demo frames; animated GIFs; MCTS root-policy diagram | `src/falling_muzero/visualization.py` | `matplotlib`, `Pillow` |
| `AppConfig` / `load_config` | Dataclass-based typed config layered over YAML defaults plus runtime overrides | `src/falling_muzero/config.py` | `PyYAML` |
| `cli.main` | argparse subcommands `train` / `evaluate` / `demo` / `plot` / `smoke-test` | `src/falling_muzero/cli.py` | `Trainer`, `visualization`, `load_config` |

### Sequence — one self-play training step

```mermaid
sequenceDiagram
    participant T as Trainer
    participant G as Game
    participant M as MCTS
    participant N as Network
    participant B as EpisodeBuffer

    T->>G: reset()
    loop for each step until done
        T->>G: stack_observations(...)
        T->>M: search(stack, exploration=True, legal_actions)
        M->>N: initial_inference(obs)
        M->>N: recurrent_inference(latent, action) (per simulation)
        M-->>T: SearchResult(policy, value)
        T->>G: step(action)
        G-->>T: StepResult(state, observation, reward, done)
    end
    T->>B: add(episode)
    T->>B: sample(batch_size, unroll_steps)
    B-->>T: TrainingBatch(obs, actions, target_*, masks)
    T->>N: initial_inference + recurrent_inference unrolled
    N-->>T: predicted policy / value / reward
    T->>T: loss + backward + optimizer.step()
```

## Directory structure

```text
MuZero Pong/
├── README.md                  # User-facing setup, training recipes, evaluation
├── VIDEO_MANUSCRIPT.md        # Demo/video script (untracked)
├── VIDEO_OUTLINE.md           # Earlier outline (gitignored)
├── pyproject.toml             # Build + console script (falling-muzero)
├── requirements.txt           # Runtime + test deps
├── train.py / evaluate.py / demo.py / plot.py / smoke_test.py
├── script_utils.py            # Shared CLI dispatch for top-level scripts
├── configs/
│   ├── default.yaml           # Falling Catch baseline (heuristic warm-start)
│   ├── catch_no_heuristic.yaml# Same game, random warm-up + prioritized replay
│   └── 2048.yaml              # 2048 extension
├── src/falling_muzero/
│   ├── cli.py
│   ├── config.py
│   ├── game.py / game_2048.py / game_factory.py
│   ├── games/                 # Actual game implementations + Protocol
│   ├── mcts.py
│   ├── networks.py
│   ├── replay_buffer.py
│   ├── trainer.py
│   └── visualization.py
├── tests/                     # pytest suite
└── artifacts/                 # Generated: checkpoints, metrics, plots, demos, GIFs
```

| Path | Description |
| --- | --- |
| `src/falling_muzero/` | Library code; pip-installable as the `falling-muzero` package |
| `src/falling_muzero/games/` | Per-game simulators and the shared `MuZeroGame` Protocol |
| `src/falling_muzero/game.py`, `game_2048.py` | Top-level package modules of 311 / 236 bytes; presumed re-exports for backward compatibility (`unverified` — not opened during this audit) |
| `tests/` | Unit + smoke tests; runs via `pytest` (config in `pyproject.toml:23-26`) |
| `configs/` | Three YAML presets covering the documented training recipes |
| `artifacts/` | All generated outputs; gitignored (`.gitignore:6`); per-game subdirectories under `2048/`, `catch_no_heuristic/`, etc. |

## Configuration

Configuration is layered: dataclass defaults in `src/falling_muzero/config.py` are deep-merged with the chosen YAML, then with runtime CLI overrides (`config.py:124-135`). Three presets ship with the project:

| Preset | File | Purpose |
| --- | --- | --- |
| `catch` (default) | `configs/default.yaml` | Falling Catch with 20-episode heuristic warm-start, 25 MCTS sims, 120 self-play episodes |
| `catch_no_heuristic` / `catch-no-heuristic` | `configs/catch_no_heuristic.yaml` | Falling Catch with random warm-up, prioritized replay (`alpha=0.8`), shallow MCTS (`max_depth=1`), 400 episodes |
| `2048` | `configs/2048.yaml` | 4×4 2048, 400 episodes, larger conv net (latent 96 / hidden 160), prioritized replay, 20 sims with `max_depth=1` |

Selected key fields (full set in `src/falling_muzero/config.py`):

| Section | Field | Default | Notes |
| --- | --- | --- | --- |
| `game` | `kind` | `"catch"` | One of `catch`, `2048` (validated by `game_factory:14-18`) |
| `game` | `history_length` | `4` | Number of stacked frames in observations |
| `network` | `architecture` | `"conv"` | `mlp` or `conv`; checkpoints are architecture-specific |
| `network` | `latent_dim` / `hidden_dim` | `64` / `128` | Trinet sizing |
| `mcts` | `simulations` | `25` | Per-step search budget during self-play |
| `mcts` | `max_depth` | `5` | Bounded tree depth; `2048.yaml` uses `1` (shallow) |
| `training` | `bootstrap_policy` | `"heuristic"` | Or `"random"`; warm-up episodes added to replay before self-play |
| `training` | `replay_priority_alpha` | `0.0` | Above 0 enables reward-weighted episode prioritization |
| `training` | `search_value_target_weight` | `0.5` | Blend factor between discounted return and MCTS root value |
| `training` | `unroll_steps` | `4` | BPTT depth for dynamics/prediction unroll |

There are no environment variables. The visualization module sets `MPLCONFIGDIR` and `XDG_CACHE_HOME` to subdirectories of `artifacts/` (`visualization.py:13-14`) so matplotlib does not write to the user's home directory.

## Usage examples

### Minimal usage

```bash
python smoke_test.py --game catch
```

### Programmatic usage

```python
from falling_muzero import load_config
from falling_muzero.trainer import Trainer

config = load_config("configs/default.yaml")
trainer = Trainer(config)
metrics = trainer.train()
print("best actor reward:", trainer.best_actor_eval)
```

### Expected output (training)

```text
game: catch
saved checkpoint: artifacts/checkpoints/best.pt
saved plots:
  artifacts/plots/training_rewards.png
  artifacts/plots/loss_curves.png
  ...
```

## API and public interfaces

The package is a CLI / library, not a service. There is no HTTP API and no OpenAPI spec.

### CLI surface (`falling-muzero` console script and equivalent `python -m` invocations)

| Subcommand | Purpose | Key flags |
| --- | --- | --- |
| `train` | Run MuZero self-play and save checkpoints/metrics | `--episodes`, `--simulations` |
| `evaluate` | Evaluate a checkpoint or a non-learning baseline | `--mode {actor,mcts,random,heuristic}`, `--episodes`, `--checkpoint` |
| `demo` | Render gameplay frames, contact sheet, GIF and MCTS root diagram | `--mode`, `--frames`, `--gif-frames`, `--gif-duration-ms`, `--gif-final-pause-ms`, `--demo-steps` |
| `plot` | Re-generate plots from saved metrics | `--metrics` |
| `smoke-test` | Tiny end-to-end run with reduced settings | (no extra flags) |

Top-level wrappers `train.py`, `evaluate.py`, `demo.py`, `plot.py`, `smoke_test.py` accept a `--game {catch, catch_no_heuristic, 2048}` shortcut and dispatch into the same CLI (`script_utils.py:14-41`).

### Library surface

Re-exported from `falling_muzero` (`src/falling_muzero/__init__.py:3-15`):

| Name | Type | Purpose |
| --- | --- | --- |
| `AppConfig` | dataclass | Top-level typed config |
| `load_config(path, overrides)` | function | Layered YAML + override loader |
| `FallingCatchGame`, `Game2048` | class | Game simulators |
| `MuZeroGame` | Protocol | Common interface used by trainer / MCTS / buffer |
| `GameState`, `Game2048State`, `StepResult` | dataclass | Frozen state and step-result types |

`Trainer` and `load_trainer_with_checkpoint` from `falling_muzero.trainer` are not re-exported by the package `__init__` but are the public entry points used by both the CLI and the smoke test.

## Quality, tests and validation

The test suite is pytest-based and covers game rules, replay-buffer mechanics, network output shapes, MCTS policy validity, and a tiny end-to-end training pass for both games. There is no CI, no coverage reporting tool configured, and no static analysis tool configured in `pyproject.toml`.

| Test type | Where | Command | Status |
| --- | --- | --- | --- |
| Unit — Falling Catch rules | `tests/test_game.py` | `pytest tests/test_game.py` | `unverified` (not run during this audit) |
| Unit — 2048 rules | `tests/test_game_2048.py` | `pytest tests/test_game_2048.py` | `unverified` |
| Unit — networks + MCTS | `tests/test_networks_mcts.py` | `pytest tests/test_networks_mcts.py` | `unverified` |
| Unit — replay buffer | `tests/test_replay_buffer.py` | `pytest tests/test_replay_buffer.py` | `unverified` |
| End-to-end smoke | `tests/test_smoke.py` | `pytest tests/test_smoke.py` | `unverified` |
| Lint | (none configured) | — | `not configured` |
| Type check | (none configured) | — | `not configured` |
| Docs tests | (none configured) | — | `not configured` |

| Measure | Value | Tool |
| --- | --- | --- |
| Line coverage | `unverified` (no coverage tool configured) | — |
| Branch coverage | `unverified` | — |
| Complexity | `unverified` (Radon / similar not configured) | — |
| Lint summary | `unverified` (no Ruff / Flake8 / Pylint / Black config in repo) | — |

Notable structural observations from the code (not from a metric tool):

- `trainer.py` is the largest module at ~520 lines and concentrates most orchestration logic. `_train_step` (`trainer.py:363-424`) and `_choose_action` (`trainer.py:201-232`) are the central control flow points.
- `visualization.py` is also large (~27 KB) because it owns plot definitions, demo frame rendering, GIF building and the MCTS diagram.
- `Game2048._evaluate_board` (`games/game_2048.py:248-273`) implements a hand-tuned snake/smoothness/corner heuristic that is the strongest non-learning baseline; it is intentionally kept out of the 2048 training warm-start (`bootstrap_policy: random` in `configs/2048.yaml:32`) so a learned actor cannot trivially imitate it.
- The package contains two small modules `src/falling_muzero/game.py` (311 B) and `src/falling_muzero/game_2048.py` (236 B) at the package root, distinct from the implementations under `src/falling_muzero/games/`. These are presumed re-exports for the "old module CLI" mentioned in `README.md:39-40`, but their contents were not opened during this audit — `unverified`.

## Results

All numbers below come directly from the saved `training_metrics.json` files. Plots and GIFs are committed under `artifacts/<preset>/` and embedded with relative paths so they render both on GitHub and when this file is exported to HTML/PDF.

### Falling Catch — default (heuristic warm-start)

Across 120 self-play episodes the actor matches the heuristic baseline exactly: actor evaluation reward `9.17` vs heuristic `9.17` vs random `−4.58` (best actor `9.17`, source `artifacts/results/training_metrics.json`). With heuristic warm-start data in the replay buffer, this is the easy regime — the learned policy reaches the ceiling immediately and stays there.

| Metric | Last | Best | Random baseline | Heuristic baseline |
| --- | --- | --- | --- | --- |
| Actor eval reward | `9.17` | `9.17` | `−4.58` | `9.17` |

![Falling Catch default — training reward over 120 episodes](artifacts/plots/training_rewards.png)

![Falling Catch default — policy/value/reward/total loss curves](artifacts/plots/loss_curves.png)

![Falling Catch default — actor playing one episode](artifacts/demo/actor_performance.gif)

![Falling Catch default — demo contact sheet (16 frames)](artifacts/demo/demo_contact_sheet.png)

![Falling Catch default — MCTS root policy after training](artifacts/demo/mcts_root_policy.png)

### Falling Catch — no heuristic warm-start (ablation)

Trained for 400 episodes from random warm-up only, with prioritized replay (`alpha=0.8`) and shallow MCTS (`max_depth=1`). The actor reaches `8.78` reward and a final miss percentage of `0.0%` over 20 evaluation episodes — clearly above the random baseline (`−3.97`) and within ~4% of the heuristic ceiling (`9.17`), with `57.5%` heuristic agreement showing the learned policy is not just copying the heuristic move-toward-ball rule. (Source: `artifacts/catch_no_heuristic/results/training_metrics.json`.)

| Metric | Last | Best | Random baseline | Heuristic baseline |
| --- | --- | --- | --- | --- |
| Actor eval reward | `8.78` | `8.78` | `−3.97` | `9.17` |
| Actor miss percentage | `0.0%` | `0.0%` (best run) | — | — |
| Heuristic-agreement (actor) | `57.5%` | — | — | — |

![Falling Catch (no heuristic) — training reward over 400 episodes](artifacts/catch_no_heuristic/plots/training_rewards.png)

![Falling Catch (no heuristic) — combined summary (reward, miss rate, catches, agreement)](artifacts/catch_no_heuristic/plots/falling_catch_summary.png)

![Falling Catch (no heuristic) — miss percentage at evaluation only](artifacts/catch_no_heuristic/plots/miss_percentage_eval_only.png)

![Falling Catch (no heuristic) — heuristic agreement at evaluation only](artifacts/catch_no_heuristic/plots/heuristic_agreement_eval_only.png)

![Falling Catch (no heuristic) — loss curves](artifacts/catch_no_heuristic/plots/loss_curves.png)

![Falling Catch (no heuristic) — actor playing one episode](artifacts/catch_no_heuristic/demo/actor_performance.gif)

![Falling Catch (no heuristic) — demo contact sheet](artifacts/catch_no_heuristic/demo/demo_contact_sheet.png)

### 2048 — extension

Trained for 1000 episodes with the conv Trinet (latent 96 / hidden 160) and shallow MCTS. The learned actor clearly beats random play (`43.0` vs `11.9` best reward) and reaches an average raw score of `5504` and a max tile of `512` at its best, but does not reach 1024 in the evaluated runs and remains well below the lookahead heuristic (`183.2` reward). This matches the README's framing of 2048 as the harder extension: dynamics errors compound quickly on this small CPU implementation. (Source: `artifacts/2048/results/training_metrics.json`.)

| Metric | Last | Best | Random baseline | Heuristic baseline |
| --- | --- | --- | --- | --- |
| Actor eval reward | `32.06` | `43.00` | `11.87` | `183.19` |
| Actor avg raw score | `4104` | `5504` | — | — |
| Actor max tile (best) | `256` | `512` | — | — |
| 1024-reached rate | `0%` | `0%` | — | — |
| Actor survival steps | `334` | `388` | — | — |

![2048 — training reward over 1000 episodes](artifacts/2048/plots/training_rewards.png)

![2048 — actor vs random vs heuristic performance summary](artifacts/2048/plots/performance_summary.png)

![2048 — raw score progression](artifacts/2048/plots/raw_score_progress.png)

![2048 — max tile reached over training](artifacts/2048/plots/max_tile_progress.png)

![2048 — survival steps per episode](artifacts/2048/plots/survival_steps_progress.png)

![2048 — loss curves](artifacts/2048/plots/loss_curves.png)

![2048 — heuristic playing one episode (best reference)](artifacts/2048/demo/heuristic_performance.gif)

![2048 — random baseline playing one episode](artifacts/2048/demo/random_performance.gif)

![2048 — demo contact sheet (16 frames)](artifacts/2048/demo/demo_contact_sheet.png)

![2048 — MCTS root policy after training](artifacts/2048/demo/mcts_root_policy.png)

> The directory `artifacts/2048/demo/MLP Actor/actor_performance.gif` contains an MLP-architecture actor demo from an earlier run. It is kept for comparison; the `conv` actor is the one whose metrics are reported above.

> Smoke-test artifacts under `artifacts/smoke/` exist to verify the pipeline end-to-end and are not included as results.

## Troubleshooting

| Problem | Likely cause | Solution |
| --- | --- | --- |
| `FileNotFoundError: checkpoint not found for <game> <mode> mode: ...` | Tried to evaluate or demo with `--mode actor` or `--mode mcts` before training | Run `python train.py --game <game>` first, or pass `--checkpoint <path>`, or use `--mode random` / `--mode heuristic` |
| `RuntimeError: checkpoint <path> is incompatible with the current game/network config` (`trainer.py:186-189`) | The saved checkpoint was trained with a different observation shape, action space, or network architecture (e.g. switched between `mlp` and `conv`) | Retrain with the new config, or revert the config change |
| `network.architecture must be 'mlp' or 'conv'` | Typo or unsupported value in the YAML | Set `network.architecture` to `mlp` or `conv` in the active config (`networks.py:46`) |
| `unknown game kind '...'; expected 'catch' or '2048'` | Bad `game.kind` in the config | Use `catch` or `2048` (`game_factory.py:18`) |
| Matplotlib writes config files outside the project | Default `MPLCONFIGDIR` not respected | Already handled by `visualization.py:13-14`, which redirects to `artifacts/.mplconfig` and `artifacts/.cache` |
| `python train.py` fails with `ModuleNotFoundError: falling_muzero` | Running from a different working directory | Run from the project root; the wrappers prepend `./src` to `sys.path` (`script_utils.py:8-11`) |

```bash
# Quick diagnostic: tiny end-to-end pass for both games
python smoke_test.py --game catch
python smoke_test.py --game 2048
pytest -q
```

## License

No `LICENSE` file is present in the repository. The intended license is `unverified`. If the project is to be shared beyond the course, a license should be added (e.g. MIT) and referenced here with its SPDX identifier.

## References

- [README](./README.md) — user-facing setup, training recipes, evaluation, configuration narrative
- [pyproject.toml](./pyproject.toml) — package metadata, console-script entry point, pytest config
- [requirements.txt](./requirements.txt) — runtime + test dependencies
- [configs/default.yaml](./configs/default.yaml) — Falling Catch baseline preset
- [configs/catch_no_heuristic.yaml](./configs/catch_no_heuristic.yaml) — heuristic-free Falling Catch ablation
- [configs/2048.yaml](./configs/2048.yaml) — 2048 extension preset
- [src/falling_muzero/cli.py](./src/falling_muzero/cli.py) — CLI entry point
- [src/falling_muzero/trainer.py](./src/falling_muzero/trainer.py) — main orchestration
- [tests/](./tests/) — pytest suite

<!-- FINAL CHECK
- Placeholders: none remaining.
- Optional sections deleted: Performance and benchmarks, Distribution and operations, Security, Contributing, Changelog, Badges (no CI to back them, no real attack surface, single-author course project, no release history).
- Results section embeds plots, GIFs, and contact sheets from each preset's artifacts/ folder, with real numbers pulled from training_metrics.json. Smoke artifacts are explicitly excluded.
- Relative links and image paths: all point to files that exist in this repo.
- Code blocks language-tagged: yes.
- Mermaid diagrams: two — flowchart TD and sequenceDiagram. Syntactically valid.
- Claims tied to evidence or marked unverified: yes (file paths and line numbers cited; missing evidence flagged explicitly).
- Documented vs inferred architecture: explicitly noted in the Architecture section.
- Executive summary: 4 sentences at top, includes headline numbers.
- Badges: none — no CI exists, so no live signals to display.
-->
