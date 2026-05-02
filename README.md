# MuZero Falling Catch

This project is a compact MuZero-style implementation for a tiny deterministic
grid arcade game called **Falling Catch**. A red object falls down the grid while
the player moves a blue paddle left, stays, or moves right. The reward is
positive for catching the object and negative for missing it, with a small
distance-shaping term to make short CPU training runs more informative.

The code is intentionally modular:

- `falling_muzero.game`: real game simulator and state manager
- `falling_muzero.mcts`: u-MCTS over learned abstract states
- `falling_muzero.networks`: representation, dynamics, and prediction networks
- `falling_muzero.replay_buffer`: episode history and BPTT training batches
- `falling_muzero.trainer`: self-play, training, evaluation, and checkpoints
- `falling_muzero.visualization`: reward/loss plots and gameplay frames

## Setup

The local environment already has the required packages. If needed:

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the project root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m falling_muzero.cli smoke-test
```

## Train

```bash
PYTHONPATH=src python3 -m falling_muzero.cli train
```

Useful shorter run:

```bash
PYTHONPATH=src python3 -m falling_muzero.cli train --episodes 30 --simulations 10
```

Training writes:

- best evaluated actor checkpoint: `artifacts/checkpoints/best.pt`
- final training checkpoint: `artifacts/checkpoints/final.pt`
- metrics: `artifacts/results/training_metrics.json`
- plots: `artifacts/plots/training_rewards.png` and `artifacts/plots/loss_curves.png`

## Evaluate

```bash
PYTHONPATH=src python3 -m falling_muzero.cli evaluate --mode actor --episodes 50
PYTHONPATH=src python3 -m falling_muzero.cli evaluate --mode random --episodes 50
PYTHONPATH=src python3 -m falling_muzero.cli evaluate --mode heuristic --episodes 50
```

`actor` uses only the representation and prediction networks after training.
`mcts` can also be evaluated, but it is slower because it searches at every move.
By default, evaluation loads `artifacts/checkpoints/best.pt`, not the final
checkpoint, because later self-play updates can sometimes degrade a policy.

## Demo Assets

```bash
PYTHONPATH=src python3 -m falling_muzero.cli demo --mode actor --frames 16 --gif-frames 80
```

This saves gameplay frames, a contact sheet, and a simple MCTS root-policy
diagram under `artifacts/demo/`. It also writes an animated GIF such as
`artifacts/demo/actor_performance.gif`.

## Tests

```bash
pytest
```

The tests cover the game rules, replay-buffer targets, network output shapes,
MCTS policy validity, and a tiny end-to-end training/checkpoint smoke run.

## Configuration

All pivotal parameters are in `configs/default.yaml`: grid size, episode length,
history length, MCTS simulations, network size, optimizer settings, replay-buffer
settings, output paths, and visualization directories. The default run includes
a short heuristic bootstrap phase before MuZero self-play. This makes the tiny
CPU-only demonstration more stable while still training the MuZero trinet and
using learned abstract-state MCTS during the main episode loop.
