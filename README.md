# MuZero Falling Catch

This project is a compact MuZero-style implementation for a tiny deterministic
grid arcade game called **Falling Catch**. A red object falls down the grid while
the player moves a blue paddle left, stays, or moves right. The reward is
positive for catching the object and negative for missing it, with a small
distance-shaping term to make short CPU training runs more informative.

The same MuZero pipeline also supports a deterministic **2048** environment.
Use `--game 2048` to train, evaluate, and render 2048 runs. Falling Catch
remains the default and the assignment-ready baseline.

The code is intentionally modular:

- `falling_muzero.games.falling_catch`: Falling Catch simulator and state manager
- `falling_muzero.games.game_2048`: deterministic 2048 simulator
- `falling_muzero.game_factory`: game selection for shared MuZero training
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

Run commands from the project root:

```bash
python smoke_test.py --game catch
python smoke_test.py --game 2048
```

The old module CLI still works for compatibility, but the top-level scripts set
up `src` automatically so you do not need to write `PYTHONPATH=src`.

## Training Recipes

The project has three useful model setups. Run commands from the project root.

### Falling Catch, default warm-start

This is the assignment-ready baseline. It uses the simple Falling Catch game and
a short heuristic bootstrap phase before MuZero self-play.

```bash
python train.py --game catch
python evaluate.py --game catch --mode actor --episodes 50
python evaluate.py --game catch --mode random --episodes 50
python evaluate.py --game catch --mode heuristic --episodes 50
python plot.py --game catch
python demo.py --game catch --mode actor --frames 16 --gif-frames 80
```

Outputs:

- checkpoint: `artifacts/checkpoints/best.pt`
- final checkpoint: `artifacts/checkpoints/final.pt`
- metrics: `artifacts/results/training_metrics.json`
- plots: `artifacts/plots/`
- demo/GIF: `artifacts/demo/`

Falling Catch plot outputs include:

- `training_rewards.png`: reward during training and evaluation
- `miss_percentage_progress.png`: how often the agent misses catch opportunities
- `miss_percentage_eval_only.png`: cleaner miss percentage plot without self-play episodes
- `heuristic_agreement_progress.png`: how often actions match the simple heuristic
- `heuristic_agreement_eval_only.png`: cleaner heuristic-agreement plot without self-play episodes
- `falling_catch_summary.png`: reward, miss rate, catches, and heuristic agreement in one figure
- `loss_curves.png`: policy/value/reward/total losses

### Falling Catch, no heuristic warm-start

This keeps the same game but removes heuristic episode data. It uses random
warm-up, prioritized replay, and shallow MCTS to test whether MuZero/self-play
learns from non-heuristic experience.

```bash
python train.py --game catch_no_heuristic
python evaluate.py --game catch_no_heuristic --mode actor --episodes 50
python evaluate.py --game catch_no_heuristic --mode random --episodes 50
python evaluate.py --game catch_no_heuristic --mode heuristic --episodes 50
python plot.py --game catch_no_heuristic
python demo.py --game catch_no_heuristic --mode actor --frames 16 --gif-frames 80
```

Outputs:

- checkpoint: `artifacts/catch_no_heuristic/checkpoints/best.pt`
- final checkpoint: `artifacts/catch_no_heuristic/checkpoints/final.pt`
- metrics: `artifacts/catch_no_heuristic/results/training_metrics.json`
- plots: `artifacts/catch_no_heuristic/plots/`
- demo/GIF: `artifacts/catch_no_heuristic/demo/`

### 2048 extension

This is the harder deterministic 2048 experiment. It is useful as an extension,
but Falling Catch remains the simpler assignment baseline.

```bash
python train.py --game 2048
python evaluate.py --game 2048 --mode actor --episodes 20
python evaluate.py --game 2048 --mode random --episodes 20
python evaluate.py --game 2048 --mode heuristic --episodes 5
python plot.py --game 2048
python demo.py --game 2048 --mode actor --gif-frames 1000 --gif-duration-ms 35 --demo-steps 1000
python demo.py --game 2048 --mode random --gif-frames 1000 --gif-duration-ms 35 --gif-final-pause-ms 1800 --demo-steps 1000
python demo.py --game 2048 --mode heuristic --gif-frames 1300 --gif-duration-ms 25 --gif-final-pause-ms 1800 --demo-steps 1200
```

Outputs:

- checkpoint: `artifacts/2048/checkpoints/best.pt`
- final checkpoint: `artifacts/2048/checkpoints/final.pt`
- metrics: `artifacts/2048/results/training_metrics.json`
- plots: `artifacts/2048/plots/`
- demo/GIF: `artifacts/2048/demo/`

Useful shorter runs while testing:

```bash
python train.py --game catch --episodes 30 --simulations 10
python train.py --game catch_no_heuristic --episodes 40 --simulations 10
python train.py --game 2048 --episodes 20 --simulations 5
```

The 2048 plot directory includes:

- `training_rewards.png`: reward during training and evaluation
- `loss_curves.png`: policy/value/reward/total losses
- `raw_score_progress.png`: raw 2048 score over training
- `max_tile_progress.png`: max tile reached, with a 1024 target line
- `survival_steps_progress.png`: how many moves the agent survives
- `performance_summary.png`: final/best actor compared with random and heuristic baselines

`configs/catch_no_heuristic.yaml` keeps Falling Catch but removes heuristic
warm-start data. It uses random replay warm-up, prioritized replay, and shallow
MCTS so the run tests whether MuZero/self-play learns a policy without copying
the simple paddle-tracking heuristic. Its outputs go under
`artifacts/catch_no_heuristic/`.

## Evaluate

```bash
python evaluate.py --game catch --mode actor --episodes 50
python evaluate.py --game catch --mode random --episodes 50
python evaluate.py --game catch --mode heuristic --episodes 50
python evaluate.py --game 2048 --mode actor --episodes 20
python evaluate.py --game catch_no_heuristic --mode actor --episodes 50
```

`actor` uses only the representation and prediction networks after training.
`mcts` can also be evaluated, but it is slower because it searches at every move.
By default, evaluation loads `artifacts/checkpoints/best.pt`, not the final
checkpoint, because later self-play updates can sometimes degrade a policy.
For 2048, evaluation also prints raw score, max tile, survival steps, and the
percentage of episodes reaching at least 1024.
For Falling Catch, evaluation also prints catches, misses, miss percentage, and
heuristic agreement. The no-heuristic configuration is the cleanest evidence
that the learned actor is not only replaying a warm-start heuristic.

## Demo Assets

```bash
python demo.py --game catch --mode actor --frames 16 --gif-frames 80
python demo.py --game 2048 --mode actor --frames 16
```

This saves gameplay frames, a contact sheet, and a simple MCTS root-policy
diagram under the configured demo directory. It also writes an animated GIF such
as `artifacts/demo/actor_performance.gif` or
`artifacts/2048/demo/actor_performance.gif`. The 2048 demo defaults to a
longer and faster GIF: up to 600 frames at 45 ms per frame, and it extends the
demo episode length so the GIF shows one longer game. GIF export shows a single
episode, stops when the agent loses, and pauses briefly on the final frame.
`--gif-frames` is only an upper limit. Override it when needed:

```bash
python demo.py --game 2048 --mode actor --gif-frames 1000 --gif-duration-ms 35 --gif-final-pause-ms 1800 --demo-steps 1000
python demo.py --game 2048 --mode random --gif-frames 1000 --gif-duration-ms 35 --gif-final-pause-ms 1800 --demo-steps 1000
python demo.py --game 2048 --mode heuristic --gif-frames 1300 --gif-duration-ms 25 --gif-final-pause-ms 1800 --demo-steps 1200
```

## Tests

```bash
pytest
```

The tests cover the game rules, replay-buffer targets, network output shapes,
MCTS policy validity, and a tiny end-to-end training/checkpoint smoke run.

## Configuration

All Falling Catch parameters are in `configs/default.yaml`; 2048 parameters are
in `configs/2048.yaml`. They cover game choice, board size, episode length,
history length, MCTS simulations, network size, optimizer settings,
replay-buffer settings, output paths, and visualization directories. The
Falling Catch defaults include a short heuristic bootstrap phase before MuZero
self-play. The 2048 default is intentionally stricter: it does not imitate the
2048 heuristic by default, and the heuristic is kept as an evaluation baseline.
The default representation network is now convolutional (`network.architecture:
conv`) because the games are grid based. The older MLP representation remains
available with `network.architecture: mlp`, but checkpoints are architecture
specific and must be retrained after changing this setting.

Value targets blend discounted returns with actual MCTS root values when those
root values exist:

```text
target_value = (1 - search_value_target_weight) * return + search_value_target_weight * mcts_root_value
```

Random and heuristic bootstrap episodes do not have MCTS root values, so they
fall back to discounted returns only.

The 2048 observation includes the tile grid plus deterministic state metadata
planes for step progress and the spawn schedule. This avoids presenting a
heuristic clone as a learned MuZero result and gives the learned dynamics model
more of the information needed to predict deterministic tile spawns. The 2048
configuration also uses random replay warm-up, prioritized replay, and shallow
MCTS by default; deeper learned-model rollouts were unstable on this small CPU
implementation because dynamics errors compound quickly.

The additional Falling Catch no-heuristic configuration is useful for checking
that the MuZero pipeline can improve from random experience. It usually does not
reach the perfect heuristic score as reliably as the default warm-start setup,
but it provides a cleaner demonstration that the learned actor can outperform a
random baseline without heuristic episode data.
