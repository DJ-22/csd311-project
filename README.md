# csd311-project

Dots-and-Boxes solvers for CSD311 Artificial Intelligence.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The `rules-prolog` player also needs [SWI-Prolog](https://www.swi-prolog.org/download/stable) (64-bit) with its `bin` folder on `PATH`. Every other player works without it.

## Test

```bash
python -m pytest -q
```

The Prolog tests skip when SWI-Prolog is not installed.

## Run a tournament

```bash
python -m harness.cli tournament --board 3x3 --games 100 --seed 0 --time-limit 1.0 \
    --players random,greedy,search,rules,mcts,mcts-flat
```

Results are written to `experiments/results/<name>_<timestamp>.csv` (`--name` sets the prefix).

| Player | Solver |
|---|---|
| `random`, `greedy` | 1: baselines |
| `exact` | 2: exact solver (2×2 and 2×3; slow on 3×3) |
| `search` | 3: heuristic search |
| `rules`, `rules-prolog` | 4: rule-based, Python or Prolog backend |
| `mcts`, `mcts-flat` | 5: UCT and flat Monte Carlo |
| `search-ga`, `search-hc`, `search-sa`, `search-tabu` | 6: tuned search (needs `data/tuned/`, see below) |
| `learned:<model.pkl>`, `search-learned:<model.pkl>` | 7: learned evaluator, one-ply or inside search (needs a model trained on the same board) |

Example with every solver on 2×3:

```bash
python -m harness.cli tournament --board 2x3 --games 50 --seed 0 \
    --players greedy,exact,search,rules,mcts,search-ga,learned:data/models/e5_mlp-64x64_both_reg_2x3_s0.pkl
```

## Generate data

`data/` is git-ignored; regenerate it before using solvers 6 and 7.

```bash
# Solver 6: tuned weights in data/tuned/ (about 2 hours with the defaults)
python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400 --seed 0

# Solver 7: datasets, then a 2x3 model for the learned players
python -m solvers.supervised.dataset --board 2x3 --source exact --n 50000 --seed 0
python -m solvers.supervised.models --data data/sup_exact_2x3_n50000_s0.npz --model mlp --input both \
    --task reg --seed 0 --params '{"hidden": [64, 64]}' --out data/models/e5_mlp-64x64_both_reg_2x3_s0.pkl
```

For 3×3, build the dataset with `--source mc --rollouts 200 --workers 4` and train models with `python -m experiments.e4_learned_evaluator --data data/sup_mc_3x3_n50000_s0.npz --seed 0 --save-models`.
