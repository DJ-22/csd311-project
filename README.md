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

## Play against the solvers

```bash
python -m app
```

A window opens with a menu: play against any solver yourself, or pick two solvers and watch them play (pause, step, speed). Choose the board (2×2 up to 5×5), who moves first and the thinking time per move. Click a line to draw it. Keys: Esc menu, R restart, Space pause and → step (bot vs bot).

Which bots a board offers:

- **On the boards a bot was built for, it is always offered.** For the bots optimised per board (tuned weights, MCTS settings, a learned model, the exact solve) these are the boards they were optimised on. For the others they are the experiment boards 2×2, 2×3 and 3×3.
- **On any other board, a bot is offered only if it beat Greedy in at least 80% of 40 test games there** (1 s per move, seed 0).

A board is listed only if it offers at least one bot. Where a bot was tested on a board, its description in the menu shows the result. Results of `python -m experiments.optimize_boards` on the reference machine (40 games against Greedy, 1 s per move, seed 0):

| Bot | 3×4 | 4×4 | 5×5 |
|---|---|---|---|
| Flat Monte Carlo \* | 98% | 96% | 100% |
| Heuristic search | 91% | 91% | 88% |
| Search + learned \* | 88% | 86% | 73% |
| Rule-based | 74% (not offered) | 90% | 95% |
| Search tuned by a GA \*† | 68% | 74% | 95% |
| Search tuned by tabu search \*† | 78% | 76% | 85% |
| Search tuned by hill climbing \*† | 73% | 65% | 85% |
| Search tuned by simulated annealing \*† | 86% | 71% | 83% |
| MCTS (UCT) \* | 76% | 53% | 8% |
| Learned, one move ahead \* | 46% | 35% | 23% |

\* Optimised for each of these boards, so it is offered whatever its score. † Plays 2 moves ahead, the depth its weights were tuned at. With 1 s per move it searches deeper, and these weights then give boxes away (5×5: 3–48% against Greedy). Exact is offered up to 3×3, where it beat Greedy in 98% of games; its first move there takes about 50 s while it solves the board.

Needs `pygame-ce` (in `requirements.txt`). The data the bots use is included in the repo, so every board and bot above is available right after cloning.

## Run a tournament

```bash
python -m harness.cli tournament --board 3x3 --games 100 --seed 0 --time-limit 1.0 \
    --players random,greedy,search,rules,mcts,mcts-flat
```

Results are written to `experiments/results/<name>_<timestamp>.csv` (`--name` sets the prefix).

`--workers N` plays games in N parallel processes with the same seeds and row order. Keep N at or below the number of physical (performance) cores: time-limited players search less when they share a core.

| Player | Solver |
|---|---|
| `random`, `greedy` | 1: baselines |
| `exact` | 2: exact solver (2×2 and 2×3; slow on 3×3) |
| `search` | 3: heuristic search |
| `rules`, `rules-prolog` | 4: rule-based, Python or Prolog backend |
| `mcts`, `mcts-flat` | 5: UCT and flat Monte Carlo |
| `search-ga`, `search-hc`, `search-sa`, `search-tabu` | 6: tuned search with the 3×3 weights in `data/tuned/` (on other boards, use the app or `optimize_boards`) |
| `learned:<model.pkl>`, `search-learned:<model.pkl>` | 7: learned evaluator, one-ply or inside search (needs a model trained on the same board) |

Example with every solver on 2×3:

```bash
python -m harness.cli tournament --board 2x3 --games 50 --seed 0 \
    --players greedy,exact,search,rules,mcts,search-ga,learned:data/models/e5_mlp-64x64_both_reg_2x3_s0.pkl
```

## Generate data

Most of `data/` is git-ignored. The repo includes only the small files the app and the solver 6 and 7 players need: the tuned weights (`data/tuned/`), the trained models (`data/models/`), and the larger boards' Greedy test results and MCTS settings (`data/board_opt/`). The datasets are not included. The commands below rebuild everything; rerunning them overwrites the included files.

```bash
# Solver 6: tuned weights in data/tuned/ (about 2 hours with the defaults)
python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400 --seed 0

# Solver 7: datasets, then a 2x3 model for the learned players
python -m solvers.supervised.dataset --board 2x3 --source exact --n 50000 --seed 0
python -m solvers.supervised.models --data data/sup_exact_2x3_n50000_s0.npz --model mlp --input both \
    --task reg --seed 0 --params '{"hidden": [64, 64]}' --out data/models/e5_mlp-64x64_both_reg_2x3_s0.pkl
```

For 3×3, build the dataset with `--source mc --rollouts 200 --workers 8` and train models with `python -m experiments.e4_learned_evaluator --data data/sup_mc_3x3_n50000_s0.npz --seed 0 --save-models`.

For the app's larger boards (3×4, 4×4, 5×5), one resumable command tunes the solver 6 weights and the MCTS settings, builds a learned model for each board and tests every bot against Greedy. It took about 5 hours on the reference machine (20 cores); press Ctrl+C to stop and rerun the same command to resume:

```bash
python -m experiments.optimize_boards           # --list shows the jobs and which are done
```

## Report experiments

Each script takes `--seed` and writes timestamped files to `experiments/results/`.

| Experiment | Command |
|---|---|
| E1 overall strength | `python -m harness.cli tournament --board 3x3 --games 50 --seed 0 --workers 8 --name e1 --players random,greedy,search,rules,mcts,mcts-flat,search-ga,search-hc,search-sa,search-tabu,learned:data/models/e4_mlp-64x64_both_reg_3x3_s0.pkl,search-learned:data/models/e4_mlp-64x64_both_reg_3x3_s0.pkl` |
| E2 search efficiency | `python -m experiments.e2_search_efficiency --seed 0` |
| E3 optimiser comparison | `python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400 --seed 0` |
| E4 learned evaluator | `python -m experiments.e4_learned_evaluator --data data/sup_exact_2x3_n50000_s0.npz --seed 0` |
| E5 deductive vs inductive | `python -m experiments.e5_deductive_vs_inductive --board 2x3 --model <model.pkl> --games 200 --seed 0` |
| E6 MCTS scaling | `python -m experiments.e6_mcts_scaling --board 3x3 --games 100 --seed 0` |
