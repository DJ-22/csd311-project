# Dots and Boxes

CSD311 Artificial Intelligence group project, Computer Science Department, Shiv Nadar University. Faculty: Snehasis Mukherjee. The final report is [`Report.pdf`](Report.pdf); its LaTeX source is in `report/`.

## Team

| # | Name | Roll number |
|---|---|---|
| 1 | Daksh Jain | 2410110113 |
| 2 | Tilika Chopra | 2410110361 |
| 3 | Arnav Jyoti | 2410110071 |
| 4 | Abhyuday Aulakh | 2410110016 |
| 5 | Dev Khemani | 2410110119 |
| 6 | Medhavee Binnani | 2410110198 |
| 7 | Disha Singh | 2410110128 |
| 8 | Vaidehi Saxena | 2410110366 |

## Setup

Requires Python 3.11+ and Git.

```bash
git clone https://github.com/DJ-22/csd311-project.git
cd csd311-project
python -m venv .venv
```

Activate the environment:

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Install the dependencies and run the tests:

```bash
pip install -r requirements.txt
python -m pytest -q
```

Optional:

- [SWI-Prolog](https://www.swi-prolog.org/download/stable) (64-bit, `bin` folder on `PATH`) for the `rules-prolog` player. The Prolog tests skip without it.
- [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/) to build the report PDF.

## Play

```bash
python -m app
```

Keys: Esc menu, R restart, Space pause and → step (bot vs bot).

## Run a tournament

```bash
python -m harness.cli tournament --board 3x3 --games 100 --seed 0 --time-limit 1.0 \
    --players random,greedy,search,rules,mcts,mcts-flat
```

Options: `--name` (CSV prefix, written to `experiments/results/`), `--workers N` (parallel games; keep N at or below the number of performance cores).

| Player | Solver |
|---|---|
| `random`, `greedy` | 1: baselines |
| `exact` | 2: exact solver |
| `search` | 3: heuristic search |
| `rules`, `rules-prolog` | 4: rule-based (Python or Prolog) |
| `mcts`, `mcts-flat` | 5: UCT and flat Monte Carlo |
| `search-ga`, `search-hc`, `search-sa`, `search-tabu` | 6: tuned search (3×3 weights) |
| `learned:<model.pkl>`, `search-learned:<model.pkl>` | 7: learned evaluator (model trained on the same board) |

## Report experiments

| Experiment | Command |
|---|---|
| E1 overall strength | `python -m harness.cli tournament --board 3x3 --games 50 --seed 0 --workers 8 --name e1 --players random,greedy,search,rules,mcts,mcts-flat,search-ga,search-hc,search-sa,search-tabu,learned:data/models/e4_mlp-64x64_both_reg_3x3_s0.pkl,search-learned:data/models/e4_mlp-64x64_both_reg_3x3_s0.pkl` |
| E2 search efficiency | `python -m experiments.e2_search_efficiency --seed 0` |
| E3 optimiser comparison | `python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400 --seed 0` |
| E4 learned evaluator | `python -m experiments.e4_learned_evaluator --data data/sup_exact_2x3_n50000_s0.npz --seed 0` |
| E5 deductive vs inductive | `python -m experiments.e5_deductive_vs_inductive --board 2x3 --model <model.pkl> --games 200 --seed 0` |
| E6 MCTS scaling | `python -m experiments.e6_mcts_scaling --board 3x3 --games 100 --seed 0` |

## Generate data

The tuned weights, models and board settings the app uses are already in `data/`. To rebuild them, and to build the datasets E4 and E5 need:

```bash
# solver 6 weights (3x3)
python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400 --seed 0

# solver 7 dataset and model (2x3)
python -m solvers.supervised.dataset --board 2x3 --source exact --n 50000 --seed 0
python -m solvers.supervised.models --data data/sup_exact_2x3_n50000_s0.npz --model mlp --input both \
    --task reg --seed 0 --params '{"hidden": [64, 64]}' --out data/models/e5_mlp-64x64_both_reg_2x3_s0.pkl

# solver 7 dataset and models (3x3)
python -m solvers.supervised.dataset --board 3x3 --source mc --n 50000 --seed 0 --rollouts 200 --workers 8
python -m experiments.e4_learned_evaluator --data data/sup_mc_3x3_n50000_s0.npz --seed 0 --save-models

# everything the app uses on 3x4, 4x4 and 5x5 (resumable; about 5 hours)
python -m experiments.optimize_boards
```

## Build the report

From `report/` (or upload the folder to Overleaf):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Then copy `report/main.pdf` to `Report.pdf` in the project root.

## Troubleshooting

- `python` not found on Windows: use `py`.
- PowerShell blocks `Activate.ps1`: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- `rules-prolog` or `pdflatex` not found after installing SWI-Prolog or LaTeX: open a new terminal.
