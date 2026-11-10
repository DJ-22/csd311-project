# csd311-project

Dots-and-Boxes solvers for CSD311 Artificial Intelligence.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
pytest -q
```

## Run a tournament

```bash
python -m harness.cli tournament --board 2x2 --games 100 --players random,greedy,exact --seed 0
```

Results are written to `experiments/results/<name>_<timestamp>.csv`.
