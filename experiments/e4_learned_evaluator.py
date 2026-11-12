"""E4 Learned evaluator: test error and % optimal moves per model x input kind x task.

For every (model, input kind, task) the model is trained on the train split,
early-stopped on validation where it applies, and scored on the test split
(positions from games never seen in training). ``pct_optimal`` (one-ply move is
one of solver 2's ``best_moves``) is reported when the dataset has exact labels.
Two reference rows are added: Greedy and Random on the same test positions.

Outputs (timestamped, never overwritten):

* ``results/e4_<board>_<timestamp>.csv``: one row per model run;
* ``results/e4_curves_<board>_<timestamp>.csv``: per-epoch train / val loss and
  error for the iterative learners (perceptron, sigmoid unit, MLP);
* ``results/e4_curves_<board>_<timestamp>.png``: MLP learning curves.

Example::

    python -m experiments.e4_learned_evaluator --data data/sup_exact_2x3_n50000_s0.npz --seed 0
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engine.board import get_board
from experiments._results import stamped_path, write_rows
from solvers.baselines import greedy_move
from solvers.supervised.dataset import INPUT_KINDS, load_dataset, make_inputs, split_indices
from solvers.supervised.models import MODEL_DIR, fit_model

# (model, params, tasks)
GRID: tuple[tuple[str, dict[str, Any], tuple[str, ...]], ...] = (
    *(("knn", {"k": k}, ("reg", "cls")) for k in (1, 3, 5, 9, 15)),
    ("perceptron", {}, ("cls",)),
    ("sigmoid", {}, ("reg", "cls")),
    ("mlp", {"hidden": [64]}, ("reg", "cls")),
    ("mlp", {"hidden": [64, 64]}, ("reg", "cls")),
    ("svm", {}, ("reg", "cls")),
    ("nb", {}, ("cls",)),
    ("tree", {}, ("reg", "cls")),
)


def model_label(name: str, params: dict[str, Any]) -> str:
    if name == "knn":
        return f"knn{params['k']}"
    if name == "mlp":
        return "mlp-" + "x".join(str(h) for h in params["hidden"])
    return name


def reference_rows(data: dict[str, np.ndarray], seed: int, base: dict[str, Any]) -> list[dict[str, Any]]:
    """% optimal moves of Greedy and Random on the test split (exact datasets only)."""
    if "best_mask" not in data:
        return []
    board = get_board(int(data["m"]), int(data["n"]))
    _, _, te = split_indices(data)
    rng = random.Random(seed)
    rows = []
    for label, pick in (
        ("greedy", lambda e: greedy_move(e, rng, board)),
        ("random", lambda e: rng.choice([m for m in range(board.E) if not (e >> m) & 1])),
    ):
        hits = sum(1 for i in te if (int(data["best_mask"][i]) >> pick(int(data["edges"][i]))) & 1)
        rows.append({**base, "model": label, "input": "-", "task": "-", "test_pct_optimal": hits / len(te)})
    return rows


def run(data_path: str, seed: int, inputs: Sequence[str], only: set[str] | None, save_models: bool) -> None:
    data = load_dataset(data_path)
    m, n = int(data["m"]), int(data["n"])
    tr, va, te = split_indices(data)
    base = {
        "board": f"{m}x{n}",
        "source": str(data["source"]),
        "seed": seed,
        "dataset": Path(data_path).name,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "n_games": len(np.unique(data["game"])),
    }
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for kind in inputs:
        X = make_inputs(data, kind)  # type: ignore[arg-type]
        for name, params, tasks in GRID:
            label = model_label(name, params)
            if only and name not in only and label not in only:
                continue
            for task in tasks:
                t0 = time.perf_counter()
                lm = fit_model(name, data, kind, task, seed, X=X, **params)  # type: ignore[arg-type]
                secs = time.perf_counter() - t0
                row = {**base, "model": label, "input": kind, "task": task, "train_seconds": round(secs, 2)}
                row.update({k: round(v, 5) for k, v in lm.metrics.items()})
                rows.append(row)
                print({k: row[k] for k in ("model", "input", "task")}, {k: v for k, v in row.items() if "test" in k})
                for key, values in lm.model.history.items():
                    for epoch, v in enumerate(values, 1):
                        curves.append({**base, "model": label, "input": kind, "task": task,
                                       "series": key, "epoch": epoch, "value": v})
                if save_models:
                    lm.save(MODEL_DIR / f"e4_{label}_{kind}_{task}_{m}x{n}_s{seed}.pkl")
    rows.extend(reference_rows(data, seed, base))
    name = f"e4_{m}x{n}"
    print("wrote", write_rows(rows, name))
    if curves:
        print("wrote", write_rows(curves, f"e4_curves_{m}x{n}"))
        plot_curves(curves, stamped_path(f"e4_curves_{m}x{n}", ".png"))


def plot_curves(curves: list[dict[str, Any]], path: Path) -> None:
    """Train vs validation error per epoch for every MLP run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted({(c["model"], c["input"], c["task"]) for c in curves if c["model"].startswith("mlp")})
    if not runs:
        return
    cols = 4
    rows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, run_key in zip(axes.flat, runs):
        for series, style in (("train_error", "-"), ("val_error", "--")):
            pts = [(c["epoch"], c["value"]) for c in curves
                   if (c["model"], c["input"], c["task"]) == run_key and c["series"] == series]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=series.replace("_", " "))
        ax.set_title(" / ".join(run_key), fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE" if run_key[2] == "reg" else "error rate")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(runs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", default=",".join(INPUT_KINDS))
    ap.add_argument("--only", default="", help="comma-separated model names or labels to run (default: all)")
    ap.add_argument("--save-models", action="store_true", help="keep every trained model under data/models/")
    args = ap.parse_args(argv)
    only = {s for s in args.only.split(",") if s} or None
    run(args.data, args.seed, [s for s in args.inputs.split(",") if s], only, args.save_models)


if __name__ == "__main__":
    main()
