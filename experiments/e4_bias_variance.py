"""E4 bias-variance: MLP train vs test error as the hidden layer grows.

One hidden layer of ``h`` units, ``h`` in ``--sizes``, several seeds. Two regimes:

* ``unregularised``: no L2, no early stopping, fixed ``--epochs``; shows
  underfitting (small ``h``: high train and test error, high bias) and
  overfitting (large ``h``: train error keeps falling, test error stops falling
  or rises, high variance);
* ``regularised``: L2 + early stopping on validation, as used in E4.

``--train-frac`` trains on a seeded fraction of the training games' positions,
which makes overfitting easier to see. The spread of test error across seeds is
reported as a rough variance proxy.

Outputs: ``results/e4_bias_variance_<board>_<timestamp>.csv`` and ``.png``.

Example::

    python -m experiments.e4_bias_variance --data data/sup_exact_2x3_n50000_s0.npz --input raw --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from experiments._results import stamped_path, write_rows
from solvers.supervised.dataset import INPUT_KINDS, TRAIN, load_dataset, make_inputs
from solvers.supervised.models import fit_model

REGIMES: dict[str, dict[str, Any]] = {
    "unregularised": {"l2": 0.0, "early_stopping": False},
    "regularised": {"l2": 1e-4, "early_stopping": True, "patience": 20},
}


def subsample_train(data: dict[str, np.ndarray], frac: float, seed: int) -> dict[str, np.ndarray]:
    """Drop a seeded ``1 - frac`` of the training positions (val / test unchanged)."""
    if frac >= 1.0:
        return data
    rng = np.random.default_rng(seed)
    keep = (data["split"] != TRAIN) | (rng.random(len(data["split"])) < frac)
    return {k: (v[keep] if v.ndim == 1 and len(v) == len(keep) else v) for k, v in data.items()}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--input", choices=INPUT_KINDS, default="raw")
    ap.add_argument("--task", choices=("reg", "cls"), default="reg")
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--merge", default="", help="comma-separated result CSVs to combine and plot (no training)")
    args = ap.parse_args(argv)
    if args.merge:
        merge([s for s in args.merge.split(",") if s])
        return

    full = load_dataset(args.data)
    board = f"{int(full['m'])}x{int(full['n'])}"
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        data = subsample_train(full, args.train_frac, seed)
        X = make_inputs(data, args.input)
        n_train = int(np.sum(data["split"] == TRAIN))
        for regime in args.regimes.split(","):
            for h in sizes:
                t0 = time.perf_counter()
                params = {"hidden": [h], "epochs": args.epochs, **REGIMES[regime]}
                lm = fit_model("mlp", data, args.input, args.task, seed, X=X, **params)
                row = {
                    "board": board, "source": str(full["source"]), "seed": seed, "input": args.input,
                    "task": args.task, "regime": regime, "hidden": h, "epochs_run": len(lm.model.history["train_loss"]),
                    "train_frac": args.train_frac, "n_train": n_train,
                    "train_error": round(lm.metrics["train_error"], 5),
                    "val_error": round(lm.metrics["val_error"], 5),
                    "test_error": round(lm.metrics["test_error"], 5),
                    "test_pct_optimal": round(lm.metrics.get("test_pct_optimal", float("nan")), 5),
                    "train_seconds": round(time.perf_counter() - t0, 2),
                }
                rows.append(row)
                print(row, flush=True)
    name = f"e4_bias_variance_{board}"
    print("wrote", write_rows(rows, name))
    plot(rows, stamped_path(name, ".png"), args.task)


def merge(paths: Sequence[str]) -> None:
    """Combine per-seed runs (e.g. run in parallel, one seed each) into one CSV and figure."""
    import csv

    rows: list[dict[str, Any]] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["hidden"] = int(r["hidden"])
                for k in ("train_error", "val_error", "test_error"):
                    r[k] = float(r[k])
                rows.append(r)
    name = f"e4_bias_variance_{rows[0]['board']}_merged"
    print("wrote", write_rows(rows, name))
    plot(rows, stamped_path(name, ".png"), rows[0]["task"])


def plot(rows: list[dict[str, Any]], path: Any, task: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({r["regime"] for r in rows})
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 4), squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        sub = [r for r in rows if r["regime"] == regime]
        sizes = sorted({r["hidden"] for r in sub})
        for key, style in (("train_error", "o-"), ("test_error", "s--")):
            vals = [[r[key] for r in sub if r["hidden"] == h] for h in sizes]
            mean = [float(np.mean(v)) for v in vals]
            std = [float(np.std(v)) for v in vals]
            ax.errorbar(sizes, mean, yerr=std, fmt=style, capsize=3, label=key.replace("_", " "))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("hidden units")
        ax.set_ylabel("MSE" if task == "reg" else "error rate")
        b, inp = sub[0]["board"], sub[0]["input"]
        ax.set_title(f"{regime} MLP, {b}, input={inp}, seeds={len({r['seed'] for r in sub})}", fontsize=9)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    main()
