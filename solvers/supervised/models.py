"""Solver 7 models: kNN, from-scratch perceptron and sigmoid unit, torch MLP, sklearn baselines.

Every model is wrapped in :class:`LearnedModel`, which also holds the input
scaler (mean / std of the training inputs) and the metadata needed to rebuild
inputs at play time (board, input kind, task).

Common output: :meth:`Model.score` returns a value in ``[-1, 1]`` that estimates
how good the position is for the player to move:

* regression: the prediction of ``y_reg`` (``V / boxes_remaining`` or ``2p - 1``);
* classification: ``2 P(winning) - 1`` (or ``tanh`` of the activation for the
  perceptron, which has no probability).

:meth:`LearnedModel.value` turns that into net boxes, ``score * boxes_remaining``,
the quantity of the value convention (CLAUDE.md §5), so a learned model can be
used by the one-ply player and as solver 3's evaluator.

From-scratch models (numpy, matching the update rules of 17.pdf):

* perceptron: ``o = sign(w·x)``, ``w <- w + eta (t - o) x`` for each example,
  targets ``t in {-1, +1}``; classification only;
* sigmoid unit: ``o = sigma(w·x)``, mini-batch gradient descent on
  squared error (``delta = (t - o) o (1 - o)``) or cross-entropy
  (``delta = t - o``), ``w <- w + eta (mean(delta x) - lambda w)``.

Both keep the weights of the epoch with the lowest validation error (early
stopping). The bias is a weight on a constant input ``x_0 = 1`` and is not
penalised.

CLI::

    python -m solvers.supervised.models --data data/sup_exact_2x3_n50000_s0.npz \
        --model mlp --input phi --task reg --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from engine.board import Board, get_board
from solvers.supervised.dataset import (
    DATA_DIR,
    INPUT_KINDS,
    InputKind,
    input_vector,
    load_dataset,
    make_inputs,
    split_indices,
)

Task = Literal["reg", "cls"]
TASKS: tuple[str, ...] = ("reg", "cls")
MODEL_DIR = DATA_DIR / "models"
MODEL_NAMES: tuple[str, ...] = ("knn", "perceptron", "sigmoid", "mlp", "svm", "nb", "tree")
CLS_ONLY = {"perceptron", "nb"}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _with_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])


def task_error(y: np.ndarray, score: np.ndarray, task: Task) -> float:
    """Test error: mean squared error (regression) or misclassification rate (classification)."""
    if task == "reg":
        return float(np.mean((score - y) ** 2))
    return float(np.mean((score > 0).astype(np.int8) != y))


# Models
class Model(ABC):
    """A trained predictor on scaled inputs."""

    name: str
    task: Task
    history: dict[str, list[float]]

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Value in ``[-1, 1]`` for the player to move (see module docstring)."""


class Perceptron(Model):
    """Rosenblatt perceptron with the perceptron training rule (17.pdf). Classification only."""

    def __init__(self, task: Task, seed: int, eta: float = 0.01, epochs: int = 50) -> None:
        if task != "cls":
            raise ValueError("the perceptron is a classifier; use task='cls'")
        self.name, self.task = "perceptron", task
        self.eta, self.epochs = eta, epochs
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(0)
        self.history = {"train_error": [], "val_error": []}

    def _out(self, Xb: np.ndarray) -> np.ndarray:
        return np.where(Xb @ self.w > 0, 1.0, -1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        Xb, Xvb = _with_bias(X.astype(np.float64)), _with_bias(Xv.astype(np.float64))
        t = np.where(y > 0, 1.0, -1.0)
        tv = np.where(yv > 0, 1.0, -1.0)
        self.w = np.zeros(Xb.shape[1])
        best_w, best_err = self.w.copy(), math.inf
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(Xb)):
                o = 1.0 if Xb[i] @ self.w > 0 else -1.0
                if o != t[i]:
                    self.w += self.eta * (t[i] - o) * Xb[i]
            tr = float(np.mean(self._out(Xb) != t))
            va = float(np.mean(self._out(Xvb) != tv)) if len(Xvb) else tr
            self.history["train_error"].append(tr)
            self.history["val_error"].append(va)
            if va < best_err:
                best_err, best_w = va, self.w.copy()
        self.w = best_w

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(_with_bias(X.astype(np.float64)) @ self.w)


class SigmoidUnit(Model):
    """Single sigmoid unit trained by mini-batch gradient descent (17.pdf), with L2 and early stopping."""

    def __init__(
        self,
        task: Task,
        seed: int,
        loss: Literal["mse", "xent"] = "xent",
        eta: float = 0.5,
        l2: float = 1e-4,
        epochs: int = 300,
        batch: int = 256,
        patience: int = 20,
    ) -> None:
        self.name, self.task = "sigmoid", task
        self.loss, self.eta, self.l2 = loss, eta, l2
        self.epochs, self.batch, self.patience = epochs, batch, patience
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(0)
        self.history = {"train_loss": [], "val_loss": [], "train_error": [], "val_error": []}

    def _target(self, y: np.ndarray) -> np.ndarray:
        # Regression targets in [-1, 1] become [0, 1] for the sigmoid output.
        return (y.astype(np.float64) + 1.0) / 2.0 if self.task == "reg" else y.astype(np.float64)

    def _loss(self, o: np.ndarray, t: np.ndarray) -> float:
        if self.loss == "mse":
            return float(0.5 * np.mean((t - o) ** 2))
        o = np.clip(o, 1e-7, 1 - 1e-7)
        return float(-np.mean(t * np.log(o) + (1 - t) * np.log(1 - o)))

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        Xb, Xvb = _with_bias(X.astype(np.float64)), _with_bias(Xv.astype(np.float64))
        t, tv = self._target(y), self._target(yv)
        self.w = np.zeros(Xb.shape[1])
        reg = np.ones_like(self.w)
        reg[0] = 0.0  # no penalty on the bias
        best_w, best_val, wait = self.w.copy(), math.inf, 0
        for _ in range(self.epochs):
            order = self.rng.permutation(len(Xb))
            for s in range(0, len(order), self.batch):
                idx = order[s : s + self.batch]
                o = _sigmoid(Xb[idx] @ self.w)
                delta = (t[idx] - o) * o * (1 - o) if self.loss == "mse" else t[idx] - o
                self.w += self.eta * ((delta @ Xb[idx]) / len(idx) - self.l2 * reg * self.w)
            o_tr, o_va = _sigmoid(Xb @ self.w), _sigmoid(Xvb @ self.w)
            val = self._loss(o_va, tv)
            self.history["train_loss"].append(self._loss(o_tr, t))
            self.history["val_loss"].append(val)
            self.history["train_error"].append(task_error(y, 2 * o_tr - 1, self.task))
            self.history["val_error"].append(task_error(yv, 2 * o_va - 1, self.task))
            if val < best_val - 1e-6:
                best_val, best_w, wait = val, self.w.copy(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.w = best_w

    def score(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * _sigmoid(_with_bias(X.astype(np.float64)) @ self.w) - 1.0


class MLP(Model):
    """Feed-forward network trained with backpropagation (torch), L2 and early stopping."""

    def __init__(
        self,
        task: Task,
        seed: int,
        hidden: Sequence[int] = (64,),
        activation: Literal["relu", "sigmoid"] = "relu",
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 1e-3,
        l2: float = 1e-4,
        epochs: int = 300,
        batch: int = 256,
        patience: int = 20,
        early_stopping: bool = True,
    ) -> None:
        self.name, self.task = "mlp", task
        self.early_stopping = early_stopping
        self.hidden = tuple(int(h) for h in hidden)
        self.activation, self.optimizer = activation, optimizer
        self.lr, self.l2 = lr, l2
        self.epochs, self.batch, self.patience = epochs, batch, patience
        self.seed = seed
        self.net: Any = None
        self.history = {"train_loss": [], "val_loss": [], "train_error": [], "val_error": []}

    def _build(self, d: int) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        act = nn.ReLU if self.activation == "relu" else nn.Sigmoid
        for h in self.hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def _scores(self, out: Any) -> np.ndarray:
        import torch

        if self.task == "reg":
            return out.clamp(-1.0, 1.0).numpy()
        return (2.0 * torch.sigmoid(out) - 1.0).numpy()

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self.net = self._build(X.shape[1])
        Xt, Xvt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Xv.astype(np.float32))
        yt, yvt = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(yv.astype(np.float32))
        loss_fn = torch.nn.MSELoss() if self.task == "reg" else torch.nn.BCEWithLogitsLoss()
        opt_cls = torch.optim.Adam if self.optimizer == "adam" else torch.optim.SGD
        # weight_decay is the L2 penalty on every weight.
        opt = opt_cls(self.net.parameters(), lr=self.lr, weight_decay=self.l2)
        best_state, best_val, wait = None, math.inf, 0
        for _ in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(Xt), generator=gen)
            for s in range(0, len(order), self.batch):
                idx = order[s : s + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                out_tr, out_va = self.net(Xt).squeeze(1), self.net(Xvt).squeeze(1)
                tr_loss, va_loss = float(loss_fn(out_tr, yt)), float(loss_fn(out_va, yvt))
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_error"].append(task_error(y, self._scores(out_tr), self.task))
            self.history["val_error"].append(task_error(yv, self._scores(out_va), self.task))
            if not self.early_stopping:
                continue  # train for all epochs and keep the final weights
            if va_loss < best_val - 1e-6:
                best_val, wait = va_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.net(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)
        return self._scores(out).astype(np.float64)


class SklearnModel(Model):
    """kNN, SVM, Naive Bayes and decision tree via scikit-learn."""

    def __init__(self, kind: str, task: Task, seed: int, k: int = 5, max_depth: int | None = None,
                 svm_max_train: int = 10_000) -> None:
        if kind in CLS_ONLY and task != "cls":
            raise ValueError(f"{kind} is a classifier; use task='cls'")
        self.kind, self.task, self.seed = kind, task, seed
        self.name = f"knn{k}" if kind == "knn" else kind
        self.k, self.max_depth, self.svm_max_train = k, max_depth, svm_max_train
        self.est: Any = None
        self.history = {}

    def _make(self) -> Any:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        reg = self.task == "reg"
        if self.kind == "knn":
            return KNeighborsRegressor(self.k) if reg else KNeighborsClassifier(self.k)
        if self.kind == "svm":
            return SVR() if reg else SVC(probability=True, random_state=self.seed)
        if self.kind == "nb":
            return GaussianNB()
        if self.kind == "tree":
            cls = DecisionTreeRegressor if reg else DecisionTreeClassifier
            return cls(max_depth=self.max_depth, min_samples_leaf=5, random_state=self.seed)
        raise ValueError(f"unknown sklearn model {self.kind!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        if self.kind == "svm" and len(X) > self.svm_max_train:
            # Kernel SVMs scale ~quadratically; train on a seeded subsample.
            idx = np.random.default_rng(self.seed).choice(len(X), self.svm_max_train, replace=False)
            X, y = X[idx], y[idx]
        self.est = self._make()
        self.est.fit(X, y)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.task == "reg":
            return np.clip(self.est.predict(X), -1.0, 1.0).astype(np.float64)
        proba = self.est.predict_proba(X)
        classes = list(self.est.classes_)
        p_win = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        return 2.0 * p_win - 1.0


def make_model(name: str, task: Task, seed: int, **params: Any) -> Model:
    """Build an untrained model. ``name`` is one of ``MODEL_NAMES``."""
    if name == "perceptron":
        return Perceptron(task, seed, **params)
    if name == "sigmoid":
        return SigmoidUnit(task, seed, **params)
    if name == "mlp":
        return MLP(task, seed, **params)
    if name in ("knn", "svm", "nb", "tree"):
        return SklearnModel(name, task, seed, **params)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


# Wrapper with scaler and metadata
@dataclass
class LearnedModel:
    """A trained model plus everything needed to evaluate it on a raw position."""

    model: Model
    m: int
    n: int
    input_kind: InputKind
    task: Task
    seed: int
    mean: np.ndarray
    std: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def board(self) -> Board:
        return get_board(self.m, self.n)

    @property
    def label(self) -> str:
        return self.model.name

    def scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def score_inputs(self, X: np.ndarray) -> np.ndarray:
        """Scores in ``[-1, 1]`` for unscaled input rows."""
        return self.model.score(self.scale(X))

    def inputs(self, positions: Sequence[tuple[int, int, int]]) -> np.ndarray:
        """Input matrix for ``(edges, to_move, first_player)`` positions."""
        b = self.board
        return np.stack([input_vector(e, t, f, b, self.input_kind) for e, t, f in positions])

    def value(self, edges: int, board: Board, mover_is_first: bool) -> float:
        """Estimated net boxes for the player to move (solver 3 ``Evaluator`` signature).

        The harness always starts with seat 0, so ``first_player = 0`` and
        ``to_move = 0 if mover_is_first else 1``.
        """
        remaining = _boxes_remaining(edges, board)
        if remaining == 0:
            return 0.0
        X = self.inputs([(edges, 0 if mover_is_first else 1, 0)])
        return float(self.score_inputs(X)[0]) * remaining

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path


def load_model(path: str | Path) -> LearnedModel:
    """Load a model saved by :func:`train` (only load files you made: this is pickle)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, LearnedModel):
        raise TypeError(f"{path} does not hold a LearnedModel")
    return obj


def _boxes_remaining(edges: int, board: Board) -> int:
    return sum(1 for mk in board.BOX_MASK if (edges & mk) != mk)


# One-ply move choice (shared by the player and the % optimal metric)
def child_positions(edges: int, to_move: int, first_player: int, board: Board) -> list[tuple[int, int, int, int, int]]:
    """``(move, k, child_edges, child_to_move, child_remaining)`` for every legal move."""
    out = []
    for e in range(board.E):
        if (edges >> e) & 1:
            continue
        k = sum(1 for b in board.EDGE_BOXES[e] if (edges & board.BOX_MASK[b]).bit_count() == 3)
        child = edges | (1 << e)
        out.append((e, k, child, to_move if k > 0 else 1 - to_move, _boxes_remaining(child, board)))
    return out


def one_ply_scores(lm: LearnedModel, edges: int, to_move: int, first_player: int, board: Board) -> dict[int, float]:
    """``score(a) = k + V̂(s')`` if ``a`` completes ``k > 0`` boxes, else ``-V̂(s')``."""
    kids = child_positions(edges, to_move, first_player, board)
    vals = lm.score_inputs(lm.inputs([(c, t, first_player) for _, _, c, t, _ in kids]))
    return {e: (k + v * r if k > 0 else -v * r) for (e, k, _, _, r), v in zip(kids, vals)}


def pct_optimal(lm: LearnedModel, data: dict[str, np.ndarray], idx: np.ndarray) -> float:
    """Fraction of positions where the one-ply move is one of solver 2's ``best_moves``.

    Batched: all children of all positions are scored in one call.
    """
    board = lm.board
    rows: list[tuple[int, int, int]] = []
    meta: list[tuple[int, int, int, int]] = []  # position, move, k, remaining
    for j, i in enumerate(idx):
        e, t, f = int(data["edges"][i]), int(data["to_move"][i]), int(data["first_player"][i])
        for mv, k, c, ct, r in child_positions(e, t, f, board):
            rows.append((c, ct, f))
            meta.append((j, mv, k, r))
    if not rows:
        return float("nan")
    vals = lm.score_inputs(lm.inputs(rows))
    best: dict[int, tuple[float, int]] = {}
    for (j, mv, k, r), v in zip(meta, vals):
        s = k + v * r if k > 0 else -v * r
        if j not in best or s > best[j][0] + 1e-9:  # ties: lowest edge index, like a deterministic player
            best[j] = (s, mv)
    hits = sum(1 for j, i in enumerate(idx) if (int(data["best_mask"][i]) >> best[j][1]) & 1)
    return hits / len(idx)


# Training
def default_model_path(name: str, m: int, n: int, input_kind: str, task: str, seed: int, tag: str = "") -> Path:
    return MODEL_DIR / f"{name}{tag}_{input_kind}_{task}_{m}x{n}_s{seed}.pkl"


def evaluate(
    lm: LearnedModel, data: dict[str, np.ndarray], X: np.ndarray, idx: np.ndarray, moves: bool = True
) -> dict[str, float]:
    """Error on ``idx`` for the model's task, plus % optimal moves when exact labels exist and ``moves``."""
    y = data["y_reg"] if lm.task == "reg" else data["y_cls"]
    s = lm.score_inputs(X[idx])
    out = {"error": task_error(y[idx], s, lm.task), "cls_error": task_error(data["y_cls"][idx], s, "cls")}
    if moves and "best_mask" in data:
        out["pct_optimal"] = pct_optimal(lm, data, idx)
    return out


def fit_model(
    model_name: str,
    data: dict[str, np.ndarray],
    input_kind: InputKind,
    task: Task,
    seed: int,
    X: np.ndarray | None = None,
    **params: Any,
) -> LearnedModel:
    """Train on the train split, early-stop on validation, report metrics on all three splits."""
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"input kind must be one of {INPUT_KINDS}")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if X is None:
        X = make_inputs(data, input_kind)
    tr, va, te = split_indices(data)
    y = data["y_reg"] if task == "reg" else data["y_cls"]
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std < 1e-8] = 1.0  # constant inputs stay unscaled
    model = make_model(model_name, task, seed, **params)
    lm = LearnedModel(model, int(data["m"]), int(data["n"]), input_kind, task, seed, mean, std, dict(params))
    model.fit(lm.scale(X[tr]), y[tr], lm.scale(X[va]), y[va])
    # % optimal moves is a test-set metric; skipping it on train / val saves time.
    for split, idx in (("train", tr), ("val", va), ("test", te)):
        for k, v in evaluate(lm, data, X, idx, moves=split == "test").items():
            lm.metrics[f"{split}_{k}"] = v
    return lm


def train(
    model_name: str,
    dataset_path: str | Path,
    input_kind: InputKind,
    task: Task,
    seed: int = 0,
    out: str | Path | None = None,
    **params: Any,
) -> Path:
    """Train one model on a saved dataset and save it; return the model path."""
    data = load_dataset(dataset_path)
    lm = fit_model(model_name, data, input_kind, task, seed, **params)
    tag = f"{params['k']}" if model_name == "knn" and "k" in params else ""
    path = Path(out) if out else default_model_path(model_name, lm.m, lm.n, input_kind, task, seed, tag)
    return lm.save(path)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a solver 7 model.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--input", choices=INPUT_KINDS, default="phi")
    ap.add_argument("--task", choices=TASKS, default="reg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="{}", help='JSON model parameters, e.g. \'{"hidden": [64, 64]}\'')
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = train(args.model, args.data, args.input, args.task, args.seed, args.out, **json.loads(args.params))
    lm = load_model(path)
    print(path)
    print(json.dumps({k: round(v, 4) for k, v in lm.metrics.items()}, indent=1))


if __name__ == "__main__":
    # Run through the package module so pickled classes resolve to
    # ``solvers.supervised.models`` rather than ``__main__``.
    from solvers.supervised.models import main as _main

    _main()
