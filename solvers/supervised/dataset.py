"""Solver 7 dataset builder.

Positions come from random self-play games between a mix of players (random,
greedy, depth-2 search), sampled at every stage of each game. Each game picks
its first player at random so ``to_move`` and ``first_player`` both vary.

Deduplication is by ``canonical(edges)`` only. Labels (exact ``V`` and the MC
estimate) depend on ``edges`` alone, so the same board with the other value of
``mover_is_first`` would be a near-duplicate with an identical label; keeping
it could put one board in two splits. A board is kept in the **first game it
appears in** (with that game's ``to_move`` / ``first_player``), so every board,
up to symmetry, belongs to exactly one game and one split.

Known limitation: the first player is drawn at random per game, but
``SearchPlayer`` assumes seat 0 moved first (``mover_is_first = to_move == 0``),
so in games where seat 1 starts the depth-2 search player has the long-chain
parity backwards. This only changes which positions get sampled, not their labels.

The train / validation / test split (70 / 15 / 15) is drawn **per game**, not
per position, so no game contributes to two splits (no leakage of near-identical
positions from one game).

Labels:

* ``source="exact"`` (2x2, 2x3): exact ``V`` from solver 2, plus a bitmask of
  solver 2's ``best_moves`` for the % optimal moves metric.
* ``source="mc"`` (3x3, 4x4): ``p = mc_label(...)``, the Monte Carlo estimate of
  P(player to move wins), from ``n_rollouts`` greedy rollouts (solver 5).

Targets:

* regression ``y_reg`` in ``[-1, 1]``: ``V / boxes_remaining`` (exact) or
  ``2p - 1`` (MC);
* classification ``y_cls`` in ``{0, 1}``: "winning for the player to move",
  ``V > 0`` (exact) or ``p > 0.5`` (MC).

Inputs are built from the stored positions by :func:`make_inputs`:

* ``"phi"``: the 12 shared features;
* ``"raw"``: the E edge bits, then ``to_move`` and ``first_player``;
* ``"both"``: ``phi`` followed by ``raw``.

CLI::

    python -m solvers.supervised.dataset --board 2x3 --source exact --n 50000 --seed 0
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np

from engine.board import Board, get_board, parse_board
from engine.game import apply_move, is_terminal, new_game
from engine.symmetry import canonical
from features.features import NUM_FEATURES, phi_edges
from solvers.base import Player
from solvers.baselines import GreedyPlayer, RandomPlayer
from solvers.exact import get_solver
from solvers.mcts import mc_label
from solvers.search import SearchPlayer

Source = Literal["exact", "mc"]
InputKind = Literal["phi", "raw", "both"]
INPUT_KINDS: tuple[str, ...] = ("phi", "raw", "both")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
TRAIN, VAL, TEST = 0, 1, 2
# Self-play mix: (player kind, sampling weight).
PLAYER_MIX: tuple[tuple[str, float], ...] = (("random", 0.3), ("greedy", 0.4), ("search", 0.3))
SEARCH_DEPTH = 2
DEFAULT_ROLLOUTS = 200
STALL_GAMES = 500  # stop sampling after this many games in a row add no new position
MAX_EXACT_EDGES = 20  # solver 2 is practical up to 2x3 (E = 17) in pure Python


def _make_player(kind: str, seed: int) -> Player:
    if kind == "random":
        return RandomPlayer(seed)
    if kind == "greedy":
        return GreedyPlayer(seed)
    if kind == "search":
        return SearchPlayer(max_depth=SEARCH_DEPTH, seed=seed)
    raise ValueError(f"unknown player kind {kind!r}")


def sample_positions(
    board: Board, n_positions: int, seed: int, max_games: int | None = None
) -> dict[str, np.ndarray]:
    """Unique positions from mixed self-play, each tagged with its game and split.

    Stops at ``n_positions`` unique positions, after ``max_games`` games, or when
    ``STALL_GAMES`` games in a row add nothing (small boards run out of positions).
    """
    rng = random.Random(seed)
    kinds = [k for k, _ in PLAYER_MIX]
    weights = [w for _, w in PLAYER_MIX]
    seen: set[int] = set()
    rows: list[tuple[int, int, int, int, int]] = []  # edges, to_move, first_player, game, split
    game = 0
    stall = 0
    while len(rows) < n_positions and stall < STALL_GAMES and (max_games is None or game < max_games):
        first = rng.randrange(2)
        split = rng.choices((TRAIN, VAL, TEST), weights=SPLIT_FRACTIONS)[0]
        seats = {
            first: _make_player(rng.choices(kinds, weights)[0], rng.getrandbits(32)),
            1 - first: _make_player(rng.choices(kinds, weights)[0], rng.getrandbits(32)),
        }
        state = new_game(board.m, board.n, to_move=first)
        added = 0
        while not is_terminal(state) and len(rows) < n_positions:
            key = canonical(state.edges, board.m, board.n)
            if key not in seen:
                seen.add(key)
                rows.append((state.edges, state.to_move, first, game, split))
                added += 1
            move = seats[state.to_move].choose_move(state, None)
            state, _ = apply_move(state, move)
        stall = 0 if added else stall + 1
        game += 1
    arr = np.array(rows, dtype=np.int64).reshape(-1, 5)
    return {
        "edges": arr[:, 0],
        "to_move": arr[:, 1].astype(np.int8),
        "first_player": arr[:, 2].astype(np.int8),
        "game": arr[:, 3],
        "split": arr[:, 4].astype(np.int8),
    }


def _best_mask(edges: int, m: int, n: int) -> int:
    mask = 0
    for e in get_solver(m, n).best_moves(edges):
        mask |= 1 << e
    return mask


def _mc_job(args: tuple[int, int, int, int, int, int]) -> float:
    edges, first, n_rollouts, m, n, job_seed = args
    return mc_label(edges, first, n_rollouts, random.Random(job_seed), get_board(m, n))


def label_positions(
    board: Board,
    pos: dict[str, np.ndarray],
    source: Source,
    seed: int,
    n_rollouts: int = DEFAULT_ROLLOUTS,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """Add labels and targets to the output of :func:`sample_positions`."""
    edges = [int(e) for e in pos["edges"]]
    filled = np.array([e.bit_count() for e in edges], dtype=np.int64)
    remaining = np.array(
        [sum(1 for mk in board.BOX_MASK if (e & mk) != mk) for e in edges], dtype=np.int64
    )
    out = dict(pos)
    out["boxes_remaining"] = remaining
    out["edges_drawn"] = filled
    if source == "exact":
        if board.E > MAX_EXACT_EDGES:
            raise ValueError(f"exact labels need E <= {MAX_EXACT_EDGES}; {board.m}x{board.n} has E={board.E}")
        solver = get_solver(board.m, board.n)
        V = np.array([solver.value(e) for e in edges], dtype=np.int64)
        out["V"] = V
        out["best_mask"] = np.array([_best_mask(e, board.m, board.n) for e in edges], dtype=np.int64)
        out["y_reg"] = (V / np.maximum(remaining, 1)).astype(np.float32)
        out["y_cls"] = (V > 0).astype(np.int8)
    elif source == "mc":
        # One seed per position (not per worker): labels do not depend on ``workers``.
        jobs = [
            (e, int(f), n_rollouts, board.m, board.n, seed * 1_000_003 + i)
            for i, (e, f) in enumerate(zip(edges, pos["first_player"]))
        ]
        if workers > 1:
            with Pool(workers) as pool:
                p = pool.map(_mc_job, jobs, chunksize=256)
        else:
            p = [_mc_job(j) for j in jobs]
        P = np.array(p, dtype=np.float32)
        out["p"] = P
        out["y_reg"] = (2.0 * P - 1.0).astype(np.float32)
        out["y_cls"] = (P > 0.5).astype(np.int8)
    else:
        raise ValueError(f"source must be 'exact' or 'mc', got {source!r}")
    return out


def default_path(board: Board, source: Source, n_positions: int, seed: int) -> Path:
    return DATA_DIR / f"sup_{source}_{board.m}x{board.n}_n{n_positions}_s{seed}.npz"


def build_dataset(
    board: Board | str,
    source: Source = "exact",
    n_positions: int = 50_000,
    seed: int = 0,
    n_rollouts: int = DEFAULT_ROLLOUTS,
    workers: int = 1,
    path: str | Path | None = None,
) -> Path:
    """Sample, label and save a dataset; return its ``.npz`` path."""
    if isinstance(board, str):
        board = parse_board(board)
    pos = sample_positions(board, n_positions, seed)
    data = label_positions(board, pos, source, seed, n_rollouts, workers)
    out = Path(path) if path is not None else default_path(board, source, n_positions, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        m=board.m,
        n=board.n,
        source=source,
        seed=seed,
        n_rollouts=n_rollouts if source == "mc" else 0,
        **data,
    )
    return out


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# Inputs
def raw_vector(edges: int, to_move: int, first_player: int, board: Board) -> np.ndarray:
    """Distributed encoding: one 0/1 input per edge, then ``to_move`` and ``first_player``."""
    bits = np.fromiter(((edges >> e) & 1 for e in range(board.E)), dtype=np.float32, count=board.E)
    return np.concatenate([bits, np.array([to_move, first_player], dtype=np.float32)])


def input_vector(edges: int, to_move: int, first_player: int, board: Board, kind: InputKind) -> np.ndarray:
    """Input vector of one position for input kind ``kind``."""
    if kind == "phi":
        return phi_edges(edges, board, to_move == first_player)
    if kind == "raw":
        return raw_vector(edges, to_move, first_player, board)
    if kind == "both":
        return np.concatenate(
            [phi_edges(edges, board, to_move == first_player), raw_vector(edges, to_move, first_player, board)]
        )
    raise ValueError(f"input kind must be one of {INPUT_KINDS}, got {kind!r}")


def input_size(board: Board, kind: InputKind) -> int:
    return {"phi": NUM_FEATURES, "raw": board.E + 2, "both": NUM_FEATURES + board.E + 2}[kind]


def make_inputs(data: dict[str, np.ndarray], kind: InputKind) -> np.ndarray:
    """``X`` matrix (float32) for every position of a loaded dataset."""
    board = get_board(int(data["m"]), int(data["n"]))
    rows = [
        input_vector(int(e), int(t), int(f), board, kind)
        for e, t, f in zip(data["edges"], data["to_move"], data["first_player"])
    ]
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, input_size(board, kind)), np.float32)


def split_indices(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = data["split"]
    return np.flatnonzero(s == TRAIN), np.flatnonzero(s == VAL), np.flatnonzero(s == TEST)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a solver 7 dataset.")
    ap.add_argument("--board", required=True, help="e.g. 2x3")
    ap.add_argument("--source", choices=("exact", "mc"), default="exact")
    ap.add_argument("--n", type=int, default=50_000, help="target number of unique positions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="MC rollouts per position")
    ap.add_argument("--workers", type=int, default=1, help="processes for MC labelling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = build_dataset(args.board, args.source, args.n, args.seed, args.rollouts, args.workers, args.out)
    data = load_dataset(path)
    tr, va, te = split_indices(data)
    print(
        f"{path}: {len(data['edges'])} positions from {len(np.unique(data['game']))} games "
        f"(train {len(tr)}, val {len(va)}, test {len(te)}); "
        f"winning fraction {data['y_cls'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
