"""E2 Search efficiency: solver 2 (exact) and solver 3 (heuristic search) ablations.

Part 1, solver 2: solve the empty board of each ``--exact-boards`` board from a
fresh solver with every combination of symmetry (canonical-form memo keys) and
alpha-beta, plus a plain no-memo brute force on boards with at most
``--brute-max-edges`` edges (the 2x2 brute force alone is ~10^9 nodes). Every
method must return the same value; nodes and seconds are recorded.

Part 2, solver 3: ``--positions`` positions per ``--search-boards`` board are
sampled from seeded Greedy self-play (stopped at a random ply, at least
``--min-free`` undrawn edges left). Each position is searched by a fresh
``SearchPlayer`` (empty TT) in five configurations: ``full``, ``no-tt``,
``no-ordering``, ``no-quiescence`` and ``none`` (all three off).

* ``depth`` mode: iterative deepening to each fixed depth in ``--depths`` with no
  time limit, so nodes compare like with like.
* ``time`` mode (unless ``--time-limit 0``): iterative deepening under the time
  limit, recording the depth reached.

``nodes`` counts every node of every iteration, quiescence nodes included, as the
harness does, and ``b*`` is computed per position from ``nodes`` and the depth
reached and then averaged (CLAUDE.md §10). Wall-clock columns depend on machine
load; nodes and depth are the primary metrics.

Output: ``results/e2_exact_<timestamp>.csv`` and ``results/e2_search_<timestamp>.csv``
plus a printed summary.

Example::

    python -m experiments.e2_search_efficiency --seed 0
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.board import Board, get_board, parse_board
from engine.game import GameState, apply_move, is_terminal, new_game
from experiments._results import RESULTS_DIR, write_rows
from harness.metrics import effective_branching_factor
from solvers.baselines import GreedyPlayer
from solvers.exact import ExactSolver
from solvers.search import SearchPlayer

SEARCH_CONFIGS: dict[str, dict[str, bool]] = {
    "full": {},
    "no-tt": {"use_tt": False},
    "no-ordering": {"use_ordering": False},
    "no-quiescence": {"use_quiescence": False},
    "none": {"use_tt": False, "use_ordering": False, "use_quiescence": False},
}

EXACT_METHODS: dict[str, tuple[bool, bool]] = {  # name -> (use_symmetry, alphabeta)
    "memo": (False, False),
    "memo+sym": (True, False),
    "alphabeta": (False, True),
    "alphabeta+sym": (True, True),
}


class _BruteForce:
    """Plain negamax over the full game tree: no memo, no pruning (counts nodes)."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self.nodes = 0

    def value(self, edges: int) -> int:
        self.nodes += 1
        b = self.board
        if edges == b.full:
            return 0
        best = -(b.num_boxes + 1)
        free = b.full & ~edges
        while free:
            bit = free & -free
            free ^= bit
            e = bit.bit_length() - 1
            k = sum(1 for mk in (b.BOX_MASK[x] for x in b.EDGE_BOXES[e]) if (edges & mk).bit_count() == 3)
            child = edges | bit
            val = k + self.value(child) if k > 0 else -self.value(child)
            if val > best:
                best = val
        return best


def exact_rows(boards: Sequence[str], brute_max_edges: int) -> list[dict[str, Any]]:
    """One row per (board, method): value, nodes and seconds to solve the empty board."""
    rows: list[dict[str, Any]] = []
    for spec in boards:
        bd = parse_board(spec)
        board = get_board(bd.m, bd.n)
        methods: list[tuple[str, Any]] = []
        if board.E <= brute_max_edges:
            methods.append(("brute-force", _BruteForce(board)))
        methods += [(name, ExactSolver(bd.m, bd.n, sym, ab)) for name, (sym, ab) in EXACT_METHODS.items()]
        for name, solver in methods:
            t0 = time.perf_counter()
            v = solver.value(0)
            rows.append({
                "board": f"{bd.m}x{bd.n}",
                "edges": board.E,
                "method": name,
                "value": v,
                "nodes": solver.nodes,
                "seconds": round(time.perf_counter() - t0, 4),
            })
    return rows


def sample_positions(m: int, n: int, count: int, min_free: int, seed: int) -> list[GameState]:
    """``count`` positions from seeded Greedy self-play, each with >= ``min_free`` undrawn edges."""
    board = get_board(m, n)
    if min_free > board.E:
        raise ValueError(f"min_free={min_free} exceeds the {board.E} edges of {m}x{n}")
    rng = random.Random(seed)
    out: list[GameState] = []
    while len(out) < count:
        state = new_game(m, n)
        player = GreedyPlayer(rng.getrandbits(32))
        stop = rng.randint(0, board.E - min_free)
        for _ in range(stop):
            state, _ = apply_move(state, player.choose_move(state))
        if not is_terminal(state):
            out.append(state)
    return out


def _search_row(board: str, i: int, state: GameState, config: str, mode: str,
                depth: int | None, time_limit: float | None, seed: int) -> dict[str, Any]:
    p = SearchPlayer(max_depth=depth, seed=seed, **SEARCH_CONFIGS[config])
    t0 = time.perf_counter()
    move = p.choose_move(state, time_limit)
    secs = time.perf_counter() - t0
    return {
        "board": board,
        "position": i,
        "free_edges": state.board.E - state.edges.bit_count(),
        "config": config,
        "mode": mode,
        "depth_limit": depth if depth is not None else "",
        "time_limit": time_limit if time_limit is not None else "",
        "depth_reached": p.depth_reached,
        "nodes": p.nodes,
        "bstar": round(effective_branching_factor(p.nodes, p.depth_reached), 4) if p.depth_reached else 0.0,
        "seconds": round(secs, 5),
        "move": move,
    }


def search_rows(boards: Sequence[str], positions: int, depths: Sequence[int], time_limit: float,
                min_free: int, seed: int, configs: Sequence[str] = tuple(SEARCH_CONFIGS),
                progress: bool = False) -> list[dict[str, Any]]:
    """One row per (board, position, config, depth or time limit)."""
    rows: list[dict[str, Any]] = []
    for spec in boards:
        bd = parse_board(spec)
        label = f"{bd.m}x{bd.n}"
        pos = sample_positions(bd.m, bd.n, positions, min_free, seed)
        jobs: list[tuple[str, int | None, float | None]] = [("depth", d, None) for d in depths]
        if time_limit > 0:
            jobs.append(("time", None, time_limit))
        for mode, depth, tl in jobs:
            for config in configs:
                for i, state in enumerate(pos):
                    rows.append(_search_row(label, i, state, config, mode, depth, tl, seed + i))
                if progress:
                    print(f"  {label} {mode} {depth or tl} {config}: done", file=sys.stderr, flush=True)
    return rows


def _print_exact(rows: Sequence[dict[str, Any]]) -> None:
    print("Solver 2: solving the empty board")
    print(f"{'board':<6}{'method':<16}{'value':>6}{'nodes':>12}{'seconds':>10}")
    for r in rows:
        print(f"{r['board']:<6}{r['method']:<16}{r['value']:>6}{r['nodes']:>12}{r['seconds']:>10.3f}")
    for b in sorted({r["board"] for r in rows}):
        values = {r["value"] for r in rows if r["board"] == b}
        if len(values) != 1:
            print(f"WARNING: methods disagree on {b}: {sorted(values)}")


def _print_search(rows: Sequence[dict[str, Any]]) -> None:
    print("\nSolver 3: means over positions (nodes ratio is relative to 'full')")
    print(f"{'board':<6}{'mode':<10}{'config':<15}{'depth':>7}{'nodes':>11}{'ratio':>7}{'b*':>7}{'ms':>9}")
    groups: dict[tuple[str, str, Any], dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        key = (r["board"], r["mode"], r["depth_limit"] if r["mode"] == "depth" else r["time_limit"])
        groups.setdefault(key, {}).setdefault(r["config"], []).append(r)
    for (board, mode, lim), by_cfg in groups.items():
        base = statistics.mean(r["nodes"] for r in by_cfg.get("full", [])) if "full" in by_cfg else None
        for config, rs in by_cfg.items():
            nodes = statistics.mean(r["nodes"] for r in rs)
            ratio = f"{nodes / base:.2f}" if base else "-"
            depth = statistics.mean(r["depth_reached"] for r in rs)
            bstar = statistics.mean(r["bstar"] for r in rs)
            ms = 1000 * statistics.mean(r["seconds"] for r in rs)
            tag = f"d={lim}" if mode == "depth" else f"{lim}s"
            print(f"{board:<6}{mode + ' ' + tag:<10}{config:<15}{depth:>7.2f}{nodes:>11.0f}{ratio:>7}{bstar:>7.2f}{ms:>9.1f}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exact-boards", default="1x2,1x3,2x2,2x3", help="boards for the solver 2 part ('' skips it)")
    ap.add_argument("--brute-max-edges", type=int, default=10, help="run the no-memo brute force up to this many edges")
    ap.add_argument("--search-boards", default="3x3", help="boards for the solver 3 part ('' skips it)")
    ap.add_argument("--positions", type=int, default=40, help="sampled positions per board")
    ap.add_argument("--depths", default="2,4,6", help="fixed search depths")
    ap.add_argument("--time-limit", type=float, default=1.0, help="seconds per position in time mode (0 skips it)")
    ap.add_argument("--min-free", type=int, default=8, help="minimum undrawn edges in a sampled position")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    out_dir = Path(args.results_dir)

    exact_boards = [b for b in args.exact_boards.split(",") if b]
    search_boards = [b for b in args.search_boards.split(",") if b]
    print(f"seed={args.seed} exact boards={exact_boards} search boards={search_boards} "
          f"positions={args.positions} depths={args.depths} time limit={args.time_limit}s min free={args.min_free}")
    if exact_boards:
        ex = exact_rows(exact_boards, args.brute_max_edges)
        _print_exact(ex)
        print("wrote", write_rows(ex, "e2_exact", out_dir))
    if search_boards:
        depths = [int(d) for d in args.depths.split(",") if d]
        sr = search_rows(search_boards, args.positions, depths, args.time_limit, args.min_free, args.seed,
                         progress=True)
        _print_search(sr)
        print("wrote", write_rows(sr, "e2_search", out_dir))


if __name__ == "__main__":
    main()
