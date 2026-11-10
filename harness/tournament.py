"""Matches, round-robin tournaments and CSV output.

CSV row per game (``CSV_COLUMNS``):

* ``p0`` / ``p1`` - names of the players seated as player 0 / player 1; player 0
  moves first. Seats alternate every game of a match.
* ``score0`` / ``score1`` - boxes won by player 0 / player 1.
* ``winner`` - 0 or 1, or -1 for a draw.
* ``moves`` - total moves in the game.
* ``time_p0`` / ``time_p1`` - mean seconds per move for that player in this game.
* ``nodes_p0`` / ``nodes_p1`` - mean nodes expanded per move (players without a
  ``nodes`` attribute report 0).
* ``depth_p0`` / ``depth_p1`` - mean search depth reached per move, from the
  player's ``depth_reached`` attribute (0 if absent).
* ``bstar_p0`` / ``bstar_p1`` - mean per-move effective branching factor,
  computed from that move's ``nodes`` and ``depth_reached`` (0 if no move had a
  depth >= 1).

Each game builds fresh players from ``factory(name, player_seed)`` where the
player seeds are derived from the game ``seed``, so every row is reproducible on
its own.
"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from engine.game import apply_move, is_terminal, legal_moves, new_game
from harness.metrics import effective_branching_factor
from solvers.base import Player

CSV_COLUMNS: tuple[str, ...] = (
    "p0",
    "p1",
    "board",
    "seed",
    "score0",
    "score1",
    "winner",
    "moves",
    "time_p0",
    "time_p1",
    "nodes_p0",
    "nodes_p1",
    "depth_p0",
    "depth_p1",
    "bstar_p0",
    "bstar_p1",
)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"
DEFAULT_TIME_LIMIT = 1.0

PlayerFactory = Callable[[str, int], Player]


def play_game(
    p0: Player, p1: Player, m: int, n: int, seed: int, time_limit: float | None = DEFAULT_TIME_LIMIT
) -> dict[str, Any]:
    """Play one game with ``p0`` moving first and return its CSV row."""
    players = (p0, p1)
    for p in players:
        p.reset()
    state = new_game(m, n, to_move=0)
    elapsed = [0.0, 0.0]
    nodes = [0, 0]
    depth = [0, 0]
    bstar = [0.0, 0.0]
    searched = [0, 0]   # moves with depth_reached >= 1
    counts = [0, 0]
    while not is_terminal(state):
        seat = state.to_move
        player = players[seat]
        t0 = time.perf_counter()
        move = player.choose_move(state, time_limit)
        elapsed[seat] += time.perf_counter() - t0
        if move not in legal_moves(state):
            raise ValueError(f"{player.name} played illegal move {move}")
        move_nodes = int(getattr(player, "nodes", 0))
        move_depth = int(getattr(player, "depth_reached", 0))
        nodes[seat] += move_nodes
        depth[seat] += move_depth
        if move_depth >= 1:
            bstar[seat] += effective_branching_factor(move_nodes, move_depth)
            searched[seat] += 1
        counts[seat] += 1
        state, _ = apply_move(state, move)
    s0, s1 = state.scores

    def mean(total: list[float] | list[int], seat: int, by: list[int] = counts) -> float:
        return total[seat] / by[seat] if by[seat] else 0.0

    return {
        "p0": p0.name,
        "p1": p1.name,
        "board": f"{m}x{n}",
        "seed": seed,
        "score0": s0,
        "score1": s1,
        "winner": 0 if s0 > s1 else 1 if s1 > s0 else -1,
        "moves": counts[0] + counts[1],
        "time_p0": mean(elapsed, 0),
        "time_p1": mean(elapsed, 1),
        "nodes_p0": mean(nodes, 0),
        "nodes_p1": mean(nodes, 1),
        "depth_p0": mean(depth, 0),
        "depth_p1": mean(depth, 1),
        "bstar_p0": mean(bstar, 0, searched),
        "bstar_p1": mean(bstar, 1, searched),
    }


def match(
    a: str,
    b: str,
    factory: PlayerFactory,
    m: int,
    n: int,
    games: int,
    seed: int,
    time_limit: float | None = DEFAULT_TIME_LIMIT,
) -> list[dict[str, Any]]:
    """``games`` games between ``a`` and ``b``; ``a`` moves first in even-numbered games.

    Game ``i`` uses seed ``seed + i``.
    """
    rows = []
    for i in range(games):
        game_seed = seed + i
        seeder = random.Random(game_seed)
        seat0_seed, seat1_seed = seeder.getrandbits(32), seeder.getrandbits(32)
        first, second = (a, b) if i % 2 == 0 else (b, a)
        p0, p1 = factory(first, seat0_seed), factory(second, seat1_seed)
        rows.append(play_game(p0, p1, m, n, game_seed, time_limit))
    return rows


def round_robin(
    names: Sequence[str],
    factory: PlayerFactory,
    m: int,
    n: int,
    games: int,
    seed: int,
    time_limit: float | None = DEFAULT_TIME_LIMIT,
) -> list[dict[str, Any]]:
    """Every pair of distinct players plays a ``games``-game match.

    Pair ``j`` (in ``itertools.combinations`` order) uses seeds starting at
    ``seed + j * games``, so no two games share a seed.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate player names in {list(names)}")
    rows: list[dict[str, Any]] = []
    for j, (a, b) in enumerate(combinations(names, 2)):
        rows.extend(match(a, b, factory, m, n, games, seed + j * games, time_limit))
    return rows


def write_csv(rows: Sequence[dict[str, Any]], name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write rows to ``<results_dir>/<name>_<timestamp>.csv``; never overwrites."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_dir / f"{name}_{stamp}.csv"
    suffix = 1
    while path.exists():
        path = results_dir / f"{name}_{stamp}_{suffix}.csv"
        suffix += 1
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})
    return path
