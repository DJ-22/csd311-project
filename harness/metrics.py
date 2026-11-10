"""Tournament metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

Z95 = 1.959963984540054


def win_rate_ci(score: float, games: int, z: float = Z95) -> tuple[float, float, float]:
    """Win rate (draw = 0.5) with a Wilson score interval.

    ``score`` is the total of 1 / 0.5 / 0 over ``games`` games. Returns
    ``(rate, low, high)``. Wilson is used because it stays informative at rates of
    0 or 1, where the normal approximation collapses to a zero-width interval.
    """
    if games <= 0:
        raise ValueError("games must be positive")
    p = score / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def effective_branching_factor(nodes: float, depth: int, tol: float = 1e-9) -> float:
    """Solve ``N = b + b^2 + ... + b^d`` for ``b*`` by bisection (8.pdf)."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if nodes <= 0:
        return 0.0

    def total(b: float) -> float:
        return sum(b**i for i in range(1, depth + 1))

    lo, hi = 0.0, max(1.0, float(nodes))
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if total(mid) < nodes:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class PlayerSummary:
    """Aggregate results of one player over a set of games."""

    player: str
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_margin: float
    mean_time_per_move: float
    mean_nodes_per_move: float
    mean_depth_per_move: float
    mean_bstar: float


def summarise(rows: Iterable[Mapping[str, Any]], player: str) -> PlayerSummary:
    """Summarise every game in ``rows`` that ``player`` took part in.

    ``time_p*``, ``nodes_p*``, ``depth_p*`` and ``bstar_p*`` in a row are per-move
    means for that player, so the reported means are averages over games of the
    per-game means. ``mean_bstar`` averages only games with a nonzero ``b*``
    (0 if the player never searched).
    """
    wins = draws = losses = 0
    margins: list[float] = []
    times: list[float] = []
    nodes: list[float] = []
    depths: list[float] = []
    bstars: list[float] = []
    for r in rows:
        if r["p0"] == player:
            seat = 0
        elif r["p1"] == player:
            seat = 1
        else:
            continue
        mine, theirs = int(r[f"score{seat}"]), int(r[f"score{1 - seat}"])
        margins.append(mine - theirs)
        times.append(float(r[f"time_p{seat}"]))
        nodes.append(float(r[f"nodes_p{seat}"]))
        depths.append(float(r[f"depth_p{seat}"]))
        b = float(r[f"bstar_p{seat}"])
        if b > 0:
            bstars.append(b)
        winner = int(r["winner"])
        if winner == -1:
            draws += 1
        elif winner == seat:
            wins += 1
        else:
            losses += 1
    games = wins + draws + losses
    if games == 0:
        raise ValueError(f"player {player!r} played no games")
    rate, lo, hi = win_rate_ci(wins + 0.5 * draws, games)
    return PlayerSummary(
        player=player,
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=rate,
        ci_low=lo,
        ci_high=hi,
        mean_margin=sum(margins) / games,
        mean_time_per_move=sum(times) / games,
        mean_nodes_per_move=sum(nodes) / games,
        mean_depth_per_move=sum(depths) / games,
        mean_bstar=sum(bstars) / len(bstars) if bstars else 0.0,
    )
