"""Solver 4: rule-based (knowledge-based) player.

Expert Dots-and-Boxes knowledge as prioritised rules; the first rule that applies
picks the move (the cut in ``rules.pl``):

1. ``endgame_double_deal`` - the opponent has opened a long chain or loop and
   other long chains or loops remain: capture all but the last 2 (chain) or 4
   (loop) boxes, then decline them (double-deal) to keep control, unless the
   sacrifice exceeds the value of keeping control.
2. ``capture`` - take any other capturable box.
3. ``parity_safe_move`` - a safe move after which ``dots + long chains`` has the
   parity this player wants (long chain rule, Berlekamp).
4. ``safe_move`` - any safe move.
5. ``min_sacrifice`` - no safe move is left: make the opening that gives the
   opponent the fewest boxes (chains before loops on ties); a 2-chain is opened
   only by its middle edge (hard-hearted handout).

Two backends implement the same rules move for move: ``"python"`` (no extra
dependencies) and ``"prolog"`` (``rules.pl`` through ``pyswip``, imported lazily).
Every tie is broken by the lowest edge index, so both backends agree exactly.

Capturable runs. ``engine.chains.find_components`` never includes a 3-sided box
in a component, so an opened chain shows up as a shorter chain with
``open_ends >= 1`` plus its capturable box(es). The rules therefore measure what
can be captured as *runs*: start at a 3-sided box and follow its missing side
through 2-sided boxes. A run is *one-ended* if it stops at the boundary or at a
box with at most 1 side drawn, and *two-ended* if it reaches another 3-sided box
(an opened loop, or a chain opened at both ends).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from engine.board import Board
from engine.chains import LONG_CHAIN, Component, find_components
from engine.game import GameState
from solvers.base import Player
from solvers.baselines import capture_run

RULE_DOUBLE_DEAL = "endgame_double_deal"
RULE_CAPTURE = "capture"
RULE_PARITY_SAFE = "parity_safe_move"
RULE_SAFE = "safe_move"
RULE_MIN_SACRIFICE = "min_sacrifice"
RULES: tuple[str, ...] = (RULE_DOUBLE_DEAL, RULE_CAPTURE, RULE_PARITY_SAFE, RULE_SAFE, RULE_MIN_SACRIFICE)

RULES_PL = Path(__file__).resolve().parent / "rules.pl"


@dataclass(frozen=True)
class Run:
    """Boxes capturable in sequence, in order from the 3-sided box the run starts at."""

    boxes: tuple[int, ...]
    two_ended: bool
    captures: tuple[int, ...]  # capture moves available now (sorted edge indices)

    @property
    def sacrifice(self) -> int:
        """Boxes to hand back to keep control: 2 (one-ended), 4 (two-ended), 0 if impossible."""
        if self.two_ended:
            return 4 if len(self.boxes) >= 4 else 0
        return 2 if len(self.boxes) >= 2 else 0


# helpers on the raw bitmask (board last, CLAUDE.md §3)


def _sides(edges: int, board: Board) -> list[int]:
    return [(edges & mk).bit_count() for mk in board.BOX_MASK]


def _undrawn_sides(edges: int, box: int, board: Board) -> list[int]:
    return [e for e in board.BOX_EDGES[box] if not (edges >> e) & 1]


def _across(edge: int, box: int, board: Board) -> int | None:
    """The box on the other side of ``edge`` from ``box`` (``None`` at the boundary)."""
    for x in board.EDGE_BOXES[edge]:
        if x != box:
            return x
    return None


def capture_runs(edges: int, board: Board) -> list[Run]:
    """All capturable runs, ordered by their starting (lowest-index) 3-sided box."""
    sides = _sides(edges, board)
    seen: set[int] = set()
    runs: list[Run] = []
    for a in range(board.num_boxes):
        if sides[a] != 3 or a in seen:
            continue
        path = [a]
        two_ended = False
        (through,) = _undrawn_sides(edges, a, board)
        nxt = _across(through, a, board)
        while nxt is not None and sides[nxt] >= 2:
            path.append(nxt)
            if sides[nxt] == 3:
                two_ended = True
                break
            through = next(e for e in _undrawn_sides(edges, nxt, board) if e != through)
            nxt = _across(through, nxt, board)
        seen.update(path)
        caps = set(_undrawn_sides(edges, path[0], board))
        if two_ended:
            caps.update(_undrawn_sides(edges, path[-1], board))
        runs.append(Run(tuple(path), two_ended, tuple(sorted(caps))))
    return runs


def safe_moves(edges: int, board: Board) -> list[int]:
    """Undrawn edges next to no 2-sided box (they make no box capturable), ascending."""
    mask = board.BOX_MASK
    return [
        e
        for e in range(board.E)
        if not (edges >> e) & 1 and all((edges & mask[b]).bit_count() != 2 for b in board.EDGE_BOXES[e])
    ]


def _is_long(c: Component) -> bool:
    return c.kind == "loop" or c.length >= LONG_CHAIN


def control_value(remaining: list[Component]) -> int:
    """Estimated boxes for the player in control: all remaining boxes minus the sacrifices
    needed to keep control (2 per long chain, 4 per loop, none on the last one taken).

    The opponent chooses the opening order, so the last component is a long chain if
    any remains (saving the controller only 2), otherwise a loop.
    """
    total = sum(c.length for c in remaining)
    chains = sum(1 for c in remaining if c.kind == "chain" and c.length >= LONG_CHAIN)
    loops = sum(1 for c in remaining if c.kind == "loop")
    last = 2 if chains else 4 if loops else 0
    return total - (2 * chains + 4 * loops - last)


def _double_deal_edge(edges: int, run: Run, board: Board) -> int:
    """The decline: leave one domino (one-ended, 2 boxes) or two dominoes (two-ended, 4)."""
    if run.two_ended:
        c1, c2 = run.boxes[1], run.boxes[2]
        (shared,) = set(_undrawn_sides(edges, c1, board)) & set(_undrawn_sides(edges, c2, board))
        return shared
    a, y = run.boxes
    (into,) = _undrawn_sides(edges, a, board)
    return next(e for e in _undrawn_sides(edges, y, board) if e != into)


# the rules


def rule_endgame_double_deal(edges: int, runs: list[Run], board: Board) -> int | None:
    """Rule 1. ``None`` when it does not apply."""
    dd = [r for r in runs if r.sacrifice]
    if not dd:
        return None
    remaining = [c for c in find_components(edges, board) if c.kind == "loop" or c.open_ends == 0]
    if not any(_is_long(c) for c in remaining):
        return None
    target = min(dd, key=lambda r: (r.sacrifice, len(r.boxes), r.captures[0]))
    others = [r for r in runs if r is not target]
    if others:  # everything else is free: take it first
        return min(e for r in others for e in r.captures)
    s = target.sacrifice
    if len(target.boxes) > s or s > control_value(remaining):
        return target.captures[0]
    return _double_deal_edge(edges, target, board)


def rule_capture(runs: list[Run]) -> int | None:
    """Rule 2."""
    caps = [e for r in runs for e in r.captures]
    return min(caps) if caps else None


def _parity_ok(edges: int, board: Board, mover_is_first: bool) -> bool:
    long_chains = sum(1 for c in find_components(edges, board) if c.kind == "chain" and c.length >= LONG_CHAIN)
    even = (board.dots + long_chains) % 2 == 0
    return even == mover_is_first  # first player wants even, second wants odd


def rule_parity_safe_move(edges: int, safe: list[int], mover_is_first: bool, board: Board) -> int | None:
    """Rule 3: the lowest safe edge after which the long-chain parity favours the mover."""
    for e in safe:
        if _parity_ok(edges | (1 << e), board, mover_is_first):
            return e
    return None


def rule_safe_move(safe: list[int]) -> int | None:
    """Rule 4: the lowest safe edge."""
    return safe[0] if safe else None


def opening_moves(edges: int, comp: Component, board: Board) -> list[int]:
    """Candidate edges for opening ``comp``: a 2-chain only by its middle edge (the
    hard-hearted handout, so the opponent cannot double-deal), any other component by
    any undrawn side of its boxes."""
    members = set(comp.boxes)
    out: set[int] = set()
    for b in comp.boxes:
        for e in _undrawn_sides(edges, b, board):
            if comp.kind == "chain" and comp.length == 2 and _across(e, b, board) not in members:
                continue
            out.add(e)
    return sorted(out)


def rule_min_sacrifice(edges: int, board: Board) -> int | None:
    """Rule 5: the opening that lets the opponent capture the fewest boxes.

    The sacrifice is measured by the capture closure after the move, not by the
    component's length: components can meet at a junction box, and opening one of
    them can then hand over the other as well. Ties: chains before loops, then the
    lowest edge.
    """
    keys = [
        (capture_run(edges | (1 << e), board)[0], 0 if c.kind == "chain" else 1, e)
        for c in find_components(edges, board)
        for e in opening_moves(edges, c, board)
    ]
    return min(keys)[2] if keys else None


def decide(edges: int, mover_is_first: bool, board: Board) -> tuple[int, str]:
    """Python backend: ``(move, rule fired)`` for the player to move."""
    if edges == board.full:
        raise ValueError("no legal moves: board is full")
    runs = capture_runs(edges, board)
    if runs:
        move = rule_endgame_double_deal(edges, runs, board)
        if move is not None:
            return move, RULE_DOUBLE_DEAL
        return rule_capture(runs), RULE_CAPTURE  # type: ignore[return-value]
    safe = safe_moves(edges, board)
    if safe:
        move = rule_parity_safe_move(edges, safe, mover_is_first, board)
        if move is not None:
            return move, RULE_PARITY_SAFE
        return rule_safe_move(safe), RULE_SAFE  # type: ignore[return-value]
    move = rule_min_sacrifice(edges, board)
    if move is None:  # unreachable: with no capture and no safe move every edge is in a component
        move = next(e for e in range(board.E) if not (edges >> e) & 1)
    return move, RULE_MIN_SACRIFICE


# Prolog backend


class PrologUnavailableError(RuntimeError):
    """pyswip or SWI-Prolog is missing."""


_PROLOG: Any = None


def _prolog() -> Any:
    """The shared pyswip engine with ``rules.pl`` consulted (imported on first use)."""
    global _PROLOG
    if _PROLOG is None:
        try:
            from pyswip import Prolog  # lazy: only the Prolog backend needs it
        except ImportError as exc:
            raise PrologUnavailableError(
                "the Prolog backend needs pyswip (`pip install pyswip`) and SWI-Prolog; "
                "use RulePlayer(backend='python') instead"
            ) from exc
        except Exception as exc:  # pyswip raises at import time when libswipl is not found
            raise PrologUnavailableError(
                f"pyswip could not load SWI-Prolog ({exc}); install SWI-Prolog and put it on PATH, "
                "or use RulePlayer(backend='python')"
            ) from exc
        engine = Prolog()
        engine.consult(RULES_PL.as_posix())
        _PROLOG = engine
    return _PROLOG


class _PrologBackend:
    """Asserts the position as facts, queries ``best_move(E, Rule)``, retracts them."""

    def __init__(self) -> None:
        self.engine = _prolog()
        self.board: Board | None = None

    def _run(self, goal: str) -> list[dict[str, Any]]:
        return list(self.engine.query(goal))

    def _load_board(self, board: Board) -> None:
        for fact in ("box(_)", "edge(_)", "side_of(_, _)", "dots(_)"):
            self._run(f"retractall({fact})")
        for b in range(board.num_boxes):
            self.engine.assertz(f"box({b})")
        for e in range(board.E):
            self.engine.assertz(f"edge({e})")
        for b, sides in enumerate(board.BOX_EDGES):
            for e in sides:
                self.engine.assertz(f"side_of({e}, {b})")
        self.engine.assertz(f"dots({board.dots})")
        self.board = board

    def decide(self, edges: int, mover_is_first: bool, board: Board) -> tuple[int, str]:
        if edges == board.full:
            raise ValueError("no legal moves: board is full")
        if board != self.board:
            self._load_board(board)
        for e in range(board.E):
            if (edges >> e) & 1:
                self.engine.assertz(f"drawn({e})")
        self.engine.assertz("first_player(0)")
        self.engine.assertz(f"to_move({0 if mover_is_first else 1})")
        try:
            sols = list(self.engine.query("best_move(E, Rule)", maxresult=1))
        finally:
            for fact in ("drawn(_)", "first_player(_)", "to_move(_)"):
                self._run(f"retractall({fact})")
        if not sols:
            raise RuntimeError("rules.pl found no move")
        rule = sols[0]["Rule"]
        rule = rule.decode() if isinstance(rule, bytes) else str(rule)
        if rule not in RULES:
            raise RuntimeError(f"rules.pl returned unknown rule {rule!r}")
        return int(sols[0]["E"]), rule


class RulePlayer(Player):
    """Deductive player: the first applicable rule picks the move. Deterministic."""

    def __init__(self, backend: Literal["python", "prolog"] = "python", seed: int = 0) -> None:
        if backend not in ("python", "prolog"):
            raise ValueError(f"backend must be 'python' or 'prolog', got {backend!r}")
        self.backend = backend
        self.name = "rules" if backend == "python" else "rules-prolog"
        self.rng = random.Random(seed)  # the rules break ties by edge index and never draw from it
        self._prolog = _PrologBackend() if backend == "prolog" else None
        self.last_rule_fired: str = ""

    def decide(self, edges: int, mover_is_first: bool, board: Board) -> tuple[int, str]:
        """``(move, rule)`` on the raw bitmask with this player's backend."""
        if self._prolog is not None:
            return self._prolog.decide(edges, mover_is_first, board)
        return decide(edges, mover_is_first, board)

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        # The harness always starts games with seat 0, so seat 0 is the first player.
        move, self.last_rule_fired = self.decide(state.edges, state.to_move == 0, state.board)
        return move

    def reset(self) -> None:
        self.last_rule_fired = ""
