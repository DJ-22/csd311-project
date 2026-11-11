"""Solver 4 (rule-based player): hand-built positions, rule semantics, backend parity.

Rule positions are ``tests/positions/rules_*.txt`` in the shared format (see
``tests/engine/test_chains.py``), with extra header lines read here::

    rule: <rule fired for the first move>
    move: h(r,c) | v(r,c)          first move
    turn: <move> <move> ...        every move the player makes before the turn passes
    gain: <k>                      boxes the player takes during that turn
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from engine.board import Board, boxes_completed, get_board
from engine.chains import LONG_CHAIN, find_components
from engine.game import GameState, apply_move, is_terminal
from harness.cli import PLAYERS, make_player
from solvers.rules.rules_player import (
    RULES,
    PrologUnavailableError,
    RulePlayer,
    capture_runs,
    decide,
    safe_moves,
)
from tests.engine.test_chains import POSITIONS, load_position

RULE_POSITIONS = sorted(POSITIONS.glob("rules_*.txt"))
EDGE_TOKEN = re.compile(r"([hv])\((\d+),\s*(\d+)\)")


def parse_edge(token: str, board: Board) -> int:
    match = EDGE_TOKEN.fullmatch(token.strip())
    assert match, f"bad edge token {token!r}"
    kind, r, c = match.group(1), int(match.group(2)), int(match.group(3))
    return board.h(r, c) if kind == "h" else board.v(r, c)


def read_header(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[: lines.index("---")]
    return {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in header if ":" in ln and not ln.startswith("#"))}


def play_turn(player: RulePlayer, state: GameState) -> tuple[list[int], list[str], GameState]:
    """Let ``player`` move until the turn passes or the game ends."""
    mover = state.to_move
    moves: list[int] = []
    rules: list[str] = []
    while not is_terminal(state) and state.to_move == mover:
        move = player.choose_move(state)
        moves.append(move)
        rules.append(player.last_rule_fired)
        state, _ = apply_move(state, move)
    return moves, rules, state


def test_rule_positions_present() -> None:
    names = {p.stem for p in RULE_POSITIONS}
    assert {"rules_double_deal_3x3", "rules_hard_hearted_3x3", "rules_last_chain_3x3"} <= names


@pytest.mark.parametrize("path", RULE_POSITIONS, ids=lambda p: p.stem)
def test_rule_position(path: Path) -> None:
    pos = load_position(path)
    header = read_header(path)
    board = pos.board
    state = GameState(board, pos.edges, (0, 0), 0)

    player = RulePlayer(backend="python")
    first = player.choose_move(state)
    assert first == parse_edge(header["move"], board)
    assert player.last_rule_fired == header["rule"]

    moves, _, end = play_turn(RulePlayer(backend="python"), state)
    assert moves == [parse_edge(t, board) for t in header["turn"].split()]
    assert end.scores[0] == int(header["gain"])


def _state(name: str) -> GameState:
    pos = load_position(POSITIONS / f"{name}.txt")
    return GameState(pos.board, pos.edges, (0, 0), 0)


def test_double_deal_leaves_a_domino() -> None:
    """(a) Two long chains: all but 2 taken, then a domino is handed over."""
    moves, rules, end = play_turn(RulePlayer(), _state("rules_double_deal_3x3"))
    assert rules == ["endgame_double_deal"] * len(moves)
    assert boxes_completed(_state("rules_double_deal_3x3").edges, moves[-1], end.board) == 0
    (run,) = capture_runs(end.edges, end.board)
    assert run.two_ended and len(run.boxes) == 2  # a domino: both boxes taken by one edge
    # The opponent must then open one of the two untouched 3-chains.
    assert sorted(c.length for c in find_components(end.edges, end.board)) == [3, 3]


def test_loop_double_deal_leaves_two_dominoes() -> None:
    moves, rules, end = play_turn(RulePlayer(), _state("rules_loop_double_deal_3x3"))
    assert rules == ["endgame_double_deal"]
    runs = capture_runs(end.edges, end.board)
    assert len(runs) == 2 and all(r.two_ended and len(r.boxes) == 2 for r in runs)


def test_hard_hearted_handout() -> None:
    """(b) The 2-chain is opened by its middle edge: two separate single boxes."""
    state = _state("rules_hard_hearted_3x3")
    player = RulePlayer()
    move = player.choose_move(state)
    assert player.last_rule_fired == "min_sacrifice"
    after, k = apply_move(state, move)
    assert k == 0
    runs = capture_runs(after.edges, after.board)
    assert len(runs) == 2 and all(len(r.boxes) == 1 for r in runs)
    assert not any(r.sacrifice for r in runs)  # nothing the opponent can double-deal


def test_last_chain_taken_in_full() -> None:
    """(c) No other long chain or loop: no double-deal, every box is captured."""
    moves, rules, end = play_turn(RulePlayer(), _state("rules_last_chain_3x3"))
    assert rules == ["capture"] * 3
    assert is_terminal(end) and end.scores[0] == 3


# rule semantics on random positions


def _random_states(board: Board, rng: random.Random, count: int) -> list[tuple[int, bool]]:
    out = []
    while len(out) < count:
        edges = rng.getrandbits(board.E)
        if edges != board.full:
            out.append((edges, rng.random() < 0.5))
    return out


def _parity_ok(edges: int, board: Board, mover_is_first: bool) -> bool:
    lc = sum(1 for c in find_components(edges, board) if c.kind == "chain" and c.length >= LONG_CHAIN)
    return ((board.dots + lc) % 2 == 0) == mover_is_first


@pytest.mark.parametrize("m, n", [(1, 1), (2, 2), (2, 3), (3, 3), (4, 4)])
def test_rules_fire_only_when_they_apply(m: int, n: int) -> None:
    board = get_board(m, n)
    rng = random.Random(100 * m + n)
    for edges, mif in _random_states(board, rng, 300):
        move, rule = decide(edges, mif, board)
        assert rule in RULES
        assert not (edges >> move) & 1
        runs = capture_runs(edges, board)
        safe = safe_moves(edges, board)
        k = boxes_completed(edges, move, board)
        if rule == "endgame_double_deal":
            assert any(r.sacrifice for r in runs)
            if k == 0:  # the decline itself leaves only dominoes behind
                after = capture_runs(edges | (1 << move), board)
                assert all(r.two_ended and len(r.boxes) == 2 for r in after)
        elif rule == "capture":
            assert k > 0
        else:
            assert not runs, "a capturable box was left without rule 1 declining it"
            if rule == "parity_safe_move":
                good = [e for e in safe if _parity_ok(edges | (1 << e), board, mif)]
                assert move == min(good)
            elif rule == "safe_move":
                assert safe and move == min(safe)
                assert not any(_parity_ok(edges | (1 << e), board, mif) for e in safe)
            else:
                assert rule == "min_sacrifice" and not safe


def test_rule_player_api() -> None:
    board = get_board(3, 3)
    player = RulePlayer(backend="python", seed=5)
    assert player.name == "rules" and player.last_rule_fired == ""
    state = GameState(board, 0, (0, 0), 0)
    move = player.choose_move(state)
    assert player.last_rule_fired in RULES
    assert state.edges == 0  # not mutated
    player.reset()
    assert player.last_rule_fired == ""
    assert move == RulePlayer(seed=99).choose_move(state)  # deterministic
    with pytest.raises(ValueError):
        RulePlayer(backend="lisp")  # type: ignore[arg-type]


def test_rules_registered_in_cli() -> None:
    player = make_player("rules", 0)
    assert isinstance(player, RulePlayer) and player.backend == "python" and player.name == "rules"
    assert "rules-prolog" in PLAYERS


# Prolog backend


@pytest.fixture
def prolog_player() -> RulePlayer:
    """A Prolog-backed player; skips unless pyswip imports and finds SWI-Prolog."""
    try:
        pytest.importorskip("pyswip")
    except Exception as exc:  # pyswip installed but SWI-Prolog missing: SwiPrologNotFoundError, not ImportError
        pytest.skip(f"SWI-Prolog unavailable: {exc}")
    return RulePlayer(backend="prolog")


def test_prolog_backend_missing_gives_clear_error() -> None:
    try:
        import pyswip  # noqa: F401
    except Exception:
        with pytest.raises(PrologUnavailableError, match="SWI-Prolog|pyswip"):
            RulePlayer(backend="prolog")
    else:
        pytest.skip("SWI-Prolog is available; the error path cannot be exercised")


def test_prolog_matches_python_on_rule_positions(prolog_player: RulePlayer) -> None:
    assert prolog_player.name == "rules-prolog"
    for path in RULE_POSITIONS:
        pos = load_position(path)
        assert prolog_player.decide(pos.edges, True, pos.board) == decide(pos.edges, True, pos.board), path.stem


def test_prolog_matches_python_on_random_positions(prolog_player: RulePlayer) -> None:
    rng = random.Random(500)
    boards = [get_board(2, 2), get_board(2, 3), get_board(3, 3)]
    for i in range(500):
        board = boards[i % len(boards)]
        edges = rng.getrandbits(board.E)
        if edges == board.full:
            continue
        mif = rng.random() < 0.5
        assert prolog_player.decide(edges, mif, board) == decide(edges, mif, board), f"edges={edges:#x}"
