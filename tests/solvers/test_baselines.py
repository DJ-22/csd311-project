import random

import pytest

from engine.board import Board, boxes_completed, get_board
from engine.game import GameState, apply_move, is_terminal, legal_moves, new_game
from solvers.baselines import GreedyPlayer, RandomPlayer, capture_run, greedy_move


def random_positions(board: Board, count: int, seed: int) -> list[int]:
    """Edge masks reached by random play, with a uniformly random number of moves."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        s = GameState(board)
        for _ in range(rng.randrange(board.E)):
            s, _ = apply_move(s, rng.choice(legal_moves(s)))
        out.append(s.edges)
    return out


def creates_three_sided(edges: int, e: int, board: Board) -> bool:
    return any((edges & board.BOX_MASK[b]).bit_count() == 2 for b in board.EDGE_BOXES[e])


@pytest.mark.parametrize("player_cls", [RandomPlayer, GreedyPlayer])
def test_returns_legal_moves(player_cls: type) -> None:
    board = get_board(3, 3)
    player = player_cls(seed=1)
    for edges in random_positions(board, 1000, seed=11):
        s = GameState(board, edges, (0, 0), 0)
        assert player.choose_move(s) in legal_moves(s)


def test_greedy_always_captures_and_never_unsafe() -> None:
    board = get_board(3, 3)
    rng = random.Random(5)
    seen_capture = seen_safe = 0
    for edges in random_positions(board, 1000, seed=12):
        free = [e for e in range(board.E) if not (edges >> e) & 1]
        ks = {e: boxes_completed(edges, e, board) for e in free}
        e = greedy_move(edges, rng, board)
        if max(ks.values()) > 0:
            seen_capture += 1
            assert ks[e] == max(ks.values())    # takes a box, and prefers a double
        elif any(not creates_three_sided(edges, x, board) for x in free):
            seen_safe += 1
            assert not creates_three_sided(edges, e, board)
    assert seen_capture > 50 and seen_safe > 50 # both branches were exercised


def test_greedy_prefers_double_capture() -> None:
    board = get_board(1, 3)
    # Boxes 0 and 1 are 3-sided sharing v(0,1); box 2 is 3-sided too, missing v(0,3).
    missing = {board.v(0, 1), board.v(0, 3)}
    edges = sum(1 << e for e in range(board.E) if e not in missing)
    assert greedy_move(edges, random.Random(0), board) == board.v(0, 1)


def test_greedy_least_damage_sacrifice() -> None:
    # On every position with no capture and no safe move, greedy's choice must
    # minimise the opponent's capture run.
    board = get_board(2, 3)
    rng = random.Random(0)
    checked = 0
    for edges in random_positions(board, 3000, seed=13):
        free = [e for e in range(board.E) if not (edges >> e) & 1]
        if any(boxes_completed(edges, e, board) for e in free):
            continue
        if any(not creates_three_sided(edges, e, board) for e in free):
            continue
        checked += 1
        costs = {e: capture_run(edges | (1 << e), board)[0] for e in free}
        assert costs[greedy_move(edges, rng, board)] == min(costs.values())
    assert checked >= 20


def test_capture_run_counts_whole_chain() -> None:
    board = get_board(1, 3)
    h = sum(1 << e for e in range(board.H))
    # Open the chain at the left end: opponent takes all three boxes.
    captured, after = capture_run(h | (1 << board.v(0, 0)), board)
    assert captured == 3 and after == board.full


def test_greedy_beats_random_on_3x3() -> None:
    from harness.cli import make_player
    from harness.metrics import summarise
    from harness.tournament import match

    rows = match("greedy", "random", make_player, 3, 3, games=500, seed=2026)
    s = summarise(rows, "greedy")
    assert s.games == 500
    assert s.win_rate > 0.9, s


def test_greedy_player_is_deterministic_given_seed() -> None:
    def play(seed: int) -> list[int]:
        a, b = GreedyPlayer(seed), GreedyPlayer(seed + 1)
        s, moves = new_game(3, 3), []
        while not is_terminal(s):
            e = (a if s.to_move == 0 else b).choose_move(s)
            moves.append(e)
            s, _ = apply_move(s, e)
        return moves

    assert play(4) == play(4)
