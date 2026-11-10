import pytest

from engine.game import GameState, apply_move, is_terminal, legal_moves, new_game


def test_new_game_and_legal_moves() -> None:
    s = new_game(2, 2)
    assert s.edges == 0 and s.scores == (0, 0) and s.to_move == 0
    assert legal_moves(s) == list(range(12))
    assert not is_terminal(s)


def test_non_completing_move_passes_turn() -> None:
    s = new_game(2, 2)
    s2, k = apply_move(s, 0)
    assert k == 0
    assert s2.to_move == 1
    assert s2.scores == (0, 0)
    assert 0 not in legal_moves(s2)
    # immutability: original untouched
    assert s.edges == 0 and s.to_move == 0


def test_single_completion_grants_extra_turn() -> None:
    s = new_game(1, 1)
    for e in (0, 1, 2):
        s, k = apply_move(s, e)
        assert k == 0
    mover = s.to_move
    s, k = apply_move(s, 3)
    assert k == 1
    assert s.to_move == mover   # extra turn
    assert s.scores[mover] == 1 and s.scores[1 - mover] == 0
    assert is_terminal(s)
    assert legal_moves(s) == []


def test_double_box_completion() -> None:
    # 1x2 board: draw everything except the shared middle edge v(0,1).
    s = new_game(1, 2)
    middle = s.board.v(0, 1)
    others = [e for e in range(s.board.E) if e != middle]
    s = GameState(s.board, sum(1 << e for e in others), (0, 0), 1)
    s2, k = apply_move(s, middle)
    assert k == 2
    assert s2.to_move == 1  # same player continues
    assert s2.scores == (0, 2)
    assert is_terminal(s2)


def test_illegal_moves_raise() -> None:
    s = new_game(1, 1)
    s2, _ = apply_move(s, 0)
    with pytest.raises(ValueError):
        apply_move(s2, 0)
    with pytest.raises(ValueError):
        apply_move(s, 4)
    with pytest.raises(ValueError):
        apply_move(s, -1)


def test_scores_sum_to_boxes_at_end() -> None:
    import random

    rng = random.Random(0)
    for _ in range(50):
        s = new_game(3, 3)
        while not is_terminal(s):
            s, _ = apply_move(s, rng.choice(legal_moves(s)))
        assert sum(s.scores) == 9
