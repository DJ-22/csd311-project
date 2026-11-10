import random

import numpy as np
import pytest

from engine.board import get_board, sides_drawn
from engine.game import GameState, new_game
from features.features import FEATURE_NAMES, NUM_FEATURES, phi


def test_shape_and_dtype() -> None:
    f = phi(new_game(3, 3), first_player=0)
    assert f.shape == (NUM_FEATURES,) == (12,)
    assert f.dtype == np.float32
    assert len(FEATURE_NAMES) == 12


def test_empty_board() -> None:
    s = new_game(3, 3)
    f = phi(s, first_player=0)
    assert f[0] == 9 and f[1] == f[2] == f[3] == 0
    assert f[4] == 24   # every edge is safe on an empty board
    assert f[5] == f[6] == f[7] == f[8] == f[9] == 0
    # 16 dots + 0 long chains = even -> favours the first player, who is to move.
    assert f[10] == 1.0
    assert f[11] == 0.0
    assert phi(s, first_player=1)[10] == -1.0


def test_parity_flips_with_dot_count() -> None:
    # 2x3 board has 12 dots (even); 2x2 has 9 dots (odd).
    assert phi(new_game(2, 3), first_player=0)[10] == 1.0
    assert phi(new_game(2, 2), first_player=0)[10] == -1.0


def test_chain_features_three_row_chains() -> None:
    # 3x3 with every horizontal edge drawn: three rows, each a long chain of 3.
    b = get_board(3, 3)
    edges = sum(1 << e for e in range(b.H))
    f = phi(GameState(b, edges, (0, 0), 0), first_player=0)
    assert f[2] == 9
    assert f[4] == 0    # every vertical edge touches a 2-sided box
    assert f[5] == 3 and f[6] == 9 and f[7] == 0
    assert f[8] == 0 and f[9] == 0
    # 16 dots + 3 long chains = 19, odd -> favours the second player.
    assert f[10] == -1.0
    assert f[11] == pytest.approx(12 / 24)


def test_loop_features() -> None:
    b = get_board(2, 2)
    outer = [b.h(0, 0), b.h(0, 1), b.h(2, 0), b.h(2, 1), b.v(0, 0), b.v(1, 0), b.v(0, 2), b.v(1, 2)]
    f = phi(GameState(b, sum(1 << e for e in outer), (0, 0), 1), first_player=0)
    assert f[8] == 1 and f[9] == 4
    assert f[5] == 0 and f[7] == 0


def test_box_counts_match_engine_on_random_positions() -> None:
    b = get_board(3, 3)
    rng = random.Random(3)
    for _ in range(200):
        edges = rng.getrandbits(b.E)
        f = phi(GameState(b, edges, (0, 0), 0), first_player=0)
        for k in range(4):
            assert f[k] == sum(1 for box in range(9) if sides_drawn(edges, box, b) == k)
        assert f[0] + f[1] + f[2] + f[3] <= 9
        assert f[6] + f[9] <= f[2]
