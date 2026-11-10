"""Solver 5 (MCTS) tests. All budgets are fixed iteration counts except the
time-limit test, which measures wall-clock time by design."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator

import pytest

from engine.board import Board, boxes_completed, get_board
from engine.game import GameState, apply_move, legal_moves, new_game
from solvers import mcts
from solvers.exact import get_solver
from solvers.mcts import MCTSPlayer, Node, mc_label, reward, rollout


def mask(edges: list[int]) -> int:
    """Bitmask with exactly the given edges drawn."""
    return sum(1 << e for e in edges)


def double_capture_position(to_move: int) -> GameState:
    """1x3 board: boxes 0 and 2 are 3-sided, box 1 has no sides.

    The best line for the player to move: take box 0 (v(0,1)), take box 2 (v(0,2)),
    then draw h(0,1) or h(1,1), which completes nothing and passes the turn; the
    opponent takes box 1. Two captures in a row, then a handover.
    """
    b = get_board(1, 3)
    drawn = [b.h(0, 0), b.h(1, 0), b.v(0, 0), b.h(0, 2), b.h(1, 2), b.v(0, 3)]
    return GameState(b, mask(drawn), (0, 0), to_move)


def walk(node: Node) -> Iterator[Node]:
    """Every node of the subtree rooted at ``node``, depth-first."""
    yield node
    for ch in node.children.values():
        yield from walk(ch)


# rollout
def test_rollout_conserves_boxes_and_is_deterministic() -> None:
    """A 3x3 rollout splits all 9 boxes and is reproducible for a fixed seed."""
    b = get_board(3, 3)
    for seed in range(50):
        net = rollout(0, 0, 0.1, random.Random(seed), b)
        assert -9 <= net <= 9 and net % 2 == 1  # 9 boxes split, so the net is odd
        assert rollout(0, 0, 0.1, random.Random(seed), b) == net


def test_rollout_full_board_and_forced_capture() -> None:
    """A full board gains nothing; a final double capture is credited to the mover."""
    b = get_board(1, 2)
    assert rollout(b.full, 0, 0.1, random.Random(0), b) == 0
    middle = b.v(0, 1)
    edges = b.full & ~(1 << middle) # one move left, completing both boxes
    assert rollout(edges, 1, 0.0, random.Random(0), b) == -2    # player 1 takes both


def test_reward_is_symmetric() -> None:
    """Both reward kinds lie in [0, 1] and satisfy r(-net) = 1 - r(net)."""
    for kind in ("win", "margin"):
        for net in range(-9, 10):
            assert reward(net, kind, 9) + reward(-net, kind, 9) == pytest.approx(1.0)
            assert 0.0 <= reward(net, kind, 9) <= 1.0
    assert reward(0, "win", 4) == 0.5 and reward(1, "win", 4) == 1.0 and reward(-1, "win", 4) == 0.0
    assert reward(4, "margin", 4) == 1.0 and reward(-4, "margin", 4) == 0.0


# flat MC
def test_flat_runs_equal_rollouts_and_takes_the_captures() -> None:
    """Flat MC spends exactly its budget and finds the capture."""
    s = double_capture_position(0)
    p = MCTSPlayer(iterations=400, flat=True, seed=1)
    assert p.name == "mcts-flat"
    move = p.choose_move(s)
    assert p.nodes == 400
    assert move in (s.board.v(0, 1), s.board.v(0, 2))


@pytest.mark.parametrize("flat", [False, True])
def test_returns_legal_moves(flat: bool) -> None:
    """UCT and flat MC return legal moves on random 3x3 positions."""
    b = get_board(3, 3)
    rng = random.Random(3)
    p = MCTSPlayer(iterations=60, flat=flat, seed=2)
    for _ in range(40):
        s = new_game(3, 3)
        for _ in range(rng.randrange(b.E)):
            s, _ = apply_move(s, rng.choice(legal_moves(s)))
        assert p.choose_move(s) in legal_moves(s)
        assert p.depth_reached == 0


def test_single_legal_move_returns_immediately() -> None:
    """With one legal move the player answers at once and runs no iterations."""
    b = get_board(1, 1)
    s = GameState(b, b.full & ~1, (0, 0), 1)
    p = MCTSPlayer(iterations=1000)
    assert p.choose_move(s) == 0 and p.nodes == 0


# perspective
@pytest.mark.parametrize("root_player", [0, 1])
@pytest.mark.parametrize("winner", [0, 1])
def test_perspective_through_double_capture(
    monkeypatch: pytest.MonkeyPatch, root_player: int, winner: int
) -> None:
    """Every rollout is rigged so ``winner`` gets every box: then each node's W must be
    n if the player who moved into it is ``winner`` and 0 otherwise."""
    net0 = 100 if winner == 0 else -100
    monkeypatch.setattr(mcts, "rollout", lambda *_: net0)  # ignores the 5 rollout arguments
    s = double_capture_position(root_player)
    b = s.board
    p = MCTSPlayer(iterations=3000, reward="win", seed=0)   # win rewards are exactly 1 or 0
    p.choose_move(s)
    root = p._root
    assert root is not None and root.n == 3000

    for node in walk(root):
        for e, ch in node.children.items():
            assert ch.player == node.to_move    # derived from the parent, not alternated
            assert ch.k == boxes_completed(node.edges, e, b)
            assert ch.to_move == (node.to_move if ch.k else 1 - node.to_move)
            assert ch.W == (ch.n if ch.player == winner else 0)

    # The double-capture line: capture, capture, hand over, opponent captures.
    line = [b.v(0, 1), b.v(0, 2), b.h(0, 1), b.h(1, 1)]
    node = root
    movers, ks = [], []
    for e in line:
        node = node.children[e]
        movers.append(node.player)
        ks.append(node.k)
    me, other = root_player, 1 - root_player
    assert movers == [me, me, me, other]
    assert ks == [1, 1, 0, 1]


@pytest.mark.parametrize("root_player", [0, 1])
def test_uct_takes_the_double_capture(root_player: int) -> None:
    """UCT finds the capture in the double-capture position for either player."""
    s = double_capture_position(root_player)
    p = MCTSPlayer(iterations=2000, seed=4)
    assert p.choose_move(s) in (s.board.v(0, 1), s.board.v(0, 2))
    assert p.nodes == 2000


# budgets
def test_time_limit_is_respected_on_empty_3x3() -> None:
    """20 consecutive time-bound UCT moves at 0.5 s each return within 0.55 s."""
    s = new_game(3, 3)
    p = MCTSPlayer(seed=0)  # no iterations: time-bound
    for _ in range(20):
        t0 = time.perf_counter()
        move = p.choose_move(s, time_limit=0.5)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.55, elapsed
        assert move in legal_moves(s) and p.nodes > 0


def test_flat_time_limit_is_respected() -> None:
    """Time-bound flat MC moves at 0.5 s return within 0.55 s."""
    s = new_game(3, 3)
    p = MCTSPlayer(flat=True, seed=0)
    for _ in range(3):
        t0 = time.perf_counter()
        p.choose_move(s, time_limit=0.5)
        assert time.perf_counter() - t0 <= 0.55


@pytest.mark.parametrize("flat", [False, True])
def test_iterations_budget_overrides_time_limit(flat: bool) -> None:
    """A fixed iteration budget is used even when a time limit is given."""
    p = MCTSPlayer(iterations=150, flat=flat, seed=0)
    p.choose_move(new_game(3, 3), time_limit=0.001)
    assert p.nodes == 150


# tree reuse
def test_tree_reuse_after_our_move_and_reply_then_reset() -> None:
    """After our move and a reply the root keeps its visits; reset() drops the tree."""
    p = MCTSPlayer(iterations=2000, seed=5)
    s = new_game(3, 3)
    ours = p.choose_move(s)
    first_root = p._root
    assert first_root is not None and first_root.n == 2000

    # Opponent replies with the move our tree explored most after our move.
    after_ours = first_root.children[ours]
    reply = max(after_ours.children, key=lambda e: after_ours.children[e].n)
    reused = after_ours.children[reply]
    visits_before = reused.n
    assert visits_before > 0

    s, k1 = apply_move(s, ours)
    s, k2 = apply_move(s, reply)
    assert k1 == k2 == 0 and s.to_move == 0
    p.choose_move(s)
    root = p._root
    assert root is not None and root.edges == s.edges and root.to_move == s.to_move
    # The kept subtree is the best-explored node for this position (the reply node
    # itself, or a transposition reached with the two moves in the other order).
    assert root.n - 2000 >= visits_before > 0   # the root started with visits > 0
    assert root is reused or root.n - 2000 > visits_before

    p.reset()
    assert p._root is None


def test_no_reuse_without_a_match() -> None:
    """A position absent from the old tree gets a fresh tree."""
    p = MCTSPlayer(iterations=500, seed=6)
    p.choose_move(new_game(3, 3))
    b = get_board(3, 3)
    # Same edges as a depth-1 node, but that node has player 1 to move (no capture);
    # player 0 to move can never occur there, so nothing may be reused.
    assert p._root is not None and p._root.children[0].to_move == 1
    unrelated = GameState(b, 1 << 0, (0, 0), 0)
    p.choose_move(unrelated)
    assert p._root is not None and p._root.n == 500


def test_no_reuse_across_a_capture() -> None:
    """Rewards are net boxes from the root, so a subtree reached through a capture
    would carry statistics measured from a different baseline."""
    s = double_capture_position(0)
    p = MCTSPlayer(iterations=500, seed=7)
    move = p.choose_move(s)
    s2, k = apply_move(s, move)
    assert k == 1
    p.choose_move(s2)
    assert p._root is not None and p._root.n == 500


# mc_label
def all_boxes_capturable_2x2() -> tuple[int, Board]:
    """2x2 with exactly one outer side missing per box: every remaining move
    completes a box, so the player to move takes all 4 under any play."""
    b = get_board(2, 2)
    missing = {b.h(0, 0), b.h(0, 1), b.h(2, 0), b.h(2, 1)}
    edges = b.full & ~mask(list(missing))
    for e in missing:
        assert boxes_completed(edges, e, b) == 1
    return edges, b


def test_mc_label_is_one_when_mover_wins_every_remaining_box() -> None:
    """mc_label is 1.0 when every remaining move gives the mover a box."""
    edges, b = all_boxes_capturable_2x2()
    for first_player in (0, 1):
        assert mc_label(edges, first_player, 200, random.Random(0), b) == 1.0
    one = get_board(1, 2)
    double_capture = one.full & ~(1 << one.v(0, 1))
    assert mc_label(double_capture, 0, 50, random.Random(0), one) == 1.0


def test_mc_label_full_board_is_a_draw() -> None:
    """On a full board nobody gains anything, so the label is 0.5."""
    b = get_board(2, 2)
    assert mc_label(b.full, 0, 10, random.Random(0), b) == 0.5


def test_mc_label_deterministic_and_in_range() -> None:
    """mc_label is seed-deterministic, in [0, 1], and ignores first_player."""
    b = get_board(3, 3)
    rng = random.Random(8)
    for _ in range(30):
        s = new_game(3, 3)
        for _ in range(rng.randrange(b.E)):
            s, _ = apply_move(s, rng.choice(legal_moves(s)))
        a = mc_label(s.edges, 0, 40, random.Random(123), b)
        assert a == mc_label(s.edges, 0, 40, random.Random(123), b)
        # first_player is bookkeeping only
        assert mc_label(s.edges, 1, 40, random.Random(123), b) == a
        assert 0.0 <= a <= 1.0


def test_mc_label_rejects_bad_arguments() -> None:
    """Invalid first_player or n_rollouts raise ValueError."""
    b = get_board(2, 2)
    with pytest.raises(ValueError):
        mc_label(0, 2, 10, random.Random(0), b)
    with pytest.raises(ValueError):
        mc_label(0, 0, 0, random.Random(0), b)


# optimal moves
def unique_optimal_2x2_positions(count: int, seed: int) -> list[tuple[int, int]]:
    """Positions from random play with >= 2 legal moves and exactly one optimal move."""
    b = get_board(2, 2)
    solver = get_solver(2, 2)
    rng = random.Random(seed)
    out: list[tuple[int, int]] = []
    while len(out) < count:
        s = new_game(2, 2)
        for _ in range(rng.randrange(b.E - 1)):
            s, _ = apply_move(s, rng.choice(legal_moves(s)))
        vals = solver.move_values(s.edges)
        if len(vals) < 2:
            continue
        best = max(vals.values())
        optimal = [e for e, v in vals.items() if v == best]
        if len(optimal) == 1:
            out.append((s.edges, optimal[0]))
    return out


def test_uct_finds_unique_optimal_moves_on_2x2() -> None:
    """UCT picks the exact solver's unique optimal move in >= 18 of 20 positions."""
    b = get_board(2, 2)
    positions = unique_optimal_2x2_positions(20, seed=2026)
    hits = 0
    for i, (edges, optimal) in enumerate(positions):
        p = MCTSPlayer(iterations=20000, seed=i)    # default c, eps and reward
        hits += p.choose_move(GameState(b, edges, (0, 0), 0)) == optimal
    assert hits >= 18, f"{hits}/20"
