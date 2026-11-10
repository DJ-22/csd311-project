"""Chain analysis on hand-built positions from ``tests/positions/``.

Position file format::

    # comment
    board: MxN
    expect: chain|loop <box,box,...> open=<k>      (zero or more)
    ---
    <ASCII diagram, 2m+1 lines>

In the diagram, dot rows look like ``+--+  +`` (``--`` = drawn horizontal edge) and
the rows between them have ``|`` at column ``3c`` for a drawn vertical edge ``v(r, c)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.board import Board, get_board, parse_board, sides_drawn
from engine.chains import find_components

POSITIONS = Path(__file__).resolve().parent.parent / "positions"


@dataclass
class Position:
    name: str
    board: Board
    edges: int
    expected: set[tuple[str, frozenset[int], int]]


def parse_diagram(lines: list[str], board: Board) -> int:
    m, n = board.m, board.n
    if len(lines) != 2 * m + 1:
        raise ValueError(f"expected {2 * m + 1} diagram lines, got {len(lines)}")
    edges = 0
    for r in range(m + 1):
        row = lines[2 * r].ljust(3 * n + 1)
        for c in range(n):
            if row[3 * c + 1 : 3 * c + 3] == "--":
                edges |= 1 << board.h(r, c)
    for r in range(m):
        row = lines[2 * r + 1].ljust(3 * n + 1)
        for c in range(n + 1):
            if row[3 * c] == "|":
                edges |= 1 << board.v(r, c)
    return edges


def load_position(path: Path) -> Position:
    text = path.read_text(encoding="utf-8").splitlines()
    sep = text.index("---")
    board: Board | None = None
    expected: set[tuple[str, frozenset[int], int]] = set()
    for line in text[:sep]:
        if line.startswith("board:"):
            board = parse_board(line.split(":", 1)[1].strip())
        elif line.startswith("expect:"):
            kind, boxes, opened = line.split(":", 1)[1].split()
            expected.add((kind, frozenset(int(b) for b in boxes.split(",")), int(opened.split("=")[1])))
    assert board is not None, f"{path.name}: missing 'board:' line"
    return Position(path.stem, board, parse_diagram(text[sep + 1 :], board), expected)


ALL_POSITIONS = sorted(POSITIONS.glob("*.txt"))


def test_at_least_four_positions() -> None:
    assert len(ALL_POSITIONS) >= 4


@pytest.mark.parametrize("path", ALL_POSITIONS, ids=lambda p: p.stem)
def test_position_components(path: Path) -> None:
    pos = load_position(path)
    comps = find_components(pos.edges, pos.board)
    got = {(c.kind, frozenset(c.boxes), c.open_ends) for c in comps}
    assert got == pos.expected
    for c in comps:
        assert c.length == len(c.boxes)


def test_diagram_parser_reads_expected_edges() -> None:
    pos = load_position(POSITIONS / "loop4_2x2.txt")
    b = pos.board
    outer = [b.h(0, 0), b.h(0, 1), b.h(2, 0), b.h(2, 1), b.v(0, 0), b.v(1, 0), b.v(0, 2), b.v(1, 2)]
    assert pos.edges == sum(1 << e for e in outer)


def _adjacent_through_undrawn(edges: int, a: int, b: int, board: Board) -> bool:
    shared = set(board.BOX_EDGES[a]) & set(board.BOX_EDGES[b])
    return len(shared) == 1 and not (edges >> shared.pop()) & 1


@pytest.mark.parametrize("m, n", [(2, 2), (3, 3), (4, 4)])
def test_random_positions_invariants(m: int, n: int) -> None:
    board = get_board(m, n)
    rng = random.Random(7)
    for _ in range(300):
        edges = rng.getrandbits(board.E)
        comps = find_components(edges, board)
        two_sided = {b for b in range(board.num_boxes) if sides_drawn(edges, b, board) == 2}
        covered = [b for c in comps for b in c.boxes]
        # Components partition the 2-sided boxes.
        assert sorted(covered) == sorted(two_sided)
        for c in comps:
            for a, b in zip(c.boxes, c.boxes[1:]):
                assert _adjacent_through_undrawn(edges, a, b, board)
            if c.kind == "loop":
                assert c.length >= 4
                assert _adjacent_through_undrawn(edges, c.boxes[-1], c.boxes[0], board)
                assert c.open_ends == 0
            else:
                assert 0 <= c.open_ends <= 2
