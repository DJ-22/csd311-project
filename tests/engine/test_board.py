import pytest

from engine.board import boxes_completed, get_board, parse_board, sides_drawn


@pytest.mark.parametrize(
    "m, n, E, boxes",
    [(1, 1, 4, 1), (2, 2, 12, 4), (2, 3, 17, 6), (3, 3, 24, 9), (4, 4, 40, 16)],
)
def test_edge_and_box_counts(m: int, n: int, E: int, boxes: int) -> None:
    b = get_board(m, n)
    assert b.E == E
    assert b.H == (m + 1) * n
    assert b.V == m * (n + 1)
    assert b.num_boxes == boxes
    assert b.dots == (m + 1) * (n + 1)
    assert b.full == (1 << E) - 1


@pytest.mark.parametrize("m, n", [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)])
def test_box_edges_are_four_distinct(m: int, n: int) -> None:
    b = get_board(m, n)
    assert len(b.BOX_EDGES) == m * n
    for sides in b.BOX_EDGES:
        assert len(set(sides)) == 4
        assert all(0 <= e < b.E for e in sides)


@pytest.mark.parametrize("m, n", [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)])
def test_every_edge_borders_one_or_two_boxes(m: int, n: int) -> None:
    b = get_board(m, n)
    assert len(b.EDGE_BOXES) == b.E
    for e, boxes in enumerate(b.EDGE_BOXES):
        assert len(boxes) in (1, 2)
        for box in boxes:
            assert e in b.BOX_EDGES[box]
    # Each box contributes exactly 4 (edge, box) incidences.
    assert sum(len(bs) for bs in b.EDGE_BOXES) == 4 * b.num_boxes


@pytest.mark.parametrize("m, n", [(1, 1), (2, 2), (2, 3)])
def test_indexing_formula(m: int, n: int) -> None:
    b = get_board(m, n)
    H = (m + 1) * n
    for r in range(m):
        for c in range(n):
            top, bottom, left, right = b.BOX_EDGES[b.box(r, c)]
            assert top == r * n + c == b.h(r, c)
            assert bottom == (r + 1) * n + c == b.h(r + 1, c)
            assert left == H + r * (n + 1) + c == b.v(r, c)
            assert right == H + r * (n + 1) + c + 1 == b.v(r, c + 1)


def test_boundary_edges_border_one_box() -> None:
    b = get_board(2, 3)
    for c in range(3):
        assert len(b.EDGE_BOXES[b.h(0, c)]) == 1
        assert len(b.EDGE_BOXES[b.h(2, c)]) == 1
        assert len(b.EDGE_BOXES[b.h(1, c)]) == 2
    for r in range(2):
        assert len(b.EDGE_BOXES[b.v(r, 0)]) == 1
        assert len(b.EDGE_BOXES[b.v(r, 3)]) == 1
        assert len(b.EDGE_BOXES[b.v(r, 1)]) == 2


def test_sides_drawn_and_boxes_completed() -> None:
    b = get_board(1, 2)
    top, bottom, left, right = b.BOX_EDGES[0]
    edges = (1 << top) | (1 << bottom)
    assert sides_drawn(edges, 0, b) == 2
    assert sides_drawn(edges, 1, b) == 0
    assert boxes_completed(edges, left, b) == 0
    edges |= 1 << left
    assert boxes_completed(edges, right, b) == 1


def test_parse_board() -> None:
    assert parse_board("2x3") is get_board(2, 3)
    with pytest.raises(ValueError):
        parse_board("3by3")
