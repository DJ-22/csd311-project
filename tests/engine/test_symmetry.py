import random

import pytest

from engine.board import get_board
from engine.symmetry import apply_symmetry, canonical, symmetries


@pytest.mark.parametrize("m, n, count", [(1, 1, 8), (2, 2, 8), (3, 3, 8), (2, 3, 4), (3, 2, 4), (1, 2, 4)])
def test_symmetry_count_and_distinct(m: int, n: int, count: int) -> None:
    perms = symmetries(m, n)
    assert len(perms) == count
    assert len(set(perms)) == count
    E = get_board(m, n).E
    assert perms[0] == tuple(range(E))
    for p in perms:
        assert sorted(p) == list(range(E))


@pytest.mark.parametrize("m, n", [(2, 2), (2, 3), (3, 3)])
def test_symmetries_preserve_box_structure(m: int, n: int) -> None:
    b = get_board(m, n)
    box_sets = {frozenset(s) for s in b.BOX_EDGES}
    for p in symmetries(m, n):
        for sides in b.BOX_EDGES:
            assert frozenset(p[e] for e in sides) in box_sets


@pytest.mark.parametrize("m, n", [(2, 2), (2, 3), (3, 3), (4, 4)])
def test_canonical_is_invariant_and_minimal(m: int, n: int) -> None:
    rng = random.Random(1)
    E = get_board(m, n).E
    perms = symmetries(m, n)
    for _ in range(300):
        e = rng.getrandbits(E)
        images = [apply_symmetry(e, p) for p in perms]
        c = canonical(e, m, n)
        assert c == min(images)
        for img in images:
            assert canonical(img, m, n) == c
        assert c.bit_count() == e.bit_count()


def test_canonical_examples() -> None:
    assert canonical(0, 2, 2) == 0
    full = get_board(2, 2).full
    assert canonical(full, 2, 2) == full
