"""Board geometry and precomputed lookup tables.

Edge indexing for an ``m x n`` box board:

* horizontal edge in row ``r in [0, m]``, column ``c in [0, n-1]`` -> ``r*n + c``
* vertical edge in row ``r in [0, m-1]``, column ``c in [0, n]`` -> ``H + r*(n+1) + c``

Box ``(r, c)`` has index ``r*n + c`` and sides top ``h(r, c)``, bottom ``h(r+1, c)``,
left ``v(r, c)``, right ``v(r, c+1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Board:
    """Immutable description of an ``m x n`` box board plus hot-loop lookup tables.

    Use :func:`get_board` to obtain a cached instance.
    """

    m: int
    n: int
    H: int = field(init=False)
    V: int = field(init=False)
    E: int = field(init=False)
    num_boxes: int = field(init=False)
    dots: int = field(init=False)
    full: int = field(init=False, compare=False)
    BOX_EDGES: tuple[tuple[int, int, int, int], ...] = field(init=False, compare=False, repr=False)
    EDGE_BOXES: tuple[tuple[int, ...], ...] = field(init=False, compare=False, repr=False)
    BOX_MASK: tuple[int, ...] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        m, n = self.m, self.n
        if m < 1 or n < 1:
            raise ValueError(f"board must have at least 1x1 boxes, got {m}x{n}")
        H = (m + 1) * n
        V = m * (n + 1)
        E = H + V
        box_edges: list[tuple[int, int, int, int]] = []
        edge_boxes: list[list[int]] = [[] for _ in range(E)]
        for r in range(m):
            for c in range(n):
                b = r * n + c
                sides = (
                    r * n + c,  # top h(r, c)
                    (r + 1) * n + c,    # bottom h(r+1, c)
                    H + r * (n + 1) + c,    # left v(r, c)
                    H + r * (n + 1) + c + 1,    # right  v(r, c+1)
                )
                box_edges.append(sides)
                for e in sides:
                    edge_boxes[e].append(b)
        set_ = object.__setattr__
        set_(self, "H", H)
        set_(self, "V", V)
        set_(self, "E", E)
        set_(self, "num_boxes", m * n)
        set_(self, "dots", (m + 1) * (n + 1))
        set_(self, "full", (1 << E) - 1)
        set_(self, "BOX_EDGES", tuple(box_edges))
        set_(self, "EDGE_BOXES", tuple(tuple(bs) for bs in edge_boxes))
        set_(self, "BOX_MASK", tuple(sum(1 << e for e in sides) for sides in box_edges))

    def h(self, r: int, c: int) -> int:
        """Index of the horizontal edge in row ``r``, column ``c``."""
        if not (0 <= r <= self.m and 0 <= c < self.n):
            raise IndexError(f"h({r}, {c}) out of range for {self.m}x{self.n}")
        return r * self.n + c

    def v(self, r: int, c: int) -> int:
        """Index of the vertical edge in row ``r``, column ``c``."""
        if not (0 <= r < self.m and 0 <= c <= self.n):
            raise IndexError(f"v({r}, {c}) out of range for {self.m}x{self.n}")
        return self.H + r * (self.n + 1) + c

    def box(self, r: int, c: int) -> int:
        """Index of box ``(r, c)``."""
        if not (0 <= r < self.m and 0 <= c < self.n):
            raise IndexError(f"box({r}, {c}) out of range for {self.m}x{self.n}")
        return r * self.n + c


@lru_cache(maxsize=None)
def get_board(m: int, n: int) -> Board:
    """Return the shared :class:`Board` for an ``m x n`` box board."""
    return Board(m, n)


def parse_board(spec: str) -> Board:
    """Parse ``"MxN"`` (boxes) into a board, e.g. ``"3x3"``."""
    try:
        m_str, n_str = spec.lower().split("x")
        return get_board(int(m_str), int(n_str))
    except ValueError as exc:
        raise ValueError(f"invalid board spec {spec!r}; expected e.g. '3x3'") from exc


def sides_drawn(edges: int, box: int, board: Board) -> int:
    """Number of drawn sides (0-4) of ``box`` in the edge bitmask ``edges``."""
    return (edges & board.BOX_MASK[box]).bit_count()


def boxes_completed(edges: int, edge: int, board: Board) -> int:
    """Boxes (0, 1 or 2) that drawing ``edge`` on ``edges`` would complete.

    Assumes ``edge`` is not drawn yet.
    """
    k = 0
    for b in board.EDGE_BOXES[edge]:
        if (edges & board.BOX_MASK[b]).bit_count() == 3:
            k += 1
    return k
