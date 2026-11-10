"""Board symmetries and canonical forms.

A square board has the 8 symmetries of the dihedral group D4; a rectangular board
has 4 (identity, horizontal flip, vertical flip, 180° rotation). Each symmetry is
represented as an edge permutation ``perm`` with ``perm[e]`` = image of edge ``e``.

Permuting a bitmask uses per-byte lookup tables, so :func:`canonical` costs
``#symmetries * ceil(E / 8)`` table lookups.
"""

from __future__ import annotations

from functools import lru_cache

from engine.board import get_board

Perm = tuple[int, ...]
Dot = tuple[int, int]

_CHUNK = 8


def _edge_dots(m: int, n: int) -> list[tuple[Dot, Dot]]:
    """The two dot endpoints of every edge, indexed by edge."""
    board = get_board(m, n)
    ends: list[tuple[Dot, Dot]] = [((0, 0), (0, 0))] * board.E
    for r in range(m + 1):
        for c in range(n):
            ends[board.h(r, c)] = ((r, c), (r, c + 1))
    for r in range(m):
        for c in range(n + 1):
            ends[board.v(r, c)] = ((r, c), (r + 1, c))
    return ends


@lru_cache(maxsize=None)
def symmetries(m: int, n: int) -> tuple[Perm, ...]:
    """Edge permutations for every symmetry of the ``m x n`` board (identity first)."""
    dot_maps = [
        lambda r, c: (r, c),    # identity
        lambda r, c: (r, n - c),    # mirror left-right
        lambda r, c: (m - r, c),    # mirror top-bottom
        lambda r, c: (m - r, n - c),    # rotate 180°
    ]
    if m == n:
        dot_maps += [
            lambda r, c: (c, r),    # transpose
            lambda r, c: (n - c, m - r),    # anti-transpose
            lambda r, c: (c, m - r),    # rotate 90°
            lambda r, c: (n - c, r),    # rotate 270°
        ]
    ends = _edge_dots(m, n)
    index = {frozenset(pair): e for e, pair in enumerate(ends)}
    perms: list[Perm] = []
    for f in dot_maps:
        perm = tuple(index[frozenset((f(*a), f(*b)))] for a, b in ends)
        perms.append(perm)
    return tuple(perms)


@lru_cache(maxsize=None)
def _chunk_tables(m: int, n: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """``tables[s][chunk][byte]`` = permuted bits contributed by ``byte`` at ``chunk``."""
    E = get_board(m, n).E
    n_chunks = (E + _CHUNK - 1) // _CHUNK
    out = []
    for perm in symmetries(m, n):
        per_chunk = []
        for ch in range(n_chunks):
            base = ch * _CHUNK
            row = []
            for byte in range(1 << _CHUNK):
                img = 0
                for bit in range(_CHUNK):
                    e = base + bit
                    if (byte >> bit) & 1 and e < E:
                        img |= 1 << perm[e]
                row.append(img)
            per_chunk.append(tuple(row))
        out.append(tuple(per_chunk))
    return tuple(out)


def apply_symmetry(edges: int, perm: Perm) -> int:
    """Image of the bitmask ``edges`` under the edge permutation ``perm``."""
    out = 0
    e = 0
    while edges:
        if edges & 1:
            out |= 1 << perm[e]
        edges >>= 1
        e += 1
    return out


def canonical(edges: int, m: int, n: int) -> int:
    """Smallest image of ``edges`` over all symmetries of the ``m x n`` board."""
    best = -1
    for tables in _chunk_tables(m, n):
        img = 0
        x = edges
        for row in tables:
            img |= row[x & 0xFF]
            x >>= _CHUNK
        if best < 0 or img < best:
            best = img
    return best
