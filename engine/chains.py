"""Chain and loop analysis, shared by features, solver 3 and solver 4.

* A **chain** is a maximal sequence of boxes with exactly 2 sides drawn, connected
  through shared undrawn edges.
* A **loop** is a chain that closes on itself.
* A **long chain** has length >= 3.

Each 2-sided box has two undrawn sides. Through each one it is either *linked* to
another 2-sided box, or the chain *ends* there (at the board boundary or at a box
with 0, 1 or 3 sides drawn).

``open_ends`` counts the ends of a chain that lead into a 3-sided (capturable) box,
i.e. ends through which the chain has already been opened. It is 0 for loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from engine.board import Board

LONG_CHAIN = 3


@dataclass(frozen=True)
class Component:
    """A chain or loop. ``boxes`` are listed in order along the component."""

    kind: Literal["chain", "loop"]
    boxes: list[int]
    length: int
    open_ends: int


def _neighbours(edges: int, box: int, board: Board) -> list[int | None]:
    """For each undrawn side of ``box``, the box across it (``None`` = boundary)."""
    out: list[int | None] = []
    for e in board.BOX_EDGES[box]:
        if not (edges >> e) & 1:
            other: int | None = None
            for b in board.EDGE_BOXES[e]:
                if b != box:
                    other = b
            out.append(other)
    return out


def find_components(edges: int, board: Board) -> list[Component]:
    """All chains and loops in the position, ordered by their smallest box index."""
    mask = board.BOX_MASK
    sides = [(edges & mk).bit_count() for mk in mask]
    two = {b for b in range(board.num_boxes) if sides[b] == 2}

    links: dict[int, list[int]] = {}
    ends: dict[int, list[int | None]] = {}
    for b in two:
        nbs = _neighbours(edges, b, board)
        links[b] = [x for x in nbs if x is not None and x in two]
        ends[b] = [x for x in nbs if x is None or x not in two]

    seen: set[int] = set()
    comps: list[Component] = []
    for start in sorted(two):
        if start in seen:
            continue
        # Collect the connected component.
        stack = [start]
        members: set[int] = {start}
        while stack:
            b = stack.pop()
            for x in links[b]:
                if x not in members:
                    members.add(x)
                    stack.append(x)
        seen |= members

        is_loop = all(len(links[b]) == 2 for b in members)
        if is_loop:
            first = min(members)
        else:
            first = min(b for b in members if len(links[b]) < 2)
        order = [first]
        prev, cur = -1, first
        while True:
            nxt = [x for x in links[cur] if x != prev and x not in order]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            order.append(cur)

        if is_loop:
            comps.append(Component("loop", order, len(order), 0))
        else:
            open_ends = sum(
                1 for b in members for x in ends[b] if x is not None and sides[x] == 3
            )
            comps.append(Component("chain", order, len(order), open_ends))
    return comps
