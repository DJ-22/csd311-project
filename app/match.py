"""One game in the app: seats, moves, box owners and the result. No pygame.

Seat 0 always moves first. The solvers assume the game's first player is seat 0
(as in the harness), so the app never starts a game with seat 1 to move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.game import GameState, apply_move, is_terminal, legal_moves, new_game
from solvers.base import Player


@dataclass
class Seat:
    """A human (``player is None``) or a solver."""

    name: str
    player: Player | None = None

    @property
    def is_human(self) -> bool:
        return self.player is None


@dataclass(frozen=True)
class MoveRecord:
    seat: int
    edge: int
    boxes: int


@dataclass
class Match:
    m: int
    n: int
    seats: tuple[Seat, Seat]
    state: GameState = field(init=False)
    owners: list[int] = field(init=False)  # box -> seat that claimed it, -1 if open
    edge_owner: dict[int, int] = field(init=False)  # edge -> seat that drew it
    history: list[MoveRecord] = field(init=False)

    def __post_init__(self) -> None:
        self.state = new_game(self.m, self.n, to_move=0)
        self.owners = [-1] * self.state.board.num_boxes
        self.edge_owner = {}
        self.history = []
        for seat in self.seats:
            if seat.player is not None:
                seat.player.reset()

    @property
    def to_move(self) -> int:
        return self.state.to_move

    @property
    def current(self) -> Seat:
        return self.seats[self.state.to_move]

    @property
    def over(self) -> bool:
        return is_terminal(self.state)

    @property
    def last(self) -> MoveRecord | None:
        return self.history[-1] if self.history else None

    def legal(self) -> list[int]:
        return legal_moves(self.state)

    def play(self, edge: int) -> int:
        """Draw ``edge`` for the seat to move; return boxes completed (the same seat moves again if > 0)."""
        if self.over:
            raise ValueError("the game is over")
        seat = self.state.to_move
        before = self.state.edges
        self.state, k = apply_move(self.state, edge)
        self.edge_owner[edge] = seat
        if k:
            board = self.state.board
            for box in board.EDGE_BOXES[edge]:
                mask = board.BOX_MASK[box]
                if (self.state.edges & mask) == mask and (before & mask) != mask:
                    self.owners[box] = seat
        self.history.append(MoveRecord(seat, edge, k))
        return k

    @property
    def winner(self) -> int | None:
        """Seat with more boxes at the end, -1 for a draw, ``None`` while the game is on."""
        if not self.over:
            return None
        s0, s1 = self.state.scores
        return 0 if s0 > s1 else 1 if s1 > s0 else -1
