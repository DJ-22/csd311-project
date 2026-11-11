"""Solver 5: Monte Carlo player - flat Monte Carlo and UCT - plus ``mc_label``.

Rewards follow the value convention: they are computed from the **net boxes
gained from the root of the search onward**; the score so far never enters.

* ``reward="win"``: 1 if the player gets more of the remaining boxes than the
  opponent, 0.5 if equal, 0 otherwise.
* ``reward="margin"``: ``(net + B) / 2B`` with ``B`` the number of boxes.

Both are symmetric (``r_opponent = 1 - r``), so one rollout result can be
credited to either player.

Every node stores ``W`` from the perspective of ``player``, the player who made
the move into the node. ``player`` is the parent's ``to_move``, never derived by
alternating: a move that completes a box keeps the turn (extra-turn rule).
"""

from __future__ import annotations

import math
import random
import time
from typing import Literal

from engine.board import Board, boxes_completed
from engine.game import GameState, legal_moves
from solvers.base import Player
from solvers.baselines import greedy_move

DEFAULT_TIME_LIMIT = 1.0
SAFETY_FRACTION = 0.05  # stop this fraction of the time limit early ...
SAFETY_SECONDS = 0.01   # ... plus this absolute margin
LABEL_EPS = 0.1

RewardKind = Literal["win", "margin"]


def rollout(edges: int, to_move: int, eps: float, rng: random.Random, board: Board) -> int:
    """Play ``edges`` to the end; return boxes won by player 0 minus player 1.

    ``to_move`` moves first. Each move is ``greedy_move`` with probability
    ``1 - eps``, otherwise a uniformly random legal move. Runs on the raw bitmask.
    """
    full = board.full
    E = board.E
    gain0 = gain1 = 0
    while edges != full:
        if eps > 0.0 and rng.random() < eps:
            free = [e for e in range(E) if not (edges >> e) & 1]
            e = free[rng.randrange(len(free))]
        else:
            e = greedy_move(edges, rng, board)
        k = boxes_completed(edges, e, board)
        edges |= 1 << e
        if k:
            if to_move == 0:
                gain0 += k
            else:
                gain1 += k
        else:
            to_move ^= 1
    return gain0 - gain1


def reward(net: int, kind: RewardKind, num_boxes: int) -> float:
    """Reward in ``[0, 1]`` for a player whose net box gain from the root is ``net``."""
    if kind == "win":
        return 1.0 if net > 0 else 0.5 if net == 0 else 0.0
    return (net + num_boxes) / (2 * num_boxes)


def mc_label(edges: int, first_player: int, n_rollouts: int, rng: random.Random, board: Board) -> float:
    """Monte Carlo estimate of P(player to move wins) from ``edges``.

    "Wins" means gets more of the remaining boxes (draw = 0.5); scores so far are
    not part of the input. Rollouts are greedy with ``eps = 0.1``.
    ``first_player`` is accepted for dataset bookkeeping (solver 7) and does not
    affect the estimate: who moved first does not change the remaining game.
    """
    if first_player not in (0, 1):
        raise ValueError(f"first_player must be 0 or 1, got {first_player}")
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be >= 1")
    total = 0.0
    for _ in range(n_rollouts):
        # The player to move is labelled 0, so the rollout's net is the mover's net.
        total += reward(rollout(edges, 0, LABEL_EPS, rng, board), "win", board.num_boxes)
    return total / n_rollouts


class Node:
    """UCT tree node. ``W`` is from the perspective of ``player`` (who moved into it)."""

    __slots__ = ("edges", "to_move", "player", "k", "n", "W", "children", "untried")

    def __init__(self, edges: int, to_move: int, player: int, k: int, untried: list[int]) -> None:
        self.edges = edges
        self.to_move = to_move
        self.player = player
        self.k = k  # boxes completed by the move into this node
        self.n = 0
        self.W = 0.0
        self.children: dict[int, Node] = {}
        self.untried = untried


class MCTSPlayer(Player):
    """UCT player (``flat=False``) or flat Monte Carlo player (``flat=True``).

    Defaults ``c=0.5`` and ``reward="margin"`` were tuned (see the spec's decision
    log); ``"win"`` scores every losing (or every winning) move the same, so the
    search stops distinguishing a 2-box loss from a 4-box loss.
    """

    name = "mcts"

    def __init__(
        self,
        iterations: int | None = None,
        c: float = 0.5,
        rollout: Literal["greedy", "random"] = "greedy",
        eps: float = 0.1,
        reward: RewardKind = "margin",
        flat: bool = False,
        seed: int = 0,
    ) -> None:
        if rollout not in ("greedy", "random"):
            raise ValueError(f"rollout must be 'greedy' or 'random', got {rollout!r}")
        if reward not in ("win", "margin"):
            raise ValueError(f"reward must be 'win' or 'margin', got {reward!r}")
        if iterations is not None and iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = iterations
        self.c = c
        self.rollout = rollout
        self.eps = 1.0 if rollout == "random" else eps
        self.reward = reward
        self.flat = flat
        self.rng = random.Random(seed)
        self.name = "mcts-flat" if flat else "mcts"
        self.nodes = 0  # iterations (UCT) or rollouts (flat) in the last choose_move
        self.depth_reached = 0  # not a depth-limited search: harness skips b*
        self._root: Node | None = None
        self._board: Board | None = None

    def reset(self) -> None:
        self._root = None
        self._board = None

    # budget
    def _deadline(self, time_limit: float | None) -> float | None:
        if self.iterations is not None:
            return None
        limit = DEFAULT_TIME_LIMIT if time_limit is None else time_limit
        return time.perf_counter() + max(0.0, limit * (1 - SAFETY_FRACTION) - SAFETY_SECONDS)

    def _within_budget(self, done: int, deadline: float | None) -> bool:
        if deadline is None:
            return done < self.iterations   # type: ignore[operator]
        return done == 0 or time.perf_counter() < deadline

    # choose_move
    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        self.nodes = 0
        moves = legal_moves(state)
        if not moves:
            raise ValueError("no legal moves: board is full")
        if len(moves) == 1:
            self._root = None
            return moves[0]
        deadline = self._deadline(time_limit)
        if self.flat:
            return self._flat(state, moves, deadline)
        return self._uct(state, deadline)

    # flat (stage A)
    def _flat(self, state: GameState, moves: list[int], deadline: float | None) -> int:
        """Equal rollouts per root move (round-robin); best mean reward wins."""
        board = state.board
        me = state.to_move
        totals = [0.0] * len(moves)
        counts = [0] * len(moves)
        children = []
        for e in moves:
            k = boxes_completed(state.edges, e, board)
            children.append((state.edges | (1 << e), me if k else 1 - me, k))
        done = 0
        while self._within_budget(done, deadline):
            i = done % len(moves)
            edges, to_move, k = children[i]
            net0 = rollout(edges, to_move, self.eps, self.rng, board)
            net_me = (net0 if me == 0 else -net0) + k
            totals[i] += reward(net_me, self.reward, board.num_boxes)
            counts[i] += 1
            done += 1
        self.nodes = done
        best = max(range(len(moves)), key=lambda i: (totals[i] / counts[i] if counts[i] else -1.0))
        return moves[best]

    # UCT (stage B)
    def _new_node(self, edges: int, to_move: int, player: int, k: int, board: Board) -> Node:
        untried = [e for e in range(board.E) if not (edges >> e) & 1]
        self.rng.shuffle(untried)
        return Node(edges, to_move, player, k, untried)

    def _find_reusable(self, state: GameState) -> Node | None:
        """Node of the previous tree matching ``state``, reached only via non-capturing moves.

        Rewards are measured from the root onward, so a subtree is only reusable
        if no boxes were won between the old root and the new one.
        """
        target = state.edges
        if self._root is None or self._root.edges & ~target:
            return None
        best: Node | None = None
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.edges == target:
                # Several move orders can reach the same position; keep the best-explored one.
                if node.to_move == state.to_move and (best is None or node.n > best.n):
                    best = node
                continue
            for ch in node.children.values():
                if ch.k == 0 and ch.edges & ~target == 0:
                    stack.append(ch)
        return best

    def _uct(self, state: GameState, deadline: float | None) -> int:
        board = state.board
        root = self._find_reusable(state) if self._board is board else None
        self._board = board
        if root is None:
            root = self._new_node(state.edges, state.to_move, 1 - state.to_move, 0, board)
        self._root = root

        c = self.c
        eps = self.eps
        rng = self.rng
        kind = self.reward
        nb = board.num_boxes
        log = math.log
        sqrt = math.sqrt
        done = 0
        while self._within_budget(done, deadline):
            node = root
            path = [root]
            gain = [0, 0]
            # 1. Selection through fully expanded nodes.
            while not node.untried and node.children:
                ln_n = log(node.n)
                best_score = -math.inf
                best_child = node
                for ch in node.children.values():
                    score = ch.W / ch.n + c * sqrt(ln_n / ch.n)
                    if score > best_score:
                        best_score, best_child = score, ch
                node = best_child
                path.append(node)
                gain[node.player] += node.k
            # 2. Expansion of one untried move.
            if node.untried:
                e = node.untried.pop()
                k = boxes_completed(node.edges, e, board)
                mover = node.to_move
                child = self._new_node(node.edges | (1 << e), mover if k else 1 - mover, mover, k, board)
                node.children[e] = child
                node = child
                path.append(node)
                gain[mover] += k
            # 3. Rollout.
            net0 = gain[0] - gain[1] + rollout(node.edges, node.to_move, eps, rng, board)
            # 4. Backpropagation, each node from the view of the player who moved into it.
            r0 = reward(net0, kind, nb)
            for nd in path:
                nd.n += 1
                nd.W += r0 if nd.player == 0 else 1.0 - r0
            done += 1
        self.nodes = done
        move, _ = max(root.children.items(), key=lambda item: (item[1].n, item[1].W / item[1].n))
        return move
