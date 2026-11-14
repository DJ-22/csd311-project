"""Which solvers can play on a board, building them for a board, and running a move off the UI thread."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.game import GameState
from harness.cli import make_player
from solvers.base import Player
from solvers.mcts import MCTSPlayer
from solvers.supervised.dataset import DATA_DIR
from solvers.tuned import tuned_player
from solvers.tuned.fitness import FITNESS_DEPTH

# Which bots the app offers on which board (the team's rule):
# * A bot is always offered on the boards it was built for ("home" boards): boards it was
#   optimised for (tuned weights, MCTS settings, a learned model trained there, an exact
#   solve), and for the untrained solvers the boards every experiment ran on (2x2, 2x3, 3x3).
# * On any other board it is offered only if it performs well: it beat Greedy in at least
#   MIN_VS_GREEDY of its test games there (`python -m experiments.optimize_boards` runs the
#   tests and stores them in data/board_opt/greedy_tests.json).
# Learned bots cannot play a board without a model trained on it.
# Tuned bots play at their tuning depth (FITNESS_DEPTH = 2) on the boards tuned by
# experiments.optimize_boards. Their weights put large values on counts that grow with the
# board (safe moves, untouched boxes); at depth 2 that cancels out between the moves compared,
# but deeper search compares positions with different players to move and gives boxes away
# (5x5: 3-48% against Greedy at 1 s per move). On 3x3 they keep 1 s iterative deepening, as in E1.
BOARDS: tuple[tuple[int, int], ...] = ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (5, 5))
TESTED_BOARDS = frozenset({(2, 2), (2, 3), (3, 3)})
EXACT_BOARDS = frozenset({(2, 2), (2, 3), (3, 3)})  # 3x3 solves in ~50 s on the first move
MIN_VS_GREEDY = 0.80
TUNED_KINDS = ("search-ga", "search-tabu", "search-sa", "search-hc")
MCTS_KINDS = ("mcts", "mcts-flat")
LEARNED_KINDS = ("search-learned", "learned")


@dataclass(frozen=True)
class BotOption:
    """One entry of the bot menu: ``key`` is a harness player name (plus ``:<model path>`` for learned bots)."""

    key: str
    label: str
    description: str


# Menu order; the kind is the key before any ":<model path>".
BOT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("exact", "Exact (perfect play)", "Solver 2: solves the whole game."),
    ("search", "Heuristic search", "Solver 3: alpha-beta with iterative deepening and a hand-set evaluation."),
    ("search-learned", "Search + learned", "Solver 7: heuristic search using a trained network as its evaluation."),
    ("search-ga", "Search tuned by a genetic algorithm", "Solver 6: heuristic search with GA-tuned weights."),
    ("search-tabu", "Search tuned by tabu search", "Solver 6: heuristic search with tabu-tuned weights."),
    ("search-sa", "Search tuned by simulated annealing", "Solver 6: heuristic search with SA-tuned weights."),
    ("search-hc", "Search tuned by hill climbing", "Solver 6: heuristic search with HC-tuned weights."),
    ("mcts", "MCTS (UCT)", "Solver 5: Monte Carlo tree search with Greedy rollouts."),
    ("mcts-flat", "Flat Monte Carlo", "Solver 5: Monte Carlo rollouts for each move, no tree."),
    ("rules", "Rule-based", "Solver 4: chain rules (long chain rule, double-dealing)."),
    ("greedy", "Greedy", "Solver 1: takes boxes when it can, otherwise avoids giving them away."),
    ("learned", "Learned, one move ahead", "Solver 7: picks the move a trained network rates best, no search."),
    ("random", "Random", "Solver 1: picks any legal move."),
)
DEFAULT_BOT = "search"

_MODEL_BOARD = re.compile(r"_(\d+)x(\d+)_")


# ----- where per-board data lives (all under data/, git-ignored) ------------------------
def tuned_dir(m: int, n: int, data_root: Path = DATA_DIR) -> Path:
    """Tuned weights for a board: E3's 3x3 weights in data/tuned/, others in data/tuned/<m>x<n>/."""
    return data_root / "tuned" if (m, n) == (3, 3) else data_root / "tuned" / f"{m}x{n}"


def board_opt_dir(data_root: Path = DATA_DIR) -> Path:
    return data_root / "board_opt"


def mcts_params_path(kind: str, m: int, n: int, data_root: Path = DATA_DIR) -> Path:
    return board_opt_dir(data_root) / f"{kind}_{m}x{n}.json"


def greedy_tests(data_root: Path = DATA_DIR) -> dict[str, dict[str, list[float]]]:
    """``{kind: {"MxN": [points, games]}}`` from the board optimisation runs (empty if none yet)."""
    path = board_opt_dir(data_root) / "greedy_tests.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def vs_greedy(kind: str, m: int, n: int, data_root: Path = DATA_DIR) -> float | None:
    rec = greedy_tests(data_root).get(kind, {}).get(f"{m}x{n}")
    return None if not rec else rec[0] / rec[1]


def _model_label(path: Path) -> str:
    """``e4_mlp-64x64_both_reg_3x3_s0.pkl`` -> ``mlp-64x64 both reg``."""
    stem = re.sub(r"^e\d+_", "", path.stem)
    stem = _MODEL_BOARD.split(stem)[0] if _MODEL_BOARD.search(stem) else stem
    return stem.replace("_", " ")


def model_files(m: int, n: int, model_dir: Path) -> list[Path]:
    """Saved solver 7 models trained on an ``m x n`` board."""
    if not model_dir.is_dir():
        return []
    out = []
    for path in sorted(model_dir.glob("*.pkl")):
        hit = _MODEL_BOARD.search(path.name)
        if hit and (int(hit.group(1)), int(hit.group(2))) == (m, n):
            out.append(path)
    return out


def home(kind: str, m: int, n: int, data_root: Path = DATA_DIR) -> bool:
    """Was ``kind`` built (or optimised) for ``m x n``?"""
    if kind == "exact":
        return (m, n) in EXACT_BOARDS
    if kind in TUNED_KINDS:
        return (tuned_dir(m, n, data_root) / f"{kind.removeprefix('search-')}_best_w.json").exists()
    if kind in LEARNED_KINDS:
        return bool(model_files(m, n, data_root / "models"))
    if kind in MCTS_KINDS and mcts_params_path(kind, m, n, data_root).exists():
        return True
    return (m, n) in TESTED_BOARDS


def offered(kind: str, m: int, n: int, data_root: Path = DATA_DIR) -> bool:
    """Home board, or measured to beat Greedy at least ``MIN_VS_GREEDY`` of the time here."""
    if home(kind, m, n, data_root):
        return True
    rate = vs_greedy(kind, m, n, data_root)
    return rate is not None and rate >= MIN_VS_GREEDY - 1e-9


def percent(points: float, games: float) -> int:
    """Win rate as a whole percentage, halves rounded up (the README tables round the same way)."""
    return math.floor(100 * points / games + 0.5)


def _greedy_note(kind: str, m: int, n: int, data_root: Path) -> str:
    rec = greedy_tests(data_root).get(kind, {}).get(f"{m}x{n}")
    return f" Beat Greedy in {percent(*rec)}% of {int(rec[1])} test games here." if rec else ""


def available_bots(m: int, n: int, data_root: Path = DATA_DIR, model_dir: Path | None = None) -> list[BotOption]:
    """The bots offered on ``m x n`` (see :func:`offered`), with their data present."""
    model_dir = model_dir if model_dir is not None else data_root / "models"
    out: list[BotOption] = []
    for kind, label, description in BOT_KINDS:
        if kind in LEARNED_KINDS:
            for path in model_files(m, n, model_dir):
                out.append(BotOption(f"{kind}:{path}", f"{label} ({_model_label(path)})",
                                     f"{description} Trained on {m}×{n}.{_greedy_note(kind, m, n, data_root)}"))
            continue
        if not offered(kind, m, n, data_root):
            continue
        if home(kind, m, n, data_root):
            if kind in TUNED_KINDS and tuned_depth(m, n) is not None:
                desc = f"{description} Tuned for {m}×{n}; looks {tuned_depth(m, n)} moves ahead, as when tuned."
            elif kind in TUNED_KINDS or (kind in MCTS_KINDS and (m, n) not in TESTED_BOARDS):
                desc = f"{description} Optimised for {m}×{n}."
            elif kind == "exact" and (m, n) == (3, 3):
                desc = f"{description} Its first move on 3×3 takes about 50 s while it solves the board."
            else:
                desc = description
        else:
            desc = f"{description} Not built for {m}×{n}."
        out.append(BotOption(kind, label, desc + _greedy_note(kind, m, n, data_root)))
    return out


def _mcts_params(kind: str, m: int, n: int, data_root: Path) -> dict[str, Any] | None:
    try:
        return json.loads(mcts_params_path(kind, m, n, data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tuned_depth(m: int, n: int) -> int | None:
    """Search depth for the tuned bots on ``m x n``: the tuning depth off 3x3, else ``None`` (time-limited)."""
    return None if (m, n) == (3, 3) else FITNESS_DEPTH


def make_bot(key: str, seed: int, board: tuple[int, int] | None = None, data_root: Path = DATA_DIR) -> Player:
    """Build the solver for a menu ``key``; with ``board``, use that board's optimised data if any."""
    if board is not None:
        m, n = board
        if key in TUNED_KINDS:
            return tuned_player(key, max_depth=tuned_depth(m, n), seed=seed, data_dir=tuned_dir(m, n, data_root))
        if key in MCTS_KINDS:
            params = _mcts_params(key, m, n, data_root)
            if params is not None:
                return MCTSPlayer(c=params.get("c", 0.5), eps=params.get("eps", 0.1),
                                  flat=key == "mcts-flat", seed=seed)
    return make_player(key, seed)


@dataclass(frozen=True)
class BotMove:
    """A finished move computation (``error`` set if the solver raised)."""

    move: int
    seconds: float
    nodes: int
    depth: int
    error: str | None = None


class BotWorker:
    """Runs ``player.choose_move`` on a daemon thread so the window stays responsive.

    :meth:`cancel` discards the pending result (the thread finishes on its own,
    within the time limit for time-bound solvers).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = 0
        self._busy = False
        self._result: BotMove | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, player: Player, state: GameState, time_limit: float | None) -> None:
        with self._lock:
            self._token += 1
            token = self._token
            self._busy = True
            self._result = None
        threading.Thread(target=self._run, args=(token, player, state, time_limit), daemon=True).start()

    def _run(self, token: int, player: Player, state: GameState, time_limit: float | None) -> None:
        t0 = time.perf_counter()
        try:
            move = player.choose_move(state, time_limit)
            result = BotMove(move, time.perf_counter() - t0, int(getattr(player, "nodes", 0)),
                             int(getattr(player, "depth_reached", 0)))
        except Exception as exc:  # shown in the UI instead of killing the thread silently
            result = BotMove(-1, time.perf_counter() - t0, 0, 0, f"{type(exc).__name__}: {exc}")
        with self._lock:
            if token == self._token:
                self._result = result
                self._busy = False

    def poll(self) -> BotMove | None:
        """The finished move, once (``None`` while thinking or after it was taken)."""
        with self._lock:
            result, self._result = self._result, None
            return result

    def cancel(self) -> None:
        with self._lock:
            self._token += 1
            self._busy = False
            self._result = None
