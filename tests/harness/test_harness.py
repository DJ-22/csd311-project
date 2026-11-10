import csv

import pytest

from harness.cli import main, make_player
from harness.metrics import effective_branching_factor, summarise, win_rate_ci
from harness.tournament import CSV_COLUMNS, match, play_game, round_robin, write_csv
from engine.game import GameState, legal_moves
from solvers.base import Player
from solvers.baselines import GreedyPlayer, RandomPlayer


def test_csv_columns_match_spec() -> None:
    assert CSV_COLUMNS == (
        "p0", "p1", "board", "seed", "score0", "score1", "winner",
        "moves", "time_p0", "time_p1", "nodes_p0", "nodes_p1",
        "depth_p0", "depth_p1", "bstar_p0", "bstar_p1",
    )


class FakeSearcher(Player):
    """Plays the lowest legal edge and reports fixed search statistics."""

    name = "fake"

    def __init__(self) -> None:
        self.nodes = 0
        self.depth_reached = 0

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        self.nodes, self.depth_reached = 14, 3
        return legal_moves(state)[0]


def test_depth_and_bstar_columns() -> None:
    row = play_game(FakeSearcher(), RandomPlayer(0), 2, 2, seed=0)
    assert row["nodes_p0"] == 14 and row["depth_p0"] == 3
    assert row["bstar_p0"] == pytest.approx(2.0, abs=1e-6)
    assert row["depth_p1"] == 0 and row["bstar_p1"] == 0.0  # random exposes no depth
    s = summarise([row], "fake")
    assert s.mean_depth_per_move == 3 and s.mean_bstar == pytest.approx(2.0, abs=1e-6)
    assert summarise([row], "random").mean_bstar == 0.0


@pytest.mark.parametrize("nodes, depth, expected", [(14, 3, 2.0), (3, 1, 3.0), (2, 2, 1.0), (39, 3, 3.0)])
def test_effective_branching_factor(nodes: int, depth: int, expected: float) -> None:
    assert effective_branching_factor(nodes, depth) == pytest.approx(expected, abs=1e-6)


def test_effective_branching_factor_below_one_and_invalid() -> None:
    b = effective_branching_factor(1, 3)
    assert b + b**2 + b**3 == pytest.approx(1, abs=1e-6)
    assert effective_branching_factor(0, 3) == 0.0
    with pytest.raises(ValueError):
        effective_branching_factor(10, 0)


def test_win_rate_ci() -> None:
    rate, lo, hi = win_rate_ci(50, 100)
    assert rate == 0.5
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    rate, lo, hi = win_rate_ci(10, 10)
    assert rate == 1.0 and hi == pytest.approx(1.0) and 0.65 < lo < 0.75    # Wilson stays informative at 100%
    rate, _, _ = win_rate_ci(1.5, 3)    # one win, one draw, one loss
    assert rate == 0.5


def test_play_game_row() -> None:
    row = play_game(GreedyPlayer(1), RandomPlayer(2), 2, 3, seed=7)
    assert set(row) == set(CSV_COLUMNS)
    assert row["board"] == "2x3" and row["seed"] == 7
    assert row["score0"] + row["score1"] == 6
    assert row["moves"] == 17
    assert row["winner"] in (0, 1, -1)
    assert row["p0"] == "greedy" and row["p1"] == "random"


def test_match_alternates_first_player_and_is_reproducible() -> None:
    rows = match("greedy", "random", make_player, 3, 3, games=6, seed=100)
    assert [r["p0"] for r in rows] == ["greedy", "random"] * 3
    assert [r["seed"] for r in rows] == list(range(100, 106))
    again = match("greedy", "random", make_player, 3, 3, games=6, seed=100)
    strip = lambda rs: [(r["p0"], r["score0"], r["score1"]) for r in rs]
    assert strip(rows) == strip(again)


def test_round_robin_and_summary() -> None:
    rows = round_robin(["random", "greedy", "exact"], make_player, 2, 2, games=4, seed=0)
    assert len(rows) == 3 * 4
    assert len({r["seed"] for r in rows}) == 12
    s = summarise(rows, "exact")
    assert s.games == 8
    assert s.wins + s.draws + s.losses == 8
    assert s.ci_low <= s.win_rate <= s.ci_high
    assert s.mean_nodes_per_move >= 0
    margin_check = sum(
        (r["score0"] - r["score1"]) if r["p0"] == "exact" else (r["score1"] - r["score0"])
        for r in rows if "exact" in (r["p0"], r["p1"])
    ) / 8
    assert s.mean_margin == pytest.approx(margin_check)


def test_write_csv_has_exact_columns_and_never_overwrites(tmp_path) -> None:
    rows = match("greedy", "random", make_player, 2, 2, games=2, seed=0)
    p1 = write_csv(rows, "t", tmp_path)
    p2 = write_csv(rows, "t", tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()
    with p1.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert len(list(reader)) == 2


def test_summarise_reads_csv_strings(tmp_path) -> None:
    rows = match("greedy", "random", make_player, 2, 2, games=4, seed=0)
    path = write_csv(rows, "t", tmp_path)
    with path.open(newline="", encoding="utf-8") as fh:
        loaded = list(csv.DictReader(fh))
    assert summarise(loaded, "greedy") == summarise(rows, "greedy")


def test_cli_tournament_writes_csv(tmp_path, monkeypatch, capsys) -> None:
    import harness.tournament as tournament

    monkeypatch.setattr(tournament, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tournament.write_csv, "__defaults__", (tmp_path,))
    main(["tournament", "--board", "2x2", "--games", "4", "--players", "random,greedy,exact", "--name", "cli"])
    out = capsys.readouterr().out
    assert "seed=0" in out and "board=2x2" in out
    assert len(list(tmp_path.glob("cli_*.csv"))) == 1


def test_unknown_player() -> None:
    with pytest.raises(ValueError):
        make_player("nope", 0)
