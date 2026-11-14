"""App logic without pygame: bot menu, background moves, match bookkeeping."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.bots import (
    MIN_VS_GREEDY,
    BotWorker,
    available_bots,
    home,
    make_bot,
    model_files,
    offered,
    percent,
)
from app.match import Match, Seat
from engine.game import legal_moves


def test_every_bot_offered_on_its_home_boards(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "e4_mlp-64x64_both_reg_3x3_s0.pkl").write_bytes(b"")
    (models / "e5_mlp-64x64_both_reg_2x3_s0.pkl").write_bytes(b"")

    def kinds(m: int, n: int) -> set[str]:
        return {b.key.split(":")[0] for b in available_bots(m, n, tmp_path)}

    untrained = {"search", "mcts", "mcts-flat", "rules", "greedy", "random", "exact"}
    assert untrained <= kinds(2, 2)
    assert untrained | {"search-learned", "learned"} <= kinds(2, 3)
    assert untrained | {"search-learned", "learned"} <= kinds(3, 3)
    for board in ((3, 4), (4, 4), (5, 5)):  # nothing optimised or tested yet in tmp_path
        assert kinds(*board) == set()


def test_off_board_rule_and_optimised_bots(tmp_path: Path) -> None:
    opt = tmp_path / "board_opt"
    opt.mkdir()
    (opt / "greedy_tests.json").write_text(json.dumps(
        {"search": {"5x5": [39, 40]}, "rules": {"5x5": [31, 40]}, "mcts": {"5x5": [4, 40]}}), encoding="utf-8")
    (opt / "mcts_5x5.json").write_text(json.dumps({"c": 1.0, "eps": 0.1}), encoding="utf-8")
    assert offered("search", 5, 5, tmp_path)  # 0.975 >= 0.8
    assert not offered("rules", 5, 5, tmp_path)  # 0.775 < 0.8
    assert offered("mcts", 5, 5, tmp_path)  # optimised for 5x5: home, whatever its test score
    assert not offered("random", 5, 5, tmp_path) and not offered("exact", 5, 5, tmp_path)
    bot = make_bot("mcts", 3, (5, 5), tmp_path)
    assert bot.c == 1.0 and not bot.flat  # type: ignore[attr-defined]
    weights = tmp_path / "tuned" / "5x5"
    weights.mkdir(parents=True)
    (weights / "ga_best_w.json").write_text(json.dumps({"weights": [0.0] * 12}), encoding="utf-8")
    assert home("search-ga", 5, 5, tmp_path) and not home("search-tabu", 5, 5, tmp_path)
    assert make_bot("search-ga", 0, (5, 5), tmp_path).max_depth == 2  # type: ignore[attr-defined]
    (tmp_path / "tuned" / "ga_best_w.json").write_text(json.dumps({"weights": [0.0] * 12}), encoding="utf-8")
    assert make_bot("search-ga", 0, (3, 3), tmp_path).max_depth is None  # type: ignore[attr-defined]
    assert not home("learned", 5, 5, tmp_path) and not home("learned", 3, 3, tmp_path)  # no model yet
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mlp-64x64_both_reg_5x5_s0.pkl").write_bytes(b"")
    assert home("learned", 5, 5, tmp_path) and offered("search-learned", 5, 5, tmp_path)  # trained here
    assert MIN_VS_GREEDY == 0.8


def test_percent_rounds_halves_up() -> None:
    assert [percent(29, 40), percent(33, 40), percent(9, 40), percent(39, 40)] == [73, 83, 23, 98]


def test_model_files_match_board(tmp_path: Path) -> None:
    for name in ("e4_mlp-64x64_both_reg_3x3_s0.pkl", "e5_mlp-64x64_both_reg_2x3_s0.pkl", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in model_files(3, 3, tmp_path)] == ["e4_mlp-64x64_both_reg_3x3_s0.pkl"]
    assert model_files(4, 4, tmp_path) == []


def _wait(worker: BotWorker, timeout: float = 10.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        r = worker.poll()
        if r is not None:
            return r
        time.sleep(0.01)
    raise AssertionError("bot did not answer")


def test_worker_returns_legal_move_and_cancel_discards() -> None:
    match = Match(3, 3, (Seat("You"), Seat("Greedy", make_bot("greedy", 1))))
    worker = BotWorker()
    worker.start(match.seats[1].player, match.state, 0.1)  # type: ignore[arg-type]
    r = _wait(worker)
    assert r.error is None and r.move in legal_moves(match.state)
    assert worker.poll() is None and not worker.busy
    worker.start(make_bot("search", 0), match.state, 0.2)
    worker.cancel()
    time.sleep(0.4)
    assert worker.poll() is None


def test_worker_reports_errors() -> None:
    match = Match(2, 2, (Seat("a"), Seat("b")))
    for e in legal_moves(match.state):
        match.play(e)
    worker = BotWorker()
    worker.start(make_bot("greedy", 0), match.state, None)  # full board: choose_move raises
    assert _wait(worker).error


def test_match_tracks_owners_and_extra_turns() -> None:
    m = Match(1, 1, (Seat("A"), Seat("B")))
    assert m.play(0) == 0 and m.to_move == 1
    assert m.play(1) == 0 and m.to_move == 0
    assert m.play(2) == 0 and m.to_move == 1
    assert m.play(3) == 1 and m.over  # seat 1 draws the 4th side
    assert m.owners == [1] and m.winner == 1
    assert m.edge_owner == {0: 0, 1: 1, 2: 0, 3: 1}
    assert [r.boxes for r in m.history] == [0, 0, 0, 1]


def test_full_bot_game_runs_to_the_end() -> None:
    m = Match(2, 2, (Seat("Greedy", make_bot("greedy", 3)), Seat("Exact", make_bot("exact", 4))))
    while not m.over:
        m.play(m.current.player.choose_move(m.state, 0.1))  # type: ignore[union-attr]
    assert sum(m.state.scores) == 4 and all(o in (0, 1) for o in m.owners)
