"""E2 script: exact-solver methods agree, search ablation rows are well formed."""

from __future__ import annotations

import csv
from pathlib import Path

from engine.board import get_board
from experiments.e2_search_efficiency import (
    SEARCH_CONFIGS,
    exact_rows,
    main,
    sample_positions,
    search_rows,
)
from solvers.exact import get_solver


def test_exact_methods_agree_and_prune() -> None:
    rows = exact_rows(["1x2", "2x2"], brute_max_edges=7)
    by = {(r["board"], r["method"]): r for r in rows}
    assert ("1x2", "brute-force") in by and ("2x2", "brute-force") not in by
    for b in ("1x2", "2x2"):
        values = {r["value"] for r in rows if r["board"] == b}
        assert values == {get_solver(*map(int, b.split("x"))).value(0)}
        assert by[(b, "memo+sym")]["nodes"] <= by[(b, "memo")]["nodes"]
        assert by[(b, "alphabeta")]["nodes"] <= by[(b, "memo")]["nodes"]
    assert by[("1x2", "memo")]["nodes"] < by[("1x2", "brute-force")]["nodes"]


def test_sample_positions_respect_min_free() -> None:
    board = get_board(3, 3)
    pos = sample_positions(3, 3, 10, min_free=8, seed=0)
    assert len(pos) == 10
    assert all(board.E - s.edges.bit_count() >= 8 for s in pos)
    again = sample_positions(3, 3, 10, min_free=8, seed=0)
    assert [s.edges for s in pos] == [s.edges for s in again]


def test_search_rows_cover_every_config() -> None:
    rows = search_rows(["2x2"], positions=3, depths=[2], time_limit=0.05, min_free=6, seed=0)
    assert len(rows) == 3 * len(SEARCH_CONFIGS) * 2  # depth mode + time mode
    for r in rows:
        assert r["nodes"] > 0 and r["depth_reached"] >= 1
        if r["mode"] == "depth":
            assert r["depth_reached"] == 2
    full = sum(r["nodes"] for r in rows if r["mode"] == "depth" and r["config"] == "full")
    no_ordering = sum(r["nodes"] for r in rows if r["mode"] == "depth" and r["config"] == "no-ordering")
    assert full <= no_ordering


def test_main_writes_both_csvs(tmp_path: Path) -> None:
    main(["--exact-boards", "1x2", "--search-boards", "2x2", "--positions", "2", "--depths", "2",
          "--time-limit", "0", "--min-free", "6", "--results-dir", str(tmp_path)])
    exact = list(tmp_path.glob("e2_exact_*.csv"))
    search = list(tmp_path.glob("e2_search_*.csv"))
    assert len(exact) == 1 and len(search) == 1
    with search[0].open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2 * len(SEARCH_CONFIGS)
