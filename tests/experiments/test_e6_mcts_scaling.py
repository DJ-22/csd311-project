"""E6 script: fixed-iteration MCTS variants against an opponent, CSV and plot output."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.e6_mcts_scaling import VARIANTS, label, main, make_mcts, run
from harness.metrics import summarise


def test_make_mcts_variants() -> None:
    uct = make_mcts("uct", 50, seed=1)
    rnd = make_mcts("uct-random", 50, seed=1)
    flat = make_mcts("flat", 50, seed=1)
    assert uct.name == "uct@50" and uct.iterations == 50 and not uct.flat and uct.rollout == "greedy"
    assert rnd.rollout == "random" and rnd.eps == 1.0
    assert flat.flat and flat.rollout == "greedy"


def test_run_rows_and_common_seeds() -> None:
    rows = run("2x2", ["uct", "flat"], [10, 30], "greedy", games=4, seed=7, time_limit=None)
    assert len(rows) == 2 * 2 * 4
    for v in ("uct", "flat"):
        for it in (10, 30):
            s = summarise(rows, label(v, it))
            assert s.games == 4
    # every (variant, budget) plays the same game seeds
    seeds = {}
    for r in rows:
        name = r["p0"] if "@" in r["p0"] else r["p1"]
        seeds.setdefault(name, []).append(r["seed"])
    assert len({tuple(s) for s in seeds.values()}) == 1


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError):
        run("2x2", ["nope"], [10], "greedy", games=2, seed=0, time_limit=None)


def test_main_writes_csv_and_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    main(["--board", "2x2", "--iterations", "10,20", "--variants", ",".join(VARIANTS), "--games", "2",
          "--results-dir", str(tmp_path)])
    assert len(list(tmp_path.glob("e6_2x2_*.csv"))) == 1
    assert len(list(tmp_path.glob("e6_2x2_*.png"))) == 1
