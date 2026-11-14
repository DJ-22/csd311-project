"""Board optimisation pipeline: every job kind end to end on 2x2 with tiny budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import experiments.optimize_boards as ob
from app.bots import available_bots, greedy_tests, home, mcts_params_path, tuned_dir


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr(ob, "ROOT", tmp_path)
    for name, value in {"TEST_GAMES": 2, "TUNE_GAMES": 2, "TIME_LIMIT": 0.05, "E3_BUDGET": 40,
                        "E3_BUDGET_GAMES": 2, "E3_SEEDS": 1, "N_POSITIONS": 150, "ROLLOUTS": 4,
                        "MODEL_HIDDEN": [8]}.items():
        monkeypatch.setattr(ob, name, value)
    monkeypatch.setattr(ob, "BIG_BOARDS", ((2, 2),))
    return argparse.Namespace(workers=2, tune_workers=2, label_workers=2)


def test_jobs_in_three_phases_quickest_first() -> None:
    jobs = ob.all_jobs({"done": {}, "tune_runs": {}})
    ids = [j.id for j in jobs]
    k = ids.index("prepare-e3-and-learned")
    first, last = jobs[:k], jobs[k + 1:]
    assert [j.minutes for j in first] == sorted(j.minutes for j in first)
    assert [j.minutes for j in last] == sorted(j.minutes for j in last)
    assert all(j.id.startswith(("test-", "exact-", "mcts")) for j in first)
    assert all(j.id.startswith(("learned-", "tuned-")) for j in last)
    assert len(set(ids)) == len(ids) == 20


def test_prepare_then_dependent_jobs(tiny: argparse.Namespace, tmp_path: Path) -> None:
    status = {"done": {}, "tune_runs": {}}
    ob.job_prepare(tiny, status)  # E3 on 2x2 and, meanwhile, the 2x2 dataset and model
    assert len(status["tune_runs"]["2x2"]) == 4 and ob._model_path(2, 2).exists()
    info = ob.job_tuned(2, 2, tiny, status)  # no new runs needed: picks, saves, tests
    assert set(info["chosen"]) == {"ga", "hc", "sa", "tabu"}
    assert {"search-ga", "search-tabu"} <= {b.key for b in available_bots(2, 2, tmp_path)}
    ob.job_learned(2, 2, tiny)
    assert {"search-learned", "learned", "search-ga"} <= set(greedy_tests(tmp_path))


def test_test_and_mcts_jobs(tiny: argparse.Namespace, tmp_path: Path) -> None:
    ob.job_test("rules", 2, 2, tiny)
    info = ob.job_mcts("mcts-flat", 2, 2, tiny)
    params = json.loads(mcts_params_path("mcts-flat", 2, 2, tmp_path).read_text(encoding="utf-8"))
    assert {params["c"], params["eps"]} == {info["params"]["c"], info["params"]["eps"]}
    tests = greedy_tests(tmp_path)
    assert tests["rules"]["2x2"][1] == 2 and tests["mcts-flat"]["2x2"][1] == 2
    assert home("mcts-flat", 2, 2, tmp_path)


def test_tuned_job_saves_weights_and_resumes(tiny: argparse.Namespace, tmp_path: Path) -> None:
    status = {"done": {}, "tune_runs": {}}
    info = ob.job_tuned(2, 2, tiny, status)
    assert set(info["chosen"]) == {"ga", "hc", "sa", "tabu"}
    assert all((tuned_dir(2, 2, tmp_path) / f"{o}_best_w.json").exists() for o in info["chosen"])
    assert len(status["tune_runs"]["2x2"]) == 4
    assert {"search-ga", "search-tabu"} <= {b.key for b in available_bots(2, 2, tmp_path)}
    again = ob.job_tuned(2, 2, tiny, status)  # every run recorded: no new optimiser runs
    assert again["chosen"] == info["chosen"]


def test_learned_job(tiny: argparse.Namespace, tmp_path: Path) -> None:
    info = ob.job_learned(2, 2, tiny)
    assert Path(info["model"]).exists() and Path(info["model"]).parent == tmp_path / "models"
    kinds = {b.key.split(":")[0] for b in available_bots(2, 2, tmp_path)}
    assert {"search-learned", "learned"} <= kinds
    assert set(greedy_tests(tmp_path)) == {"search-learned", "learned"}
