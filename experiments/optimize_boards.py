"""Optimise the solvers for boards they were not built for, and test each against Greedy.

Each bot is optimised for each big board (where it has anything to optimise) and then
plays ``TEST_GAMES`` games against Greedy at ``TIME_LIMIT`` s per move, seats alternating.
The work is scheduled in three phases to keep every core busy:

1. Timed jobs that need nothing else, quickest first, on ``--workers`` processes (keep at
   or below the performance cores: a time-limited player searches less on a slow core).
2. ``prepare-e3-and-learned``: all untimed work at once. Every E3 optimiser run for every
   big board in one pool of ``--tune-workers`` processes (all cores; fixed-depth fitness
   games give the same result on any core), longest runs first; meanwhile, in a thread,
   the learned datasets (labelling on ``--label-workers`` processes) and models.
3. The timed tests of what phase 2 produced, quickest first.

Jobs:

* ``test-<bot>-<board>``: search and rules have nothing to optimise; only the Greedy test.
* ``exact-3x3``: the exact solver solves 3x3 (each worker solves it once, ~50 s), then the test.
* ``mcts-<board>`` / ``mcts-flat-<board>``: pick the exploration constant ``c`` (UCT) or the
  rollout ``eps`` (flat) with the best score against Greedy over ``TUNE_GAMES`` games (all
  settings in one batch), save it to ``data/board_opt/<bot>_<board>.json``, then the test
  (on fresh seeds).
* ``learned-<board>``: test search-learned and the one-ply learned player with the model
  phase 2 trained (50,000-position Monte Carlo dataset, MLP 64x64, both inputs, regression,
  in ``data/models/``).
* ``tuned-<board>``: from phase 2's E3 runs (4 optimisers x 3 seeds x 600 fitness
  evaluations) keep the best weights per optimiser in ``data/tuned/<board>/``, then test the
  four tuned players, built as the app builds them (search depth 2, their tuning depth).

Everything is resumable: finished jobs (and finished E3 runs inside a job) are recorded
in ``data/board_opt/status.json`` and skipped on the next start. Greedy test results go to
``data/board_opt/greedy_tests.json`` (read by the app) and a summary CSV to
``experiments/results/board_opt_summary_<timestamp>.csv``.

Usage::

    python -m experiments.optimize_boards                 # run every pending job
    python -m experiments.optimize_boards --list          # jobs, estimates, status
    python -m experiments.optimize_boards --only learned  # jobs whose id contains "learned"
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from app.bots import (
    DATA_DIR,
    TUNED_KINDS,
    board_opt_dir,
    greedy_tests,
    make_bot,
    mcts_params_path,
    offered,
    tuned_depth,
    tuned_dir,
)
from experiments._results import write_rows
from harness.metrics import summarise
from harness.tournament import _match_jobs, _run_jobs, match, write_csv
from solvers.base import Player
from solvers.baselines import GreedyPlayer
from solvers.mcts import MCTSPlayer

TEST_GAMES = 40
TEST_SEED = 0
TIME_LIMIT = 1.0
TUNE_GAMES = 30
TUNE_SEED = 100_000  # clear of the test seeds
UCT_GRID = ({"c": 0.5, "eps": 0.1}, {"c": 0.25, "eps": 0.1}, {"c": 1.0, "eps": 0.1}, {"c": 1.41, "eps": 0.1})
FLAT_GRID = ({"c": 0.5, "eps": 0.1}, {"c": 0.5, "eps": 0.0}, {"c": 0.5, "eps": 0.3})  # defaults first: ties keep them
N_POSITIONS, ROLLOUTS, MODEL_HIDDEN = 50_000, 200, [64, 64]
E3_BUDGET, E3_BUDGET_GAMES, E3_SEEDS = 600, 40, 3
BIG_BOARDS: tuple[tuple[int, int], ...] = ((3, 4), (4, 4), (5, 5))
ROOT: Path = DATA_DIR  # all outputs live under this data folder (tests point it elsewhere)


@dataclass(frozen=True)
class Job:
    id: str
    minutes: float  # estimate on the reference machine, used for the order
    run: Callable[[argparse.Namespace], dict[str, Any]]


# ----- persistence --------------------------------------------------------------------
def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_status() -> dict[str, Any]:
    try:
        return json.loads((board_opt_dir(ROOT) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"done": {}, "tune_runs": {}}


def save_status(status: dict[str, Any]) -> None:
    _write_json(board_opt_dir(ROOT) / "status.json", status)


def record_test(kind: str, m: int, n: int, points: float, games: int) -> None:
    tests = greedy_tests(ROOT)
    tests.setdefault(kind, {})[f"{m}x{n}"] = [points, games]
    _write_json(board_opt_dir(ROOT) / "greedy_tests.json", tests)


# ----- players for worker processes (module level, so they pickle) ---------------------
def _factory(m: int, n: int, root: Path, name: str, seed: int) -> Player:
    if name == "greedy":
        return GreedyPlayer(seed)
    if name.startswith("cfg|"):  # cfg|<kind>|c=..|eps=..: an MCTS configuration being tried
        _, kind, c, eps = name.split("|")
        p = MCTSPlayer(c=float(c[2:]), eps=float(eps[4:]), flat=kind == "mcts-flat", seed=seed)
        p.name = name
        return p
    return make_bot(name, seed, (m, n), root)


def _score(rows: list[dict[str, Any]]) -> tuple[str, float, int]:
    """(name of the non-Greedy seat, its points, games)."""
    name = rows[0]["p0"] if rows[0]["p0"] != "greedy" else rows[0]["p1"]
    s = summarise(rows, name)
    return name, s.wins + 0.5 * s.draws, s.games


def greedy_test(key: str, kind: str, m: int, n: int, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    rows = match(key, "greedy", partial(_factory, m, n, ROOT), m, n, TEST_GAMES, TEST_SEED, TIME_LIMIT,
                 workers=args.workers)
    write_csv(rows, f"{kind}_{m}x{n}", board_opt_dir(ROOT) / "tests")
    _, pts, games = _score(rows)
    record_test(kind, m, n, pts, games)
    print(f"    {kind} on {m}x{n} vs Greedy: {pts:g}/{games} = {pts / games:.3f}  "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return {"kind": kind, "points": pts, "games": games}


# ----- jobs ---------------------------------------------------------------------------
def job_test(kind: str, m: int, n: int, args: argparse.Namespace) -> dict[str, Any]:
    return {"tests": [greedy_test(kind, kind, m, n, args)]}


def job_mcts(kind: str, m: int, n: int, args: argparse.Namespace) -> dict[str, Any]:
    """Try every setting in one batch of games (no idle workers between settings), keep the best."""
    grid = UCT_GRID if kind == "mcts" else FLAT_GRID
    names = [f"cfg|{kind}|c={cfg['c']}|eps={cfg['eps']}" for cfg in grid]
    jobs = [job for name in names for job in _match_jobs(name, "greedy", TUNE_GAMES, TUNE_SEED)]
    rows = _run_jobs(jobs, partial(_factory, m, n, ROOT), m, n, TIME_LIMIT, args.workers, False)
    scores: dict[str, float] = {}
    for cfg, name in zip(grid, names):
        _, pts, games = _score([r for r in rows if name in (r["p0"], r["p1"])])
        scores[f"c={cfg['c']},eps={cfg['eps']}"] = pts / games
        print(f"    {kind} c={cfg['c']} eps={cfg['eps']}: {pts:g}/{games}", flush=True)
    best = max(range(len(grid)), key=lambda i: list(scores.values())[i])  # first of the ties
    params = {**grid[best], "board": f"{m}x{n}", "selection_games": TUNE_GAMES, "selection_scores": scores}
    _write_json(mcts_params_path(kind, m, n, ROOT), params)
    print(f"    chose {grid[best]}", flush=True)
    return {"params": grid[best], "tests": [greedy_test(kind, kind, m, n, args)]}


def _model_path(m: int, n: int) -> Path:
    return ROOT / "models" / f"mlp-64x64_both_reg_{m}x{n}_s0.pkl"


def build_learned(m: int, n: int, workers: int) -> Path:
    """Dataset (Monte Carlo labels, untimed: any number of workers) and MLP for ``m x n``; skips what exists."""
    from engine.board import get_board
    from solvers.supervised.dataset import build_dataset, default_path
    from solvers.supervised.models import train

    data = ROOT / default_path(get_board(m, n), "mc", N_POSITIONS, 0).name
    if not data.exists():
        t0 = time.perf_counter()
        build_dataset(f"{m}x{n}", "mc", N_POSITIONS, 0, ROLLOUTS, workers, data)
        print(f"    dataset {data.name} built ({time.perf_counter() - t0:.0f} s)", flush=True)
    model = _model_path(m, n)
    if not model.exists():
        t0 = time.perf_counter()
        train("mlp", data, "both", "reg", 0, model, hidden=MODEL_HIDDEN)
        print(f"    model {model.name} trained ({time.perf_counter() - t0:.0f} s)", flush=True)
    return model


def job_learned(m: int, n: int, args: argparse.Namespace) -> dict[str, Any]:
    model = build_learned(m, n, args.tune_workers)  # normally already built by the prepare job
    return {"model": str(model), "tests": [greedy_test(f"search-learned:{model}", "search-learned", m, n, args),
                                           greedy_test(f"learned:{model}", "learned", m, n, args)]}


def _tune_run(opt: str, seed: int, m: int, n: int, budget: int, budget_games: int) -> tuple[str, int, float, list[float]]:
    from experiments.e3_tune import CORES
    from solvers.tuned.fitness import FitnessEvaluator

    evaluator = FitnessEvaluator((m, n), budget_games, seed)
    w, trace = CORES[opt](evaluator, budget, seed)
    return opt, seed, float(trace[-1][2]), [float(x) for x in w]


def run_e3(boards: Sequence[tuple[int, int]], args: argparse.Namespace, status: dict[str, Any],
           during: Callable[[], None] | None = None) -> None:
    """Every missing E3 run for ``boards`` in one process pool, longest first (biggest board, tabu last:
    its cache hits make it the shortest). ``during`` runs in a thread meanwhile. Each finished run
    is saved at once, so an interruption loses only the runs in progress."""
    from solvers.tuned import OPTIMISERS

    runs = status.setdefault("tune_runs", {})
    order = sorted(OPTIMISERS, key=lambda o: o == "tabu")
    todo = [(m, n, o, s) for m, n in sorted(boards, key=lambda b: -b[0] * b[1]) for o in order
            for s in range(E3_SEEDS) if f"{o}_{s}" not in runs.get(f"{m}x{n}", {})]
    worker: threading.Thread | None = None
    errors: list[BaseException] = []
    if during is not None:
        def guarded() -> None:
            try:
                during()  # type: ignore[misc]
            except BaseException as exc:  # reported after the pool finishes
                errors.append(exc)
        worker = threading.Thread(target=guarded, daemon=True)
        worker.start()
    if todo:
        print(f"    {len(todo)} E3 runs on {min(args.tune_workers, len(todo))} processes", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.tune_workers, len(todo))) as pool:
            futures = {pool.submit(_tune_run, o, s, m, n, E3_BUDGET, E3_BUDGET_GAMES): (m, n) for m, n, o, s in todo}
            for fut in as_completed(futures):
                m, n = futures[fut]
                opt, seed, best_f, w = fut.result()
                runs.setdefault(f"{m}x{n}", {})[f"{opt}_{seed}"] = {"best_f": best_f, "w": w}
                save_status(status)
                left = sum(not f.done() for f in futures)
                print(f"    [{datetime.now():%H:%M}] E3 {m}x{n} {opt} seed {seed}: best fitness {best_f:.4f} "
                      f"({left} runs left)", flush=True)
    if worker is not None:
        worker.join()
    if errors:
        raise errors[0]


def job_prepare(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    """All untimed work at once on every core: E3 on the big boards, and meanwhile the learned datasets and models."""
    def learned_all() -> None:
        for m, n in BIG_BOARDS:
            build_learned(m, n, args.label_workers)

    run_e3(BIG_BOARDS, args, status, during=learned_all)
    return {"boards": [f"{m}x{n}" for m, n in BIG_BOARDS]}


def job_tuned(m: int, n: int, args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    from solvers.tuned import OPTIMISERS, save_best_w

    run_e3([(m, n)], args, status)  # normally already done by the prepare job
    board = f"{m}x{n}"
    runs = status["tune_runs"][board]
    chosen = {}
    for opt in OPTIMISERS:
        seed, rec = max(((s, runs[f"{opt}_{s}"]) for s in range(E3_SEEDS)), key=lambda t: t[1]["best_f"])
        save_best_w(opt, rec["w"], data_dir=tuned_dir(m, n, ROOT), board=board, seed=seed, budget=E3_BUDGET,
                    fitness=rec["best_f"])
        chosen[opt] = {"seed": seed, "fitness": rec["best_f"]}
    return {"chosen": chosen, "tests": [greedy_test(k, k, m, n, args) for k in TUNED_KINDS]}


def all_jobs(status: dict[str, Any]) -> list[Job]:
    """Three phases: timed jobs that need nothing (quickest first), then all untimed work on every
    core, then the timed tests of what that produced (quickest first)."""
    est = {"rules": (0.1, 0.1, 0.1), "search": (0.4, 0.6, 0.5), "mcts": (5, 6, 8), "mcts-flat": (5, 6, 8),
           "learned": (2, 3, 4), "tuned": (2, 3, 3)}
    first = [Job("exact-3x3", 2, partial(job_test, "exact", 3, 3))]
    last: list[Job] = []
    for i, (m, n) in enumerate(BIG_BOARDS):
        b = f"{m}x{n}"
        first += [Job(f"test-rules-{b}", est["rules"][i], partial(job_test, "rules", m, n)),
                  Job(f"test-search-{b}", est["search"][i], partial(job_test, "search", m, n)),
                  Job(f"mcts-{b}", est["mcts"][i], partial(job_mcts, "mcts", m, n)),
                  Job(f"mcts-flat-{b}", est["mcts-flat"][i], partial(job_mcts, "mcts-flat", m, n))]
        last += [Job(f"learned-{b}", est["learned"][i], partial(job_learned, m, n)),
                 Job(f"tuned-{b}", est["tuned"][i], lambda a, m=m, n=n: job_tuned(m, n, a, status))]
    prepare = Job("prepare-e3-and-learned", 265, lambda a: job_prepare(a, status))  # measured: 4.4 h
    return sorted(first, key=lambda j: j.minutes) + [prepare] + sorted(last, key=lambda j: j.minutes)


def write_summary() -> Path | None:
    rows = []
    for kind, boards in sorted(greedy_tests(ROOT).items()):
        for board, (pts, games) in sorted(boards.items()):
            m, n = map(int, board.split("x"))
            depth = tuned_depth(m, n) if kind in TUNED_KINDS else None
            rows.append({"bot": kind, "board": board, "seed": TEST_SEED, "time_limit_s": TIME_LIMIT,
                         "max_depth": depth if depth is not None else "", "points": pts, "games": games,
                         "win_rate_vs_greedy": round(pts / games, 4), "offered_in_app": offered(kind, m, n, ROOT)})
    return write_rows(rows, "board_opt_summary") if rows else None


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8, help="processes for timed games (<= performance cores)")
    ap.add_argument("--tune-workers", type=int, default=20, help="processes for untimed work: E3 runs (all cores)")
    ap.add_argument("--label-workers", type=int, default=8,
                    help="processes for dataset labelling while E3 runs (untimed, may share cores)")
    ap.add_argument("--only", default="", help="run only jobs whose id contains this text")
    ap.add_argument("--list", action="store_true", help="list jobs and exit")
    args = ap.parse_args(argv)

    status = load_status()
    jobs = [j for j in all_jobs(status) if args.only in j.id]
    pending = [j for j in jobs if j.id not in status["done"]]
    total = sum(j.minutes for j in pending)
    print(f"{len(pending)} of {len(jobs)} jobs pending, estimated {total / 60:.1f} h "
          f"(workers={args.workers}, tune workers={args.tune_workers})")
    for j in jobs:
        state = "done" if j.id in status["done"] else "pending"
        print(f"  {j.id:<24} ~{j.minutes:>5.1f} min  {state}")
    if args.list:
        return
    left = total
    for j in pending:
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M}] {j.id} (~{j.minutes:g} min; ~{left / 60:.1f} h left)", flush=True)
        info = j.run(args)
        info["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        info["finished"] = datetime.now().isoformat(timespec="seconds")
        status["done"][j.id] = info
        save_status(status)
        left -= j.minutes
        print(f"  done in {info['minutes']} min", flush=True)
    if all(j.id in status["done"] for j in all_jobs(status)):
        print(f"\nall jobs done; summary: {write_summary()}")


if __name__ == "__main__":
    main()
