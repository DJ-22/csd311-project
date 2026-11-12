"""Timestamped CSV / figure paths for experiment scripts (never overwrite old results)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def stamped_path(name: str, suffix: str, results_dir: Path = RESULTS_DIR) -> Path:
    """``<results_dir>/<name>_<timestamp><suffix>`` that does not exist yet."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_dir / f"{name}_{stamp}{suffix}"
    i = 1
    while path.exists():
        path = results_dir / f"{name}_{stamp}_{i}{suffix}"
        i += 1
    return path


def write_rows(rows: Sequence[dict[str, Any]], name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write dict rows (columns from the first row) to a new timestamped CSV."""
    path = stamped_path(name, ".csv", results_dir)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return path
