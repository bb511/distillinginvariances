#!/usr/bin/env python3
"""Average the results table over the folds of each run.

One output row per distinct run and probe (everything the contract identifies a row by
except the fold), carrying the mean and the population standard deviation of every
metric and the number of folds behind them. The spread is the one across folds, not an
error on the mean: the folds share the pool, so they are not independent samples. The
main run's rows are no fold and are left out.

python3 scripts/analysis/summarise.py python3 scripts/analysis/summarise.py --results
results/results.csv --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.vocabulary import MAIN_FOLD, RESULT_COLUMNS  # noqa: E402

_FIRST_METRIC = RESULT_COLUMNS.index("accuracy")
# epochs_run is a measured quantity, not a knob: early stopping ends each fold at its own
# epoch, so it is averaged with the metrics rather than grouped on.
GROUP_COLUMNS = tuple(
    column
    for column in RESULT_COLUMNS[:_FIRST_METRIC]
    if column not in ("fold", "epochs_run")
)
VALUE_COLUMNS = ("epochs_run", *RESULT_COLUMNS[_FIRST_METRIC:])


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["fold"] != MAIN_FOLD]


def _grouped(rows: list[dict[str, str]]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[column] for column in GROUP_COLUMNS), []).append(
            row
        )
    return groups


def _values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    """The column across the folds of one run, unreadable entries read as NaN."""
    return np.array([float(row.get(column) or "nan") for row in rows], dtype=float)


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _summary(key: tuple[str, ...], rows: list[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, object] = dict(zip(GROUP_COLUMNS, key))
    summary["n_folds_seen"] = len(rows)
    for column in VALUE_COLUMNS:
        mean, std = _mean_std(_values(rows, column))
        summary[f"{column}_mean"], summary[f"{column}_std"] = mean, std
    return summary


def _header() -> list[str]:
    tail = [f"{column}_{stat}" for column in VALUE_COLUMNS for stat in ("mean", "std")]
    return [*GROUP_COLUMNS, "n_folds_seen", *tail]


def _write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_header())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    groups = _grouped(_read(args.results))
    if not groups:
        raise SystemExit(f"{args.results} holds no rows.")
    _write([_summary(key, groups[key]) for key in sorted(groups)], args.out)
    print(f"{len(groups)} run(s) summarised over their folds -> {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "results.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
