#!/usr/bin/env python3
"""Per-fold differences between one method and a reference, with a bootstrap interval.

Each fold's run is compared with the reference run of the same cell and the same fold, so
the fold-to-fold spread of the pool cancels. The interval is a percentile bootstrap over
the per-fold differences, a description of five paired numbers rather than a test: the
folds share the pool and are not independent.

    python3 scripts/analysis/paired_differences.py --method kd
    python3 scripts/analysis/paired_differences.py --method relational --baseline kd \\
        --split test --metrics accuracy nll fid_top1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.bootstrap import bootstrap_ci  # noqa: E402

# The cell a pairing happens inside, plus the two knobs a method shares with the method it is
# compared against (kd against hint at one temperature, say).
JOIN = ("dataset", "teacher", "regime", "fold", "temperature", "alpha")
# What the pairs are reported per, the fold having been averaged out of them.
CELL = ("dataset", "teacher", "regime", "temperature", "alpha")
METRICS = ("accuracy", "nll", "ece", "fid_top1", "inv_1mjsd")
OUT_COLUMNS = (
    *CELL,
    "method",
    "reference",
    "eval_split",
    "metric",
    "n_folds",
    "mean_delta",
    "ci_low",
    "ci_high",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    # Both keys: a baseline carries no temperature and no alpha, so the full key of a
    # distilled run never matches it and the short key is what finds it.
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        index[tuple(row[column] for column in JOIN)] = row
        index.setdefault(tuple(row[column] for column in JOIN[:4]), row)
    return index


def _pairs(
    rows: list[dict[str, str]], reference: dict[tuple[str, ...], dict[str, str]]
) -> dict[tuple[str, ...], list[tuple[dict[str, str], dict[str, str]]]]:
    pairs: dict[tuple[str, ...], list[tuple[dict[str, str], dict[str, str]]]] = {}
    for row in rows:
        key = tuple(row[column] for column in JOIN)
        other = reference.get(key) or reference.get(key[:4])
        if other is not None:
            pairs.setdefault(tuple(row[column] for column in CELL), []).append(
                (row, other)
            )
    return pairs


def _select(
    rows: list[dict[str, str]], method: str, split: str
) -> list[dict[str, str]]:
    return [
        row for row in rows if row["method"] == method and row["eval_split"] == split
    ]


def _deltas(
    pairs: list[tuple[dict[str, str], dict[str, str]]], metric: str
) -> list[float]:
    values = [_number(row[metric]) - _number(other[metric]) for row, other in pairs]
    return [value for value in values if not math.isnan(value)]


def _row(
    key: tuple[str, ...], deltas: list[float], metric: str, args: argparse.Namespace
) -> dict[str, object]:
    # One fold gives one difference and no interval, which bootstrap_ci reports as NaN.
    low, high = bootstrap_ci(deltas)
    return {
        **dict(zip(CELL, key)),
        "method": args.method,
        "reference": args.baseline,
        "eval_split": args.split,
        "metric": metric,
        "n_folds": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "ci_low": low,
        "ci_high": high,
    }


def _table(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> list[dict[str, object]]:
    reference = _index(_select(rows, args.baseline, args.split))
    pairs = _pairs(_select(rows, args.method, args.split), reference)
    return [
        _row(key, deltas, metric, args)
        for key in sorted(pairs)
        for metric in args.metrics
        if (deltas := _deltas(pairs[key], metric))
    ]


def _write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    table = _table(_read(args.results), args)
    if not table:
        raise SystemExit(f"No {args.method} row pairs with a {args.baseline} row.")
    _write(table, args.out)
    for row in table:
        cell = " ".join(str(row[column]) for column in CELL)
        print(
            f"{cell:>40} {row['metric']:>10}  {row['mean_delta']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  n={row['n_folds']}"
        )
    print(f"\n{len(table)} difference(s) -> {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "results.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "paired_differences.csv"
    )
    parser.add_argument("--method", required=True, help="the method being measured")
    parser.add_argument(
        "--baseline", default="baseline", help="the method compared against"
    )
    parser.add_argument("--split", default="test", choices=("fold", "test"))
    parser.add_argument("--metrics", nargs="+", default=list(METRICS))
    return parser.parse_args()


if __name__ == "__main__":
    main()
