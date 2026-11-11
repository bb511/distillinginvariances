#!/usr/bin/env python3
"""Per-fold differences between one method and a reference, with the corrected t-test.

Each fold's run is compared with the reference run of the same cell and the same fold,
on the same rows under the same probe, so the fold-to-fold spread of the pool cancels.
The folds share most of their training rows, so the mean difference is tested by the
corrected resampled t-test (`src.utils.statistics`, ratio n_test / n_train = 1 / (K - 1),
0.25 for five folds); with --margin, the TOST p-value for a difference within it is
added. With --reference-teacher, the reference rows are that teacher's (a control teacher)
and a pair shares the cell's group in place of its teacher, so that a student is compared
with the control-teacher student of its cell at the same temperature and alpha (H1a, H1c).
Rows of a variant other than --variant (none by default) are left out.

    python3 scripts/analysis/paired_differences.py --method kd
    python3 scripts/analysis/paired_differences.py --method relational --baseline kd \\
        --split test --metrics accuracy nll fid_top1
    python3 scripts/analysis/paired_differences.py --method kd --baseline kd \\
        --reference-teacher mlp --margin 0.05
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.statistics import corrected_t, equivalence_p  # noqa: E402
from src.vocabulary import MAIN_FOLD  # noqa: E402

# The rows a pair must share beside its cell: the data evaluated and the probe.
PROBE = (
    "eval_data",
    "probe_group",
    "probe_eta_max",
    "probe_max_shift",
    "probe_n_swaps",
)
# The knobs a method shares with the method it is compared against (kd against hint at one
# temperature, say), or that split its own rows (label smoothing at each epsilon); a
# baseline carries none.
SHARED_KNOBS = ("temperature", "alpha", "label_smoothing")
# The cell a pairing happens inside, the fold, the probe and the shared knobs.
JOIN = ("dataset", "teacher", "regime", "fold", *PROBE, *SHARED_KNOBS)
# What the pairs are reported per, the fold having been averaged out of them.
CELL = ("dataset", "teacher", "regime", *PROBE, *SHARED_KNOBS)
METRICS = ("accuracy", "nll", "ece", "fid_top1", "inv_top1", "inv_sqrtjsd")
OUT_COLUMNS = (
    *CELL,
    "method",
    "reference",
    "eval_split",
    "metric",
    "n_folds",
    "mean_delta",
    "se",
    "t",
    "df",
    "p_value",
    "ci_low",
    "ci_high",
    "margin",
    "tost_p",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


_SHORT = len(JOIN) - len(SHARED_KNOBS)


def _join(args: argparse.Namespace) -> tuple[str, ...]:
    # A control teacher differs from the cell's teacher, so its rows pair by group.
    if args.reference_teacher is None:
        return JOIN
    return tuple("group" if column == "teacher" else column for column in JOIN)


def _index(
    rows: list[dict[str, str]], join: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    # A baseline carries no temperature and no alpha, so the full key of a distilled run
    # never matches it and the short key is what finds it. Only such a row gets the short
    # key: a distilled reference pairs at its own knobs alone.
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        index[tuple(row[column] for column in join)] = row
        if not any(row[column] for column in SHARED_KNOBS):
            index.setdefault(tuple(row[column] for column in join[:_SHORT]), row)
    return index


def _pairs(
    rows: list[dict[str, str]],
    reference: dict[tuple[str, ...], dict[str, str]],
    join: tuple[str, ...],
) -> dict[tuple[str, ...], list[tuple[dict[str, str], dict[str, str]]]]:
    pairs: dict[tuple[str, ...], list[tuple[dict[str, str], dict[str, str]]]] = {}
    for row in rows:
        key = tuple(row[column] for column in join)
        other = reference.get(key) or reference.get(key[:_SHORT])
        if other is not None:
            pairs.setdefault(tuple(row[column] for column in CELL), []).append(
                (row, other)
            )
    return pairs


def _select(
    rows: list[dict[str, str]], method: str, args: argparse.Namespace
) -> list[dict[str, str]]:
    # The main run is no fold, so it never enters a per-fold difference.
    return [
        row
        for row in rows
        if row["method"] == method
        and row["eval_split"] == args.split
        and row["fold"] != MAIN_FOLD
        and row["variant"] == args.variant
    ]


def _deltas(
    pairs: list[tuple[dict[str, str], dict[str, str]]], metric: str
) -> list[float]:
    values = [_number(row[metric]) - _number(other[metric]) for row, other in pairs]
    return [value for value in values if not math.isnan(value)]


def _row(
    key: tuple[str, ...], deltas: list[float], metric: str, args: argparse.Namespace
) -> dict[str, object]:
    test = corrected_t(deltas, args.ratio)
    reference = args.baseline
    if args.reference_teacher is not None:
        reference = f"{args.reference_teacher}:{args.baseline}"
    cells = dict(zip(CELL, key)) | {"method": args.method, "reference": reference}
    cells |= {"eval_split": args.split, "metric": metric, "n_folds": len(deltas)}
    cells |= {"mean_delta": test.pop("mean")} | test
    if args.margin is None:
        return cells
    return cells | {
        "margin": args.margin,
        "tost_p": equivalence_p(deltas, args.ratio, args.margin),
    }


def _table(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> list[dict[str, object]]:
    join, own = _join(args), _select(rows, args.method, args)
    references = _select(rows, args.baseline, args)
    if args.reference_teacher is not None:
        own = [r for r in own if r["teacher"] != args.reference_teacher]
        references = [r for r in references if r["teacher"] == args.reference_teacher]
    pairs = _pairs(own, _index(references, join), join)
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
    parser.add_argument("--ratio", type=float, default=0.25, help="n_test / n_train")
    parser.add_argument("--margin", type=float, default=None, help="TOST margin")
    parser.add_argument(
        "--reference-teacher", default=None, help="the control teacher compared against"
    )
    parser.add_argument("--variant", default="", help="the variant compared (none)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
