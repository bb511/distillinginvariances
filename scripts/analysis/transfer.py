#!/usr/bin/env python3
"""The transfer fraction of every distilled row, as Block 3 of the registration states.

Per fold, (row - baseline) / (ceiling - baseline) in ``inv_top1`` under the cell's own
probe, each read on the same rows (eval_data): the baseline of the row's own regime, the
ceiling the cell's augmented baseline evaluated on those rows. Over the folds, the fraction
and the row's paired gain over the baseline get the corrected resampled t-interval
(`src.utils.statistics`), and the gain the TOST p-value against the cell's margin.

The margin is Blocks 5 and 6's: max(0.02, 0.1 x (ceiling - baseline)), each the mean over
the seeds of the main run's validation rows, read per regime on that regime's rows (the
ceiling of a canonical-regime row is the augmented baseline evaluated on canonical rows).
Where the ceiling's gain over the baseline there is below the margin, the fraction is
undefined and reported as such (``fraction_defined`` False, fraction blank). Rows of a
variant (``variant``, the double-width and p4 students) are left out, so that only the
registered baseline serves as B and C.

    python3 scripts/analysis/transfer.py
    python3 scripts/analysis/transfer.py --split fold --out results/transfer_fold.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.statistics import (  # noqa: E402
    corrected_t,
    equivalence_p,
    transfer_fraction,
)
from src.vocabulary import MAIN_FOLD  # noqa: E402

# Blocks 2 and 5: margin = max(FLOOR, SHARE x (ceiling - baseline)) in top-1 invariance.
FLOOR, SHARE = 0.02, 0.1
PROBE = (
    "eval_data",
    "probe_group",
    "probe_eta_max",
    "probe_max_shift",
    "probe_n_swaps",
)
# What a row, its baseline and its ceiling share, beside the fold and the split.
CELL = ("dataset", "teacher", *PROBE)
# What a row of the output is per, the fold averaged out.
ARM = ("dataset", "teacher", "regime", "method", "temperature", "alpha", "beta")
ARM += ("discrepancy", "label_smoothing", "eval_split", *PROBE)
STATS = ("mean", "ci_low", "ci_high")
OUT_COLUMNS = (
    *ARM,
    "metric",
    "n_folds",
    "gap_main",
    "margin",
    "fraction_defined",
    *(f"fraction_{s}" for s in STATS),
    *(f"gain_{s}" for s in STATS),
    "tost_p",
)


def _read(path: Path) -> list[dict[str, str]]:
    """Every row of no variant under its cell's own probe."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [r for r in rows if r["probe_group"] == r["group"] and not r["variant"]]


def _baselines(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    """The baseline rows, keyed by cell, fold, split and regime."""
    return {
        (*(r[c] for c in CELL), r["fold"], r["eval_split"], r["regime"]): r
        for r in rows
        if r["method"] == "baseline"
    }


def _main_means(
    rows: list[dict[str, str]], metric: str
) -> dict[tuple[str, ...], float]:
    """The seed mean of every baseline on the main run's validation split, keyed by cell
    and regime; a seed evaluated twice counts once, at its last row."""
    by_seed: dict[tuple[str, ...], dict[str, float]] = {}
    for r in rows:
        if (
            r["method"] == "baseline"
            and r["fold"] == MAIN_FOLD
            and r["eval_split"] == "fold"
        ):
            key = (*(r[c] for c in CELL), r["regime"])
            by_seed.setdefault(key, {})[r["seed"]] = float(r[metric])
    return {key: sum(v.values()) / len(v) for key, v in by_seed.items()}


def _margin(
    means: dict[tuple[str, ...], float], row: dict[str, str]
) -> tuple[float, float]:
    """The ceiling's gain over the row's baseline on the main run's validation split,
    and the margin it gives; NaN where either main-run row is missing."""
    cell = tuple(row[c] for c in CELL)
    own, ceiling = means.get((*cell, row["regime"])), means.get((*cell, "augmented"))
    if own is None or ceiling is None:
        return math.nan, math.nan
    return ceiling - own, max(FLOOR, SHARE * (ceiling - own))


def _pair(
    row: dict[str, str], baselines: dict[tuple[str, ...], dict[str, str]], metric: str
) -> tuple[float, float] | None:
    """The fold's (fraction, gain) of a distilled row with its baseline and ceiling."""
    if row["method"] in ("baseline", "teacher", "constant"):
        return None
    shared = (*(row[c] for c in CELL), row["fold"], row["eval_split"])
    own = baselines.get((*shared, row["regime"]))
    ceiling = baselines.get((*shared, "augmented"))
    if own is None or ceiling is None:
        return None
    student, base, top = (float(r[metric]) for r in (row, own, ceiling))
    return transfer_fraction(student, base, top), student - base


def _arms(
    rows: list[dict[str, str]], split: str, metric: str
) -> dict[tuple[str, ...], tuple[dict[str, str], list[tuple[float, float]]]]:
    """Per arm, one of its rows and the (fraction, gain) of every fold."""
    baselines, arms = _baselines(rows), {}
    for row in rows:
        if row["fold"] == MAIN_FOLD or row["eval_split"] != split:
            continue
        pair = _pair(row, baselines, metric)
        if pair is not None:
            arm = tuple(row[c] for c in ARM)
            arms.setdefault(arm, (row, []))[1].append(pair)
    return arms


def _row(
    arm: tuple[str, ...],
    pairs: list[tuple[float, float]],
    margin: tuple[float, float],
    metric: str,
    ratio: float,
) -> dict[str, object]:
    gap, limit = margin
    defined = not math.isnan(gap) and gap >= limit
    fractions = [f for f, _ in pairs if not math.isnan(f)] if defined else []
    gains = [g for _, g in pairs]
    fraction, gain = corrected_t(fractions, ratio), corrected_t(gains, ratio)
    cells = dict(zip(ARM, arm)) | {"metric": metric, "n_folds": len(pairs)}
    cells |= {"gap_main": gap, "margin": limit, "fraction_defined": defined}
    cells |= {f"fraction_{s}": fraction[s] if defined else "" for s in STATS}
    cells |= {f"gain_{s}": gain[s] for s in STATS}
    tost = equivalence_p(gains, ratio, limit) if not math.isnan(limit) else math.nan
    return cells | {"tost_p": tost}


def main() -> None:
    args = _parse_args()
    rows = _read(args.results)
    arms, means = _arms(rows, args.split, args.metric), _main_means(rows, args.metric)
    if not arms:
        raise SystemExit("No distilled row has both its baseline and its ceiling.")
    table = [
        _row(arm, pairs, _margin(means, row), args.metric, args.ratio)
        for arm, (row, pairs) in sorted(arms.items())
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(table)} arm(s) -> {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "results.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "transfer.csv")
    parser.add_argument("--metric", default="inv_top1", help="Block 3: inv_top1")
    parser.add_argument("--split", default="test", choices=("fold", "test"))
    parser.add_argument("--ratio", type=float, default=0.25, help="n_test / n_train")
    return parser.parse_args()


if __name__ == "__main__":
    main()
