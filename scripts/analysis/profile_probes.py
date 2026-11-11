#!/usr/bin/env python3
"""Fold summaries of the orbit profiles and concept probes, for H2 and H3 of Block 1.

Every arm (a checkpoint's identity with its fold left out, per probe and layer, or per
probe target and layer) gets the mean over folds 1 to 5 of the test split with the
corrected resampled t-interval (`src.utils.statistics`). A random-initialisation row has
no fold (``fold`` main) and is reported as its single value. Three paired comparisons
follow, each per fold and then corrected over the folds:

- ``control``: a cell teacher's student against the control-teacher student of its cell
  (same group, regime, method, temperature and alpha), for every control teacher found;
- ``random_init``: a cell teacher's student against the same architecture at its seeded
  initialisation, matched by cell, regime and rows alone;
- ``hint_vs_kd``: the Hint student against the logit-KD student of the same cell, regime
  and alpha, each at its selected temperature (H2a).

Block 2's erasure rule: the frame counts as used at a layer where the LEACE effect on
accuracy, fidelity or top-1 invariance lies outside the 95 % range of the equal-rank
random removals in at least two of the five folds, with the same sign in all five.

    python3 scripts/analysis/profile_probes.py --profile profile.csv --probes probes.csv \\
        --out-dir results/profile_probes
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.statistics import corrected_t  # noqa: E402
from src.vocabulary import CONTROL_TEACHERS, MAIN_FOLD  # noqa: E402

POINT = ("dataset", "teacher", "group", "regime", "method", "temperature", "alpha")
POINT += ("beta", "discrepancy", "eval_data", "init")
ARM = {
    "profile": (*POINT, "probe_group", "layer"),
    "probes": (*POINT, "kind", "target", "statistic", "layer"),
}
VALUE = {"profile": "invariance", "probes": "value"}
KNOBS = ("method", "temperature", "alpha", "beta", "discrepancy")
STATS = ("mean", "se", "t", "df", "p_value", "ci_low", "ci_high")
ERASURE = ("d_accuracy_g", "d_fidelity_g", "d_invariance")
Row = dict[str, str]
Key = tuple[str, ...]


def _read(path: Path) -> list[dict[str, str]]:
    """The test-split rows of no variant."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [r for r in rows if r["eval_split"] == "test" and not r["variant"]]


def _key(row: dict[str, str], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row[c] for c in columns)


def _by_arm(rows: list[Row], arm: Key, value: str) -> dict[Key, dict[str, float]]:
    """Per arm, its value on every fold."""
    folds: dict[Key, dict[str, float]] = {}
    for r in rows:
        folds.setdefault(_key(r, arm), {})[r["fold"]] = float(r[value])
    return folds


def _stats(values: list[float], ratio: float) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": values[0]} | dict.fromkeys(STATS[1:], math.nan)
    return corrected_t(values, ratio)


def _summary(rows: list[Row], table: str, ratio: float) -> list[dict[str, object]]:
    out = []
    for key, folds in sorted(_by_arm(rows, ARM[table], VALUE[table]).items()):
        values = [v for f, v in folds.items() if f != MAIN_FOLD] or list(folds.values())
        cells = dict(zip(ARM[table], key)) | {"n_folds": len(values)}
        out.append(cells | _stats(values, ratio))
    return out


def _paired(
    students: list[Row], references: list[Row], match: Key, value: str
) -> dict[tuple[tuple[str, str], ...], list[float]]:
    """Per student arm, the per-fold differences to the reference matched on *match*; a
    reference without folds (random init) serves every fold."""
    index = {(*_key(r, match), r["fold"]): float(r[value]) for r in references}
    deltas: dict[tuple[tuple[str, str], ...], list[float]] = {}
    for r in students:
        other = index.get((*_key(r, match), r["fold"]))
        other = index.get((*_key(r, match), MAIN_FOLD)) if other is None else other
        if other is not None and r["fold"] != MAIN_FOLD:
            deltas.setdefault(tuple(r.items()), []).append(float(r[value]) - other)
    return deltas


def _without(columns: tuple[str, ...], dropped: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in columns if c not in dropped)


def _comparisons(
    rows: list[Row], table: str
) -> Iterator[tuple[str, str, list[Row], list[Row], Key]]:
    """(name, reference, students, references, match) for every comparison."""
    arm, trained = ARM[table], [r for r in rows if r["init"] == "trained"]
    cells = [r for r in trained if r["teacher"] not in CONTROL_TEACHERS]
    for control in sorted({r["teacher"] for r in trained} & set(CONTROL_TEACHERS)):
        refs = [r for r in trained if r["teacher"] == control]
        yield "control", control, cells, refs, _without(arm, ("teacher",))
    random = [r for r in rows if r["init"] == "random"]
    yield "random_init", "random", cells, random, _without(arm, (*KNOBS, "init"))
    hint, kd = ([r for r in cells if r["method"] == m] for m in ("hint", "kd"))
    yield "hint_vs_kd", "kd", hint, kd, _without(arm, ("method", "temperature", "beta"))


def _differences(rows: list[Row], table: str, ratio: float) -> list[dict[str, object]]:
    out, arm = [], ARM[table]
    for name, reference, students, refs, match in _comparisons(rows, table):
        merged: dict[Key, list[float]] = {}
        for row, deltas in _paired(students, refs, match, VALUE[table]).items():
            merged.setdefault(_key(dict(row), arm), []).extend(deltas)
        for key, deltas in sorted(merged.items()):
            cells = dict(zip(arm, key)) | {"comparison": name, "reference": reference}
            out.append(cells | {"n_folds": len(deltas)} | _stats(deltas, ratio))
    return out


def _is_erasure(r: Row) -> bool:
    return (
        r["kind"] == "erasure" and r["statistic"] in ERASURE and r["fold"] != MAIN_FOLD
    )


def _erasure(rows: list[Row]) -> list[dict[str, object]]:
    """Block 2's rule per arm, layer and statistic, the folds counted."""
    arm = ARM["probes"]
    by_arm: dict[Key, list[Row]] = {}
    for r in filter(_is_erasure, rows):
        by_arm.setdefault(_key(r, arm), []).append(r)
    return [dict(zip(arm, key)) | _rule(folds) for key, folds in sorted(by_arm.items())]


def _rule(folds: list[dict[str, str]]) -> dict[str, object]:
    values = [float(r["value"]) for r in folds]
    outside = sum(
        v < float(r["control_low"]) or v > float(r["control_high"])
        for v, r in zip(values, folds)
    )
    same_sign = all(v > 0 for v in values) or all(v < 0 for v in values)
    used = len(folds) == 5 and outside >= 2 and same_sign
    return {"n_folds": len(folds), "n_outside": outside, "same_sign": same_sign} | {
        "used": used
    }


def _write(rows: list[dict[str, object]], path: Path) -> None:
    columns = list(dict.fromkeys(c for row in rows for c in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for table, path in (("profile", args.profile), ("probes", args.probes)):
        rows = _read(path)
        _write(_summary(rows, table, args.ratio), args.out_dir / f"{table}_folds.csv")
        diffs = _differences(rows, table, args.ratio)
        _write(diffs, args.out_dir / f"{table}_differences.csv")
        if table == "probes":
            _write(_erasure(rows), args.out_dir / "erasure_rule.csv")
    print(f"tables -> {args.out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.25, help="n_test / n_train")
    return parser.parse_args()


if __name__ == "__main__":
    main()
