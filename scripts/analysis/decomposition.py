#!/usr/bin/env python3
"""The fidelity-invariance decomposition of every per-sample dump under the given paths.

One output row per (dump, magnitude bin, statistic): the mean of a per-sample term of
`src.metrics.decomposition` in top-1 or sqrt(JSD) form, or the student's agreement with
the teacher's errors at x and at g.x, with a percentile bootstrap interval over the
dump's samples. Without ``--bins`` only the whole draw is summarised. Bins are half-open,
[low, high), the last one closed; every magnitude also exists as ``abs_<name>``, and the
torus shift as ``shift_dist``, its Euclidean length in pixels.

    python3 scripts/analysis/decomposition.py --dumps logs/eval
    python3 scripts/analysis/decomposition.py --dumps logs/eval/jets --workers 16 \\
        --bins abs_rapidity:0,0.25,0.5,1 moved:0,10,30,50
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.decomposition import sample_columns, summarise_columns  # noqa: E402
from src.metrics.magnitude import bin_masks, with_derived  # noqa: E402
from src.utils.samples import read_samples  # noqa: E402
from src.vocabulary import RESULT_COLUMNS  # noqa: E402

IDENTITY = (
    *RESULT_COLUMNS[: RESULT_COLUMNS.index("accuracy")],
    "stage",
    "variant",
    "probe_tag",
    "checkpoint_path",
    "config_hash",
)
BIN = ("magnitude", "bin_low", "bin_high", "n")
STATISTIC = ("statistic", "mean", "ci_low", "ci_high")


def _dumps(paths: list[Path]) -> list[Path]:
    found = [sorted(p.glob("**/samples/*.npz")) if p.is_dir() else [p] for p in paths]
    return [dump for group in found for dump in group]


def _bins(spec: str) -> tuple[str, list[float]]:
    name, edges = spec.split(":")
    return name, [float(edge) for edge in edges.split(",")]


def _groups(arrays: dict[str, np.ndarray], bins: dict[str, list[float]]) -> list[tuple]:
    """(magnitude, low, high, mask): the whole draw, then each requested bin it has."""
    groups = [("", "", "", np.ones(len(arrays["targets"]), dtype=bool))]
    sizes = with_derived({n[4:]: v for n, v in arrays.items() if n.startswith("mag_")})
    for name in sorted(bins.keys() & sizes.keys()):
        groups += [(name, *mask) for mask in bin_masks(sizes[name], bins[name])]
    return [group for group in groups if group[-1].any()]


def _rows(
    path: Path, bins: dict[str, np.ndarray], n_boot: int, seed: int
) -> list[dict[str, object]]:
    arrays, meta = read_samples(path)
    temperatures = (meta["temperature_ts"], meta["temperature_teacher_ts"])
    columns = sample_columns(arrays, temperatures)
    identity = {column: meta.get(column, "") for column in IDENTITY}
    rows = []
    for name, low, high, mask in _groups(arrays, bins):
        cells = identity | dict(zip(BIN, (name, low, high, int(mask.sum()))))
        selected = {key: value[mask] for key, value in columns.items()}
        rows += [cells | stat for stat in summarise_columns(selected, n_boot, seed)]
    return rows


def main() -> None:
    args = _parse_args()
    dumps = _dumps(args.dumps)
    if not dumps:
        raise SystemExit(f"No dumps under {[str(p) for p in args.dumps]}.")
    bins = dict(map(_bins, args.bins))
    task = partial(_rows, bins=bins, n_boot=args.n_boot, seed=args.seed)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for part in pool.map(task, dumps) for row in part]
    _write(rows, args.out)
    print(f"{len(dumps)} dump(s), {len(rows)} row(s) -> {args.out}")


def _write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*IDENTITY, *BIN, *STATISTIC])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dumps", type=Path, nargs="+", default=[ROOT / "logs" / "eval"]
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "decomposition.csv"
    )
    parser.add_argument("--bins", nargs="*", default=[], help="name:edge,edge,...")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
