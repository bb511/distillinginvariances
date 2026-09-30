#!/usr/bin/env python3
"""Pick each method's hyperparameter point from the fold-1 evaluations.

The study selects on fold 1 and then fills folds 2 to 5 at the selected point. Selection
reads the ``fold`` rows (the validation half), never the ``test`` ones, by accuracy with
the NLL as the tie-break, and writes one row per arm to ``results/hyperparameters.csv``
along with the Hydra overrides that reproduce it.

python3 scripts/analysis/select_hparams.py python3 scripts/analysis/select_hparams.py
--results results/results.csv --fold 1
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# One selection per arm the paper reports separately: the discrepancy and the CE weight are
# columns of its tables, so each of them keeps its own best point.
KEY = ("dataset", "teacher", "regime", "method", "discrepancy", "alpha")
# The knobs a catalogue substitutes into the folds 2 to 5 commands, and their override names.
OVERRIDE_OF = {
    "temperature": "algorithm.temperature",
    "alpha": "algorithm.alpha",
    "beta": "algorithm.beta",
    "discrepancy": "algorithm.discrepancy",
    "lambda_is": "algorithm.lambda_is",
    "lambda_ic": "algorithm.lambda_ic",
    "lr": "algorithm.optimizer.lr",
}
EXPERIMENT_OF = {
    "cnn_circular": "mnist_cnn",
    "cnn_zeros": "mnist_cnn",
    "deepsets": "jets_deepsets",
    "lorentznet": "jets_lorentznet",
}
OUT_COLUMNS = (
    *KEY,
    "stage",
    "lr",
    "temperature",
    "beta",
    "lambda_is",
    "lambda_ic",
    "batch_size",
    "accuracy",
    "nll",
    "run_dir",
    "overrides",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str, missing: float) -> float:
    """*value* as a float, with a blank or a NaN read as *missing*, so ranking is
    total."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return missing if math.isnan(number) else number


def _rank(row: dict[str, str]) -> tuple[float, float]:
    # Highest accuracy first; the NLL breaks a tie, which a coarse accuracy grid does produce.
    return -_number(row["accuracy"], -math.inf), _number(row["nll"], math.inf)


def _candidates(rows: list[dict[str, str]], fold: int) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["eval_split"] == "fold"
        and int(row["fold"]) == fold
        and row["method"] not in ("teacher", "constant")
    ]


def _best_per_arm(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[column] for column in KEY)
        if key not in best or _rank(row) < _rank(best[key]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def _overrides(row: dict[str, str]) -> str:
    """The command line that trains this point on another fold."""
    head = [
        f"experiment={EXPERIMENT_OF.get(row['teacher'], row['teacher'])}",
        f"method={row['stage']}",
        f"data.regime={row['regime']}",
    ]
    knobs = [
        f"{name}={row[column]}" for column, name in OVERRIDE_OF.items() if row[column]
    ]
    return " ".join(head + knobs)


def _out_row(row: dict[str, str]) -> dict[str, str]:
    picked = {column: row.get(column, "") for column in OUT_COLUMNS}
    picked["overrides"] = _overrides(row)
    return picked


def _write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    rows = _best_per_arm(_candidates(_read(args.results), args.fold))
    if not rows:
        raise SystemExit(f"{args.results} holds no fold-{args.fold} validation rows.")
    _write([_out_row(row) for row in rows], args.out)
    for row in rows:
        print(f"{row['accuracy']:>10}  {_overrides(row)}")
    print(f"\n{len(rows)} arm(s) selected on fold {args.fold} -> {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "results.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "hyperparameters.csv"
    )
    parser.add_argument("--fold", type=int, default=1, help="the fold selected on")
    return parser.parse_args()


if __name__ == "__main__":
    main()
