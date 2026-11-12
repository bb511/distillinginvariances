#!/usr/bin/env python3
"""Select each arm's point and each cell's firmware student, as Block 3 of the
registration states.

Selection reads the main run's validation rows alone (``eval_split`` fold, ``fold`` main),
on the run's own rows under its cell's own probe, and ranks by ``accuracy_macro``, which
on those rows is the training loop's ``val/acc_best``. Within an arm (cell, regime,
channel, alpha, discrepancy) the point of highest score wins; points within 0.001 of it
tie, and a tie goes to the lower temperature, then the lower beta. A selected temperature
at an end of the grid {1, ..., 16} whose next point (0.5 or 32) is absent is flagged in
``extend_with``; once that point exists the rule applies once more and flags nothing.
A Hint-beta arm, a beta grid at one selected temperature, is never flagged. The
firmware student of a cell is the canonical-regime distilled row of highest score
among the selected points. The label-smoothed students are a curve over epsilon,
reported whole and never selected. Only a cell teacher's own grid, at the default seed, is
ranked: the control teachers' rows, the variants (``variant``) and the extra seeds are
trained at the selected point.

python3 scripts/analysis/select_hparams.py python3 scripts/analysis/select_hparams.py
--results results/results.csv --out results/hyperparameters.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.vocabulary import CONTROL_TEACHERS, DATA_OF_REGIME, MAIN_FOLD  # noqa: E402

# One selection per arm the paper reports separately.
KEY = ("dataset", "teacher", "regime", "method", "discrepancy", "alpha")
SCORE = "accuracy_macro"
# Scores closer than this tie (Block 3).
TIE = 0.001
# The grid ends and the point each adds when selected (Block 3's edge rule).
EDGES = {1.0: 0.5, 16.0: 32.0}
# The methods trained over beta at one selected temperature: Block 3 gives the edge rule
# to the temperature grid alone, so their arms are never flagged.
BETA_ARMS = ("hint_beta",)
# Label smoothing is reported at every epsilon, so it has no selected point.
NOT_SELECTED = ("teacher", "constant", "label_smoothing")
# The grid is trained at the default seed. The extra seeds of a headline row come after its
# selection and give intervals, never a selection.
SELECTION_SEED = "12345"
NOT_DISTILLED = ("baseline", "teacher", "constant", "label_smoothing")
# The knobs a catalogue substitutes into the fold commands, and their override names.
OVERRIDE_OF = {
    "temperature": "algorithm.temperature",
    "alpha": "algorithm.alpha",
    "beta": "algorithm.beta",
    "discrepancy": "algorithm.discrepancy",
    "lambda_is": "algorithm.lambda_is",
    "lambda_ic": "algorithm.lambda_ic",
    "lr": "algorithm.optimizer.lr",
}
# The MNIST experiments, whose names are not <dataset>_<teacher>.
EXPERIMENT_OF = {"cnn_circular": "mnist_cnn", "cnn_zeros": "mnist_cnn_zeros"}
OUT_COLUMNS = (
    *KEY,
    "stage",
    "lr",
    "temperature",
    "beta",
    "lambda_is",
    "lambda_ic",
    "batch_size",
    SCORE,
    "accuracy",
    "n_points",
    "extend_with",
    "run_dir",
    "overrides",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str, missing: float) -> float:
    """*value* as a float, with a blank or a NaN read as *missing*."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return missing if math.isnan(number) else number


def _grid_point(row: dict[str, str]) -> bool:
    """A point of a cell teacher's own grid: neither a control teacher's row, nor a
    variant (double width, four-momentum) run at the selected point, nor an extra
    seed."""
    return (
        row["teacher"] not in CONTROL_TEACHERS
        and not row.get("variant")
        and (row.get("seed") or SELECTION_SEED) == SELECTION_SEED
    )


def _candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # One row per checkpoint: its own rows under its cell's probe. The other probe rows
    # repeat its scores, and another regime's rows are no validation of it.
    return [
        row
        for row in rows
        if row["eval_split"] == "fold"
        and row["fold"] == MAIN_FOLD
        and row["method"] not in NOT_SELECTED
        and row["probe_group"] == row["group"]
        and row["eval_data"] == DATA_OF_REGIME[row["regime"]]
        and _grid_point(row)
    ]


def _arms(rows: list[dict[str, str]]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    arms: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        arms.setdefault(tuple(row[column] for column in KEY), []).append(row)
    return arms


def _select(points: list[dict[str, str]]) -> dict[str, str]:
    """The arm's point: highest score, ties within `TIE` to the lower T, then beta."""
    top = max(_number(row[SCORE], -math.inf) for row in points)
    tied = [row for row in points if _number(row[SCORE], -math.inf) >= top - TIE]
    knobs = ("temperature", "beta")
    best = min(tied, key=lambda row: tuple(_number(row[k], 0.0) for k in knobs))
    return best | {
        "n_points": str(len(points)),
        "extend_with": _extension(best, points),
    }


def _extension(best: dict[str, str], points: list[dict[str, str]]) -> str:
    """The temperature still to train when the selection sits on a grid end."""
    if best["method"] in BETA_ARMS:
        return ""
    selected = _number(best["temperature"], math.nan)
    nxt = EDGES.get(selected)
    trained = {_number(row["temperature"], math.nan) for row in points}
    return "" if nxt is None or nxt in trained else str(nxt)


def _firmware(selected: list[dict[str, str]]) -> list[dict[str, str]]:
    """Per cell, the canonical-regime distilled selection of highest score."""
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        if row["regime"] != "canonical" or row["method"] in NOT_DISTILLED:
            continue
        cell = (row["dataset"], row["teacher"])
        score = _number(row[SCORE], -math.inf)
        if cell not in cells or score > _number(cells[cell][SCORE], -math.inf):
            cells[cell] = row
    return [cells[cell] for cell in sorted(cells)]


def _experiment(row: dict[str, str]) -> str:
    default = f"{row['dataset']}_{row['teacher']}"
    return EXPERIMENT_OF.get(row["teacher"], default)


def _overrides(row: dict[str, str]) -> str:
    """The command line that trains this point on another fold."""
    head = [
        f"experiment={_experiment(row)}",
        f"method={row['stage']}",
        f"data.regime={row['regime']}",
    ]
    knobs = [
        f"{name}={row[column]}" for column, name in OVERRIDE_OF.items() if row[column]
    ]
    return " ".join(head + knobs)


def _write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(
            {c: row.get(c, "") for c in OUT_COLUMNS} | {"overrides": _overrides(row)}
            for row in rows
        )


def main() -> None:
    args = _parse_args()
    arms = _arms(_candidates(_read(args.results)))
    if not arms:
        raise SystemExit(f"{args.results} holds no main-run validation rows.")
    selected = [_select(arms[key]) for key in sorted(arms)]
    _write(selected, args.out)
    _write(_firmware(selected), args.firmware_out)
    for row in selected:
        flag = f"  extend with T={row['extend_with']}" if row["extend_with"] else ""
        print(f"{row[SCORE]:>10}  {_overrides(row)}{flag}")
    print(f"\n{len(selected)} arm(s) -> {args.out}; firmware -> {args.firmware_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "results.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "hyperparameters.csv"
    )
    parser.add_argument(
        "--firmware-out", type=Path, default=ROOT / "results" / "firmware.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
