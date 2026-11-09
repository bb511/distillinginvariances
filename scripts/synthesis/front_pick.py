#!/usr/bin/env python3
"""Pick the epoch of a quantised run that goes to synthesis under the latency budget
(03_student.md, 9 Nov 2026).

A design that misses ``--budget-ns`` takes the epoch of its HGQ2 front with the highest
validation accuracy that meets it. Every checkpoint the run kept (``method=hgq`` keeps one
every 10 epochs) is exported on the run's validation rows (``export_hgq.py --split fold``,
so nothing reads the test split) and estimated by alkaid (``synthesise.py
--estimate-only``); the pick is the most accurate epoch within the budget, the fewest
EBOPs on a tie. The most accurate feasible epoch lies on the front of accuracy against
latency, so no front is built. Each epoch's row goes to ``<record>/front.csv`` and the pick
to ``<record>/pick.json``; an epoch whose estimate exists is not run again::

    python3 scripts/synthesis/front_pick.py --run-dir <hgq run> \\
        --work /scratch/distilling_symmetries_bulk/front/<design> --record logs/front/<design>
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIELDS = (
    "epoch",
    "checkpoint",
    "accuracy",
    "ebops",
    "cycles",
    "latency_ns",
    "lut",
    "dsp",
)


def pick(rows: list[dict], budget_ns: float) -> dict | None:
    """The most accurate row within *budget_ns*, the fewest EBOPs on a tie; None when no
    row meets it."""
    feasible = [row for row in rows if row["latency_ns"] <= budget_ns]
    return max(feasible, key=lambda r: (r["accuracy"], -r["ebops"]), default=None)


def _run(command: list[str], log: Path) -> None:
    """:raises subprocess.CalledProcessError: when it fails; *log* holds why."""
    with log.open("w") as out:
        subprocess.run(
            command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True
        )


def epoch_row(run_dir: Path, checkpoint: Path, work: Path) -> dict:
    """Export and estimate one checkpoint under *work*, unless done before."""
    out = work / checkpoint.stem
    if not (out / "estimate" / "estimate.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        export = [sys.executable, "scripts/synthesis/export_hgq.py", "--run-dir"]
        export += [
            str(run_dir),
            "--ckpt",
            str(checkpoint),
            "--out",
            str(out / "export"),
        ]
        _run(export + ["--split", "fold"], out / "export.log")
        estimate = [
            sys.executable,
            "scripts/synthesis/synthesise.py",
            "--estimate-only",
        ]
        estimate += [
            "--export-dir",
            str(out / "export"),
            "--out",
            str(out / "estimate"),
        ]
        _run(estimate, out / "estimate.log")
    found = json.loads((out / "estimate" / "estimate.json").read_text())
    meta = json.loads((out / "export" / "export.json").read_text())
    row = {"epoch": checkpoint.stem, "checkpoint": str(checkpoint)}
    row |= {"accuracy": meta["accuracy"], "ebops": meta["ebops"]}
    return row | {key: found[key] for key in ("cycles", "latency_ns", "lut", "dsp")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="an hgq run")
    parser.add_argument("--work", type=Path, required=True, help="bulk outputs")
    parser.add_argument("--record", type=Path, required=True, help="csv and pick")
    parser.add_argument("--budget-ns", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.record.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted((args.run_dir / "checkpoints").glob("epoch_*.ckpt"))
    rows = [epoch_row(args.run_dir, c, args.work) for c in checkpoints]
    with (args.record / "front.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    chosen = pick(rows, args.budget_ns)
    if chosen is None:
        print(f"no epoch of {args.run_dir} within {args.budget_ns} ns", file=sys.stderr)
        return 1
    (args.record / "pick.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
