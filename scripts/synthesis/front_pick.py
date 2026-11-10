#!/usr/bin/env python3
"""Pick the epoch of a quantised run that goes to synthesis under the latency budget
(03_student.md, 9 Nov 2026), or at most a target EBOPs (08_tests.md, Block 7).

A design that misses ``--budget-ns`` takes the epoch of its HGQ2 front with the highest
validation accuracy that meets it. Every checkpoint the run kept (``method=hgq`` keeps one
every 10 epochs) is exported on the run's validation rows (``export_hgq.py --split fold``,
so nothing reads the test split) and estimated by alkaid (``synthesise.py
--estimate-only``); the pick is the most accurate epoch within the budget, the fewest
EBOPs on a tie. The most accurate feasible epoch lies on the front of accuracy against
latency, so no front is built. Each epoch's row goes to ``<record>/front.csv`` and the pick
to ``<record>/pick.json``; an epoch whose export or estimate exists is not run again::

    python3 scripts/synthesis/front_pick.py --run-dir <hgq run> \\
        --work /scratch/distilling_symmetries_bulk/front/<design> --record logs/front/<design>

Under ``--max-ebops`` (Block 7, a distilled design matched to its baseline's EBOPs) every
kept epoch is exported, and the pick is the epoch of highest macro accuracy on the
validation rows among those whose exported EBOPs are at or below the target, the fewest
EBOPs on a tie, that meets ``--budget-ns``; only the candidates are estimated, in that
order. An epoch that Vivado could not route is named in ``--skip``, and the next takes
its place. ``front.csv`` then carries the estimate of every candidate, the ones that gave
way included, and flags the skipped epochs.
"""

import argparse
import csv
import functools
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIELDS = (
    "epoch",
    "checkpoint",
    "accuracy",
    "accuracy_macro",
    "ebops",
    "cycles",
    "latency_ns",
    "lut",
    "dsp",
    "skipped",
)


def pick(rows: list[dict], budget_ns: float) -> dict | None:
    """The most accurate row within *budget_ns*, the fewest EBOPs on a tie; None when no
    row meets it."""
    feasible = [row for row in rows if row["latency_ns"] <= budget_ns]
    return max(feasible, key=lambda r: (r["accuracy"], -r["ebops"]), default=None)


def matched_candidates(
    rows: list[dict],
    max_ebops: int,
    budget_ns: float,
    estimate: Callable[[dict], dict],
    skip: set[str],
) -> list[dict]:
    """Block 7's candidates, estimated in order of macro accuracy, then fewest EBOPs, at
    or below *max_ebops* and outside *skip*, up to the first that meets *budget_ns*."""
    eligible = [r for r in rows if r["ebops"] <= max_ebops and r["epoch"] not in skip]
    estimated = []
    for row in sorted(eligible, key=lambda r: (-r["accuracy_macro"], r["ebops"])):
        estimated.append(estimate(row))
        if estimated[-1]["latency_ns"] <= budget_ns:
            break
    return estimated


def matched_front(
    rows: list[dict],
    max_ebops: int,
    budget_ns: float,
    estimate: Callable[[dict], dict],
    skip: set[str],
) -> tuple[list[dict], dict | None]:
    """Every row with its estimate when it was a candidate and whether Vivado's failure
    skipped it, so that the record shows each epoch that gave way; and Block 7's
    pick."""
    estimated = matched_candidates(rows, max_ebops, budget_ns, estimate, skip)
    by_epoch = {row["epoch"]: row for row in estimated}
    rows = [by_epoch.get(r["epoch"], r) | {"skipped": r["epoch"] in skip} for r in rows]
    found = bool(estimated) and estimated[-1]["latency_ns"] <= budget_ns
    return rows, estimated[-1] if found else None


def macro_accuracy(export_dir: Path) -> float:
    """The mean per-class recall of an export's logits; the export itself records the
    micro accuracy."""
    with np.load(export_dir / "reference.npz") as reference:
        predicted = reference["logits_clean"].argmax(axis=1)
        labels = reference["labels"]
    recalls = [np.mean(predicted[labels == c] == c) for c in np.unique(labels)]
    return float(np.mean(recalls))


def _run(command: list[str], log: Path) -> None:
    """:raises subprocess.CalledProcessError: when it fails; *log* holds why."""
    with log.open("w") as out:
        subprocess.run(
            command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True
        )


def export_epoch(run_dir: Path, checkpoint: Path, work: Path) -> dict:
    """Export one checkpoint on the validation rows under *work*, unless done before."""
    out = work / checkpoint.stem
    if not (out / "export" / "export.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        export = [sys.executable, "scripts/synthesis/export_hgq.py"]
        export += ["--run-dir", str(run_dir), "--ckpt", str(checkpoint)]
        _run(
            export + ["--out", str(out / "export"), "--split", "fold"],
            out / "export.log",
        )
    meta = json.loads((out / "export" / "export.json").read_text())
    row = {"epoch": checkpoint.stem, "checkpoint": str(checkpoint)}
    row |= {"accuracy": meta["accuracy"], "ebops": meta["ebops"]}
    return row | {"accuracy_macro": macro_accuracy(out / "export")}


def estimate_epoch(row: dict, work: Path) -> dict:
    """Add alkaid's estimate of an exported epoch, unless done before."""
    out = work / row["epoch"]
    if not (out / "estimate" / "estimate.json").exists():
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
    return row | {key: found[key] for key in ("cycles", "latency_ns", "lut", "dsp")}


def _front(args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    """The rows of every kept epoch and the pick of the rule the command line names."""
    checkpoints = sorted((args.run_dir / "checkpoints").glob("epoch_*.ckpt"))
    rows = [export_epoch(args.run_dir, c, args.work) for c in checkpoints]
    estimate = functools.partial(estimate_epoch, work=args.work)
    if args.max_ebops is None:
        rows = [estimate(row) for row in rows]
        return rows, pick(rows, args.budget_ns)
    skip = set(args.skip)
    return matched_front(rows, args.max_ebops, args.budget_ns, estimate, skip)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="an hgq run")
    parser.add_argument("--work", type=Path, required=True, help="bulk outputs")
    parser.add_argument("--record", type=Path, required=True, help="csv and pick")
    parser.add_argument("--budget-ns", type=float, default=100.0)
    parser.add_argument("--max-ebops", type=int, help="Block 7's target; None: Block 4")
    parser.add_argument("--skip", nargs="*", default=[], help="epochs Vivado failed")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.record.mkdir(parents=True, exist_ok=True)
    rows, chosen = _front(args)
    with (args.record / "front.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if chosen is None:
        print(f"no epoch of {args.run_dir} meets the rule", file=sys.stderr)
        return 1
    (args.record / "pick.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
