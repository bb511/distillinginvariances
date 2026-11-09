#!/usr/bin/env python3
"""Pick a jets or top-tagging student under the latency budget by walking down the
ranking of its search (03_student.md, 9 Nov 2026).

The candidates are the first ``--budget`` (200) COMPLETE trials of the study, ranked by
the main run's val/acc_best, zero-scored trials left out, at most ``--top`` of them. Each
in turn:

1. its trial directory is found among the sweep directories ``--trials`` (a glob) by
   matching the trial's parameters against each directory's ``.hydra/overrides.yaml``;
2. its HGQ2 port is trained from that trial's float checkpoint (``algorithm.init_from``)
   for ``EPOCHS`` epochs, beta rising geometrically from ``BETA[0]`` to ``BETA[1]``, at
   the trial's learning rate and weight decay and the students' batch of 1024;
3. ``export_hgq.py --split fold`` traces the port on the training split and exports it
   on the run's validation rows, so that nothing reads the test split;
4. ``synthesise.py --estimate-only`` writes alkaid's estimate at 5 ns.

The first candidate whose estimated latency is within ``--budget-ns`` is the pick. Each
candidate's row is appended to ``<record>/walkdown.csv`` and the pick written to
``<record>/pick.json``; a candidate whose ``estimate.json`` exists is not run again, so
a stopped walk-down resumes. It runs on the pod whose ``/scratch`` holds the trial
directories::

    python3 scripts/search/walkdown.py student_jets_pt_eta_phi_canonical \\
        --experiment student/jets_pt_eta_phi \\
        --trials '/scratch/distilling_symmetries_bulk/searches/student_jets_abs_s*' \\
        --work /scratch/distilling_symmetries_bulk/walkdown/jets_abs \\
        --record logs/walkdown/jets_abs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studies import DEFAULT_DB, TRIAL_BUDGET, complete_trials, stack_of  # noqa: E402

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

REPO = Path(__file__).resolve().parents[2]
# The short HGQ2 training of each candidate (user, 9 Nov 2026).
EPOCHS = 50
BETA = (1e-8, 1e-6)
BATCH_SIZE = 1024
FIELDS = (
    "rank",
    "trial",
    "val_acc_best",
    "stack",
    "lr",
    "weight_decay",
    "cycles",
    "latency_ns",
    "lut",
    "dsp",
    "ebops",
    "method0",
    "hard_dc",
    "within",
)


def ranking(complete: list[FrozenTrial], budget: int, top: int) -> list[FrozenTrial]:
    """The first *budget* trials by number, ranked by objective (the earliest first on a
    tie), the zero-scored over-budget stacks left out, at most *top* of them."""
    window = sorted(complete, key=lambda t: t.number)[:budget]
    ranked = sorted(window, key=lambda t: (-t.value, t.number))
    return [t for t in ranked if t.value > 0][:top]


def _overrides(path: Path) -> dict[str, str]:
    """A trial directory's overrides as key and value strings."""
    return dict(item.split("=", 1) for item in yaml.safe_load(path.read_text()))


def _matches(params: dict, overrides: dict[str, str]) -> bool:
    """Whether every parameter of a trial is the override of the same key; floats are
    compared to 1e-12 relative, since their decimal forms may differ."""
    for key, value in params.items():
        given = overrides.get(key)
        if given is None:
            return False
        if isinstance(value, float):
            if not math.isclose(float(given), value, rel_tol=1e-12):
                return False
        elif str(value).replace(" ", "") != given.replace(" ", ""):
            return False
    return True


def trial_dir(trial: FrozenTrial, pattern: str) -> Path:
    """The one sweep directory whose overrides are the trial's parameters.

    :raises LookupError: when none or several match.
    """
    hits = [
        path.parent.parent
        for path in map(Path, sorted(glob(f"{pattern}/*/.hydra/overrides.yaml")))
        if _matches(trial.params, _overrides(path))
    ]
    if len(hits) != 1:
        raise LookupError(f"trial {trial.number}: {len(hits)} directories match")
    return hits[0]


def float_checkpoint(directory: Path) -> Path:
    """The one checkpoint a search trial kept, its best epoch.

    The trial's manifest names it by an absolute path on the pod that ran the search, so
    it is found here.

    :raises LookupError: when there is not exactly one.
    """
    found = sorted((directory / "checkpoints").glob("*.ckpt"))
    if len(found) != 1:
        raise LookupError(f"{directory}: {len(found)} checkpoints")
    return found[0]


def stage(candidates: list[FrozenTrial], pattern: str, dest: Path) -> int:
    """Copy each candidate's overrides and checkpoint to *dest*, keeping the last two
    components of its directory, so that ``<dest>/<sweep dir name>`` serves as
    ``--trials`` on another pod.

    :return: the bytes copied.
    """
    total = 0
    for trial in candidates:
        source = trial_dir(trial, pattern)
        target = dest / source.parent.name / source.name
        checkpoint = float_checkpoint(source)
        for path, sub in (
            (source / ".hydra" / "overrides.yaml", ".hydra"),
            (checkpoint, "checkpoints"),
        ):
            (target / sub).mkdir(parents=True, exist_ok=True)
            total += shutil.copy2(path, target / sub / path.name).stat().st_size
    return total


def hgq_command(experiment: str, trial: FrozenTrial, source: Path, run: Path) -> list:
    """The short HGQ2 training of a candidate from its float checkpoint."""
    params = trial.params
    stack = ",".join(map(str, stack_of({k: str(v) for k, v in params.items()})))
    schedule = f"[[0,{BETA[0]},log],[{EPOCHS},{BETA[1]},constant]]"
    command = [sys.executable, "src/train.py", f"experiment={experiment}", "method=hgq"]
    command += [f"algorithm.init_from='{source}'", f"student_net.hidden=[{stack}]"]
    command += [
        f"algorithm.scheduler.initial_learning_rate={params['algorithm.optimizer.lr']}"
    ]
    command += [
        f"+algorithm.optimizer.weight_decay={params['algorithm.optimizer.weight_decay']}"
    ]
    command += [f"algorithm.beta_schedule={schedule}", f"trainer.max_epochs={EPOCHS}"]
    command += [f"data.batch_size={BATCH_SIZE}", "logger=none", f"hydra.run.dir={run}"]
    return command + [f"hp_tag=walkdown_t{trial.number}"]


def _run(command: list, log: Path) -> None:
    """Run *command* from the repository root, its output into *log*.

    :raises subprocess.CalledProcessError: when it fails; the log holds why.
    """
    with log.open("w") as out:
        subprocess.run(
            command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True
        )


def estimate(args: argparse.Namespace, trial: FrozenTrial) -> dict:
    """Train, export and estimate one candidate under ``--work``; reuse a finished one.

    ``--overrides`` are appended to the HGQ2 training, for a smoke test at
    ``debug=limit``.
    """
    experiment, pattern, work = args.experiment, args.trials, args.work
    out = work / f"t{trial.number}"
    if (out / "estimate" / "estimate.json").exists():
        return json.loads((out / "estimate" / "estimate.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    source = float_checkpoint(trial_dir(trial, pattern))
    training = hgq_command(experiment, trial, source, out / "run") + args.overrides
    _run(training, out / "train.log")
    export = [sys.executable, "scripts/synthesis/export_hgq.py", "--run-dir"]
    _run(
        export + [str(out / "run"), "--out", str(out / "export"), "--split", "fold"],
        out / "export.log",
    )
    synthesis = [sys.executable, "scripts/synthesis/synthesise.py", "--estimate-only"]
    _run(
        synthesis
        + ["--export-dir", str(out / "export"), "--out", str(out / "estimate")],
        out / "estimate.log",
    )
    return json.loads((out / "estimate" / "estimate.json").read_text())


def row_of(rank: int, trial: FrozenTrial, found: dict, budget_ns: float) -> dict:
    params = trial.params
    stack = stack_of({k: str(v) for k, v in params.items()})
    row = {"rank": rank, "trial": trial.number, "val_acc_best": trial.value}
    row |= {"stack": stack, "lr": params["algorithm.optimizer.lr"]}
    row |= {"weight_decay": params["algorithm.optimizer.weight_decay"]}
    row |= {key: found.get(key) for key in FIELDS if key in found}
    return row | {"within": found["latency_ns"] <= budget_ns}


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="the study name, regime included")
    parser.add_argument("--experiment", help="the study's experiment")
    parser.add_argument("--trials", required=True, help="glob of its sweep directories")
    parser.add_argument("--work", type=Path, help="bulk outputs")
    parser.add_argument("--record", type=Path, help="csv and pick")
    parser.add_argument(
        "--stage", type=Path, help="copy the candidates there, and stop"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--budget", type=int, default=TRIAL_BUDGET)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--budget-ns", type=float, default=100.0)
    parser.add_argument("--overrides", nargs="*", default=[], help="for the training")
    args = parser.parse_args()
    if args.stage is None and None in (args.experiment, args.work, args.record):
        parser.error("--experiment, --work and --record are needed without --stage")
    return args


def main() -> int:
    args = _parse_args()
    candidates = ranking(complete_trials(args.db, args.study), args.budget, args.top)
    if args.stage is not None:
        total = stage(candidates, args.trials, args.stage)
        print(f"staged {len(candidates)} candidates, {total} bytes, to {args.stage}")
        return 0
    args.record.mkdir(parents=True, exist_ok=True)
    for rank, trial in enumerate(candidates, 1):
        found = estimate(args, trial)
        row = row_of(rank, trial, found, args.budget_ns)
        _append(args.record / "walkdown.csv", row)
        print(json.dumps(row), flush=True)
        if row["within"]:
            (args.record / "pick.json").write_text(
                json.dumps(row | {"study": args.study}, indent=2)
            )
            return 0
    print(
        f"none of {len(candidates)} candidates within {args.budget_ns} ns",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
