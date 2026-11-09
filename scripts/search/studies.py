#!/usr/bin/env python3
"""Read the selected point of an Optuna search; write a student search's into its net
config.

The selection is the sibling's budget rule: the best of the first ``--budget`` (200)
COMPLETE trials by trial number, the earliest on a tie. Drivers sharing a study overshoot
its target, and searches compare fairly only on an equal number of trials. A stack over
``max_macs`` scores 0 as a COMPLETE trial, so it takes its place in the window::

    python3 scripts/search/studies.py best student_jets_pt_eta_phi_canonical
    python3 scripts/search/studies.py write-net student_jets_pt_eta_phi_canonical \\
        configs/net/mlp_jets_abs.yaml

``best`` prints the trial, its objective, the student's stack when the search drew one,
and every other parameter as a hydra override for a catalogue. ``write-net`` replaces the
``hidden`` line of a net config by the stack (the first ``student_net.depth`` of the
widths ``width_1`` to ``width_7``, or the chosen ``student_net.hidden``) under a comment
naming the study, the trial, its objective, learning rate and weight decay.

Optuna is imported inside the reader, so ``--help`` and the tests of the rule run without
it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

REPO = Path(__file__).resolve().parents[2]
TRIAL_BUDGET = 200
DEFAULT_DB = REPO / "logs" / "optuna" / "student.db"
# The parameters that set the student's architecture rather than its training.
ARCHITECTURE = re.compile(r"^(width_\d+|student_net\.(depth|hidden))$")
COMMENT = "# Selected by scripts/search/studies.py:"

# (trial number, objective value, params) of the selected trial.
Trial = tuple[int, float, dict[str, str]]


def quote(value: str) -> str:
    """Single-quote a value the shell would otherwise treat as a glob or a brace."""
    return f"'{value}'" if any(c in value for c in "[]{},") else value


def best_within_budget(complete: list[FrozenTrial], budget: int) -> Trial | None:
    """The best of the first *budget* trials by trial number, the earliest on a tie;
    every search maximises."""
    window = sorted(complete, key=lambda t: t.number)[:budget]
    if not window:
        return None
    trial = max(window, key=lambda t: (t.value, -t.number))
    params = {k: str(v) for k, v in trial.params.items()}
    return trial.number, float(trial.value), params


def complete_trials(db: Path, study: str) -> list[FrozenTrial]:
    """The COMPLETE trials of *study*, none when the database or the study is absent."""
    if not db.exists():
        return []
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        loaded = optuna.load_study(study_name=study, storage=f"sqlite:///{db}")
    except KeyError:
        return []
    states = (optuna.trial.TrialState.COMPLETE,)
    return loaded.get_trials(deepcopy=False, states=states)


def stack_of(params: dict[str, str]) -> list[int] | None:
    """The hidden widths a student search drew, None for a search of no student."""
    if "student_net.hidden" in params:
        return [int(w) for w in params["student_net.hidden"].strip("[]").split(",")]
    if "student_net.depth" not in params:
        return None
    depth = int(params["student_net.depth"])
    return [int(params[f"width_{k}"]) for k in range(1, depth + 1)]


def overrides_of(params: dict[str, str]) -> list[str]:
    """The training parameters as hydra override tokens, sorted by key; Optuna's names
    are the override paths of ``hydra.sweeper.params``."""
    keys = sorted(k for k in params if not ARCHITECTURE.match(k))
    return [f"{k}={quote(params[k])}" for k in keys]


def provenance(study: str, trial: Trial, budget: int) -> str:
    """The comment line that heads a written stack."""
    number, value, params = trial
    lr = params.get("algorithm.optimizer.lr")
    decay = params.get("algorithm.optimizer.weight_decay")
    return (
        f"{COMMENT} study {study}, trial {number} (best of the first {budget} complete),"
        f" val/acc_best {value:.5f}, lr {lr}, weight_decay {decay}."
    )


def write_net(path: Path, study: str, trial: Trial, budget: int) -> str:
    """Replace the ``hidden`` line of a net config, and a previous selection comment, by
    the trial's stack under its provenance; return the new text.

    :raises ValueError: when the trial drew no stack or the config has no ``hidden``.
    """
    stack = stack_of(trial[2])
    text = path.read_text()
    if stack is None or not re.search(r"^hidden:", text, re.M):
        raise ValueError(f"no stack in the trial, or no 'hidden:' line in {path}")
    text = re.sub(rf"^{re.escape(COMMENT)}.*\n", "", text, flags=re.M)
    line = f"{provenance(study, trial, budget)}\nhidden: [{', '.join(map(str, stack))}]"
    return re.sub(r"^hidden:.*$", lambda _: line, text, count=1, flags=re.M)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("best", "write-net"))
    parser.add_argument("study", help="the study name, regime included")
    parser.add_argument("net", nargs="?", type=Path, help="net config for write-net")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--budget", type=int, default=TRIAL_BUDGET)
    return parser.parse_args()


def _show(study: str, trial: Trial, n_complete: int, budget: int) -> None:
    number, value, params = trial
    print(f"{study}: trial {number}, objective {value:.5f}")
    print(f"window: first {min(n_complete, budget)} of {n_complete} complete trials")
    print(f"stack: {stack_of(params)}")
    print("overrides: " + " ".join(overrides_of(params)))


def main() -> int:
    args = _parse_args()
    complete = complete_trials(args.db, args.study)
    trial = best_within_budget(complete, args.budget)
    if trial is None:
        raise SystemExit(f"no complete trial of {args.study} in {args.db}")
    if args.command == "best":
        _show(args.study, trial, len(complete), args.budget)
        return 0
    if args.net is None:
        raise SystemExit("write-net needs the net config to write")
    args.net.write_text(write_net(args.net, args.study, trial, args.budget))
    print(f"wrote {stack_of(trial[2])} to {args.net}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
