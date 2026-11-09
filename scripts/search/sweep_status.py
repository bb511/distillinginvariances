#!/usr/bin/env python3
"""Status of every Optuna search of this repository, and the commands to finish it.

For each block of a ``scripts/<dataset>/run*_search.sh`` catalogue: the COMPLETE trials of
its study against the ``hydra.sweeper.n_trials`` the block asks for, and whether a driver
is still alive to produce the rest. Prints a table, then the launch commands::

    python3 scripts/search/sweep_status.py              # table and commands
    python3 scripts/search/sweep_status.py --commands   # commands only

The study name is not on the command line: it is the experiment's ``search_study`` and
the block's regime, as ``configs/hparams_search/student/*.yaml`` joins them. Liveness
comes from the driver logs, not from the trial counts, since a dead driver leaves its last
trial marked RUNNING for ever. One study may be swept by several drivers at once, each
tagged into a log of its own (``scripts/cluster/submit_search_ngt.sh``): every worker's
log is read, the study counts as alive while any one of them is, and the ``workers``
column says how many were found.
"""

import argparse
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAIL = 15
ERR = re.compile(
    r"Traceback \(most recent call last\)"
    r"|HYDRA_FULL_ERROR"
    r"|mismatched input"
    r"|LexerNoViableAltException"
    r"|optuna\.exceptions\."
    r"|^[A-Za-z_.]*(Error|Exception):"
    r"|^\S+\.py: error:",
    re.M,
)
FINISHED = re.compile(r"\[HYDRA\]\s+(Best|Value)", re.M)
HDR = "%-40s %-34s %6s %8s %6s %7s  %-8s %s"


def field(block: str, key: str) -> str | None:
    """The value of override *key* (a regex) in a flattened block, quotes stripped."""
    match = re.search(rf"(?:^|\s){key}=(\S+)", block)
    return match.group(1).strip("\"'") if match else None


def study_of(block: str) -> str:
    """The study a block sweeps: its experiment's ``search_study`` and its regime.

    :raises ValueError: when the experiment config names no ``search_study``.
    """
    experiment = field(block, "experiment")
    text = (REPO / "configs" / "experiment" / f"{experiment}.yaml").read_text()
    match = re.search(r"^search_study:\s*(\S+)", text, re.M)
    if match is None:
        raise ValueError(f"configs/experiment/{experiment}.yaml sets no search_study")
    regime = field(block, r"data\.regime") or "canonical"
    return f"{match.group(1)}_{regime}"


def db_of(block: str) -> Path:
    """The sqlite file of a block's storage: its own override, else its config's."""
    url = field(block, r"hydra\.sweeper\.storage")
    if url is None:
        config = f"configs/hparams_search/{field(block, 'hparams_search')}.yaml"
        text = (REPO / config).read_text()
        url = re.search(r"^\s*storage:\s*(\S+)", text, re.M).group(1)
    return REPO / url.removeprefix("sqlite:///").split("?")[0]


def blocks(script: Path) -> list[dict]:
    """Index, label, study, database and trial target of each block of a catalogue; the
    label is the one ``submit_search.sh`` names its driver logs with."""
    parts = re.split(r"^# python3 src/train\.py", script.read_text(), flags=re.M)[1:]
    out = []
    for idx, part in enumerate(parts, 1):
        block = part.replace("\\\n", " ").replace("#", " ")
        regime = field(block, r"data\.regime") or "canonical"
        label = f"{field(block, 'experiment').split('/')[-1]}_{regime}"
        target = int(field(block, r"hydra\.sweeper\.n_trials") or 0)
        place = dict(idx=idx, label=label, script=str(script.relative_to(REPO)))
        out.append(place | dict(study=study_of(block), db=db_of(block), target=target))
    return out


def _tail_of(log: Path) -> str:
    """The last TAIL non-blank lines of a driver log: a Hydra config error ends several
    lines below its exception."""
    lines = [
        line for line in log.read_text(errors="replace").splitlines() if line.strip()
    ]
    return "\n".join(lines[-TAIL:])


def classify(log: Path) -> tuple[str, float]:
    """(state, age in minutes) of a driver, from the tail of its log."""
    age = (
        datetime.now() - datetime.fromtimestamp(os.path.getmtime(log))
    ).total_seconds()
    tail = _tail_of(log)
    if ERR.search(tail):
        return "DEAD", age / 60
    return ("FINISHED" if FINISHED.search(tail) else "alive"), age / 60


def classify_workers(logs: list[Path]) -> tuple[str, float | None]:
    """(state, age) over a study's workers: alive if any is, else the most recently
    touched."""
    if not logs:
        return "no-log", None
    states = (classify(log) for log in logs)
    return max(states, key=lambda state: (state[0] == "alive", -state[1]))


def complete_of(db: Path, study: str, writable: bool = False) -> int:
    """COMPLETE trials in a study, zero-scored over-budget ones included.

    A driver that crashed mid-transaction leaves a hot ``<db>-journal`` that a read-only
    open cannot roll back ('database is locked'); ``writable`` is for copies of such
    databases, never the live files.
    """
    if not db.exists():
        return 0
    uri = f"file:{db}" if writable else f"file:{db}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=60)) as con:
        query = "select study_id from studies where study_name=?"
        row = con.execute(query, (study,)).fetchone()
        if not row:
            return 0
        counted = "select count(*) from trials where study_id=? and state='COMPLETE'"
        return con.execute(counted, (row[0],)).fetchone()[0]


def status_of(done: int, target: int, state: str) -> str:
    if done >= target:
        return "DONE"
    if state == "alive":
        return "running"
    # A driver that exited cleanly below its target, or one that never ran or crashed.
    return "short" if state == "FINISHED" else "LAUNCH"


def row_of(block: dict, writable: bool) -> dict:
    """One table row: the block, its trial count and the state of its workers."""
    script = Path(block["script"])
    logdir = REPO / "logs" / "searches" / f"{script.parent.name}_{script.stem}"
    logs = sorted(logdir.glob(f"{block['idx']}_{block['label']}*.log"))
    done = complete_of(block["db"], block["study"], writable)
    state, age = classify_workers(logs)
    status = status_of(done, block["target"], state)
    return block | dict(
        done=done, state=state, age=age, workers=len(logs), status=status
    )


def launch_command(row: dict) -> str:
    """The submit_search.sh invocation that tops a study up to its target."""
    todo = row["target"] - row["done"]
    return (
        f"bash scripts/cluster/submit_search.sh --only {row['label']} {row['script']}"
        f" hydra.sweeper.n_trials={todo}"
    )


def print_table(rows: list[dict]) -> None:
    print(
        HDR
        % ("study", "catalogue", "done", "target", "wkrs", "age_min", "state", "status")
    )
    for r in rows:
        age = "-" if r["age"] is None else f"{r['age']:.0f}"
        cells = (r["study"], r["script"], r["done"], r["target"], r["workers"], age)
        print(HDR % (*cells, r["state"], r["status"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commands", action="store_true", help="commands only")
    parser.add_argument(
        "--writable", action="store_true", help="open copies of databases read-write"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scripts = sorted(REPO.glob("scripts/*/run*_search.sh"))
    rows = [row_of(b, args.writable) for script in scripts for b in blocks(script)]
    if not args.commands:
        print_table(rows)
        print()
    for row in rows:
        if row["status"] in ("LAUNCH", "short"):
            print(launch_command(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
