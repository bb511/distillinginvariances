import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import ROOT, compose_root


def _walkdown():
    path = ROOT / "scripts" / "search" / "walkdown.py"
    spec = importlib.util.spec_from_file_location("walkdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(number: int, value: float, **params) -> SimpleNamespace:
    return SimpleNamespace(number=number, value=value, params=params)


JETS = {
    "student_net.depth": 3,
    **{f"width_{k}": w for k, w in enumerate([64, 44, 32, 128, 88, 32, 32], 1)},
    "algorithm.optimizer.lr": 0.0012345678901234,
    "algorithm.optimizer.weight_decay": 1e-06,
}


def test_the_ranking_walks_the_budget_window_down_without_zeros() -> None:
    walkdown = _walkdown()
    trials = [
        _trial(0, 0.0),
        _trial(1, 0.70),
        _trial(2, 0.75),
        _trial(3, 0.70),
        _trial(4, 0.90),
    ]

    ranked = walkdown.ranking(trials, budget=4, top=10)

    assert [t.number for t in ranked] == [2, 1, 3]
    assert [t.number for t in walkdown.ranking(trials, budget=5, top=2)] == [4, 2]


def test_a_trial_finds_its_own_sweep_directory(tmp_path: Path) -> None:
    walkdown = _walkdown()
    for job, lr in ((0, "0.0012345678901234"), (1, "0.002")):
        hydra = tmp_path / "student_jets_abs_s1234" / str(job) / ".hydra"
        hydra.mkdir(parents=True)
        lines = [f"- {k}={v}" for k, v in JETS.items() if k != "algorithm.optimizer.lr"]
        lines += [
            f"- algorithm.optimizer.lr={lr}",
            "- experiment=student/jets_pt_eta_phi",
        ]
        (hydra / "overrides.yaml").write_text("\n".join(lines) + "\n")

    found = walkdown.trial_dir(_trial(7, 0.8, **JETS), str(tmp_path / "student_*"))

    assert found == tmp_path / "student_jets_abs_s1234" / "0"
    with pytest.raises(LookupError):
        walkdown.trial_dir(
            _trial(8, 0.8, **(JETS | {"width_1": 32})), str(tmp_path / "*")
        )


def test_the_short_hgq2_training_composes(tmp_path: Path) -> None:
    walkdown = _walkdown()
    command = walkdown.hgq_command(
        "student/jets_pt_eta_phi", _trial(7, 0.8, **JETS), tmp_path, tmp_path / "run"
    )

    cfg = compose_root("train.yaml", *command[2:])

    assert cfg.stage == "hgq" and cfg.algorithm.init_from == str(tmp_path)
    assert list(cfg.student_net.hidden) == [64, 44, 32]
    assert cfg.algorithm.beta_schedule[0][1] == pytest.approx(1e-8)
    assert cfg.algorithm.beta_schedule[1][:2] == [50, pytest.approx(1e-6)]
    assert cfg.algorithm.optimizer.weight_decay == pytest.approx(1e-6)
    assert cfg.data.batch_size == 1024 and cfg.trainer.max_epochs == 50


def test_staging_copies_what_another_pod_walks_down(tmp_path: Path) -> None:
    walkdown = _walkdown()
    source = tmp_path / "scratch" / "student_jets_abs_s1234" / "3"
    (source / ".hydra").mkdir(parents=True)
    (source / "checkpoints").mkdir()
    overrides = [f"- {k}={v}" for k, v in JETS.items()]
    (source / ".hydra" / "overrides.yaml").write_text("\n".join(overrides) + "\n")
    (source / "checkpoints" / "epoch_041.ckpt").write_bytes(b"weights")
    trial = _trial(7, 0.8, **JETS)

    total = walkdown.stage(
        [trial], str(tmp_path / "scratch" / "*"), tmp_path / "shared"
    )
    staged = walkdown.trial_dir(trial, str(tmp_path / "shared" / "student_*"))

    assert staged == tmp_path / "shared" / "student_jets_abs_s1234" / "3"
    assert walkdown.float_checkpoint(staged).read_bytes() == b"weights"
    assert total == len(b"weights") + len("\n".join(overrides) + "\n")


def test_the_front_pick_takes_the_most_accurate_epoch_within_the_budget() -> None:
    path = ROOT / "scripts" / "synthesis" / "front_pick.py"
    spec = importlib.util.spec_from_file_location("front_pick", path)
    front = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(front)
    rows = [
        {"epoch": "epoch_0010", "accuracy": 0.80, "ebops": 900, "latency_ns": 120.0},
        {"epoch": "epoch_0020", "accuracy": 0.78, "ebops": 700, "latency_ns": 95.0},
        {"epoch": "epoch_0030", "accuracy": 0.78, "ebops": 600, "latency_ns": 90.0},
        {"epoch": "epoch_0040", "accuracy": 0.70, "ebops": 300, "latency_ns": 60.0},
    ]

    assert front.pick(rows, 100.0)["epoch"] == "epoch_0030"
    assert front.pick(rows, 50.0) is None


def _front_pick():
    path = ROOT / "scripts" / "synthesis" / "front_pick.py"
    spec = importlib.util.spec_from_file_location("front_pick", path)
    front = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(front)
    return front


def test_the_matched_pick_takes_the_most_accurate_epoch_at_the_target() -> None:
    front = _front_pick()
    rows = [
        {"epoch": "epoch_0100", "accuracy_macro": 0.90, "ebops": 200},
        {"epoch": "epoch_0200", "accuracy_macro": 0.80, "ebops": 120},
        {"epoch": "epoch_0300", "accuracy_macro": 0.80, "ebops": 110},
        {"epoch": "epoch_0400", "accuracy_macro": 0.79, "ebops": 100},
        {"epoch": "epoch_0500", "accuracy_macro": 0.70, "ebops": 90},
    ]
    latency = {"epoch_0300": 120.0}
    estimated = []

    def estimate(row: dict) -> dict:
        estimated.append(row["epoch"])
        return row | {"latency_ns": latency.get(row["epoch"], 60.0)}

    def chosen(max_ebops: int, skip: set[str]) -> str | None:
        found = front.matched_front(rows, max_ebops, 100.0, estimate, skip)[1]
        return None if found is None else found["epoch"]

    assert chosen(120, set()) == "epoch_0200"
    assert chosen(115, set()) == "epoch_0400"
    assert chosen(115, {"epoch_0400"}) == "epoch_0500"
    assert chosen(80, set()) is None
    assert "epoch_0100" not in estimated


def test_the_matched_record_keeps_the_candidates_that_gave_way() -> None:
    front = _front_pick()
    rows = [
        {"epoch": "epoch_0100", "accuracy_macro": 0.90, "ebops": 200},
        {"epoch": "epoch_0300", "accuracy_macro": 0.80, "ebops": 110},
        {"epoch": "epoch_0400", "accuracy_macro": 0.79, "ebops": 100},
        {"epoch": "epoch_0500", "accuracy_macro": 0.70, "ebops": 90},
    ]
    latency = {"epoch_0300": 120.0, "epoch_0500": 60.0}

    def estimate(row: dict) -> dict:
        return row | {"latency_ns": latency[row["epoch"]], "cycles": 1}

    record, found = front.matched_front(rows, 115, 100.0, estimate, {"epoch_0400"})
    by_epoch = {row["epoch"]: row for row in record}

    assert found["epoch"] == "epoch_0500"
    assert [row["epoch"] for row in record] == [row["epoch"] for row in rows]
    assert by_epoch["epoch_0300"]["latency_ns"] == 120.0
    assert by_epoch["epoch_0500"]["latency_ns"] == 60.0
    assert "latency_ns" not in by_epoch["epoch_0100"]
    assert by_epoch["epoch_0400"]["skipped"] is True
    assert by_epoch["epoch_0300"]["skipped"] is False


def test_the_macro_accuracy_is_the_mean_per_class_recall(tmp_path: Path) -> None:
    logits = np.eye(3)[[0, 0, 0, 1, 2, 2]]
    labels = np.array([0, 0, 1, 1, 2, 2])
    np.savez(tmp_path / "reference.npz", logits_clean=logits, labels=labels)

    assert _front_pick().macro_accuracy(tmp_path) == pytest.approx((1 + 0.5 + 1) / 3)
