import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
