import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from src.algorithms.components.mlp import MLPStudent
from src.algorithms.utils.budget import feasible_stacks, linear_macs, stack_macs
from src.train import train
from tests.conftest import ROOT, compose_root, point_into

WIDTHS = [32, 44, 64, 88, 128]
DEPTHS = range(3, 8)


def _studies():
    path = ROOT / "scripts" / "search" / "studies.py"
    spec = importlib.util.spec_from_file_location("studies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _composed(*overrides: str):
    return compose_root(
        "train.yaml", "method=baseline", "trainer=cpu", "logger=none", *overrides
    )


def test_the_budget_prices_the_published_stacks() -> None:
    # The 2024 student on 96 inputs and the sibling's MLP on 150.
    assert stack_macs([96, 84, 88, 32, 32, 44, 32, 44, 5]) == 23_740
    assert stack_macs([150, 128, 64, 32, 5]) == 29_600
    assert linear_macs(MLPStudent(150, (128, 64, 32), 5)) == 29_600


def test_the_feasible_sets_match_the_plan_and_the_mnist_choice() -> None:
    jets = feasible_stacks(150, 5, WIDTHS, DEPTHS, 30_000)
    mnist = feasible_stacks(784, 10, WIDTHS, DEPTHS, 30_000)
    config = ROOT / "configs" / "hparams_search" / "student" / "mnist_optuna.yaml"
    line = re.search(r"student_net\.hidden: choice\((.*)\)", config.read_text())
    choice = [
        tuple(int(w) for w in s.split(","))
        for s in re.findall(r"\[([\d,]+)\]", line.group(1))
    ]

    assert len(jets) == 24_674
    assert len(mnist) == 16 and choice == mnist


def test_depth_and_width_factor_shape_the_student() -> None:
    hidden = (64, 32, 44, 88, 128, 32, 32)
    net = MLPStudent(150, hidden, 5, depth=3)
    wide = MLPStudent(150, hidden, 5, depth=3, width_factor=2)

    logits, activations = net.forward_with_hidden(torch.randn(2, 50, 3))
    assert [a.shape[1] for a in activations] == [64, 32, 44]
    assert logits.shape == (2, 5)
    assert [layer.out_features for layer in wide.hidden] == [128, 64, 88]
    assert linear_macs(net) == stack_macs([150, 64, 32, 44, 5])


def test_an_over_budget_student_scores_zero_without_building_data(
    tmp_path: Path,
) -> None:
    cfg = _composed(
        "experiment=student/jets_pt_etarel_phirel",
        "max_macs=1000",
        "optimized_metric=val/acc_best",
    )
    HydraConfig().set_config(point_into(cfg, tmp_path))
    with open_dict(cfg):
        # Building the data would raise.
        cfg.data._target_ = "tests.no_such_module.NoSuchDataModule"

    metrics, objects = train(cfg)

    assert float(metrics["val/acc_best"]) == 0.0 and objects == {}


def test_the_variants_follow_the_searched_stacks() -> None:
    base = instantiate(_composed("experiment=student/jets_pt_eta_phi").student_net)
    wide = instantiate(
        _composed(
            "experiment=student/jets_pt_eta_phi", "net@student_net=mlp_jets_abs_wide"
        ).student_net
    )
    p4 = instantiate(
        _composed(
            "experiment=student/jets_pt_eta_phi", "net@student_net=mlp_jets_p4"
        ).student_net
    )

    widths = [layer.out_features for layer in base.hidden]
    assert [layer.out_features for layer in wide.hidden] == [2 * w for w in widths]
    assert [layer.out_features for layer in p4.hidden] == widths
    assert p4.hidden[0].in_features == 200
    assert p4.required_features == ["px", "py", "pz", "e"]
    assert wide.required_features == base.required_features == ["pt", "eta", "phi"]


def test_the_student_searches_compose_into_their_studies() -> None:
    pytest.importorskip("hydra_plugins.hydra_optuna_sweeper")
    jets = _composed(
        "experiment=student/jets_pt_eta_phi",
        "hparams_search=student/jets_optuna",
        "width_1=88",
        "width_2=44",
        "student_net.depth=3",
    )
    mnist = _composed(
        "experiment=student/mnist",
        "hparams_search=student/mnist_optuna",
        "student_net.hidden=[32,44,32]",
    )

    jets_net, mnist_net = (instantiate(c.algorithm.net) for c in (jets, mnist))
    assert [layer.out_features for layer in jets_net.hidden] == [88, 44, 32]
    assert [layer.out_features for layer in mnist_net.hidden] == [32, 44, 32]
    assert jets.hydra.sweeper.study_name == "student_jets_pt_eta_phi_canonical"
    assert mnist.hydra.sweeper.study_name == "student_mnist_canonical"
    assert jets.max_macs == mnist.max_macs == 30_000
    assert jets.optimized_metric == "val/acc_best"
    assert jets.callbacks.model_checkpoint.save_last is False


def test_the_top_tagging_search_composes_on_300_inputs() -> None:
    pytest.importorskip("hydra_plugins.hydra_optuna_sweeper")
    cfg = _composed(
        "experiment=student/top_tagging_pt_eta_phi",
        "hparams_search=student/jets_optuna",
        "width_1=64",
        "student_net.depth=3",
    )

    net = instantiate(cfg.algorithm.net)
    assert net.hidden[0].in_features == 300 and net.output.out_features == 2
    assert cfg.hydra.sweeper.study_name == "student_top_tagging_pt_eta_phi_canonical"
    assert len(feasible_stacks(300, 2, WIDTHS, DEPTHS, 30_000)) == 10_132


def _trial(number: int, value: float, **params) -> SimpleNamespace:
    return SimpleNamespace(number=number, value=value, params=params)


def test_the_budget_rule_takes_the_best_of_the_first_trials_first_on_a_tie() -> None:
    studies = _studies()
    trials = [_trial(3, 0.7), _trial(0, 0.5), _trial(1, 0.7), _trial(2, 0.9)]

    assert studies.best_within_budget(trials, 4)[:2] == (2, 0.9)
    assert studies.best_within_budget(trials, 2)[:2] == (1, 0.7)
    assert studies.best_within_budget(trials[:3], 4)[:2] == (1, 0.7)
    assert studies.best_within_budget([], 2) is None


def test_the_selected_stack_and_overrides_are_read_off_the_params() -> None:
    studies = _studies()
    jets = {"student_net.depth": "3", "algorithm.optimizer.lr": "0.002"}
    jets |= {f"width_{k}": str(w) for k, w in enumerate(WIDTHS + [32, 32], 1)}
    mnist = {
        "student_net.hidden": "[32,44,32]",
        "algorithm.optimizer.weight_decay": "0",
    }

    assert studies.stack_of(jets) == [32, 44, 64]
    assert studies.stack_of(mnist) == [32, 44, 32]
    assert studies.stack_of({"algorithm.optimizer.lr": "0.1"}) is None
    assert studies.overrides_of(jets) == ["algorithm.optimizer.lr=0.002"]


def test_writing_a_net_config_twice_leaves_one_selection(tmp_path: Path) -> None:
    studies = _studies()
    net = tmp_path / "mlp.yaml"
    net.write_text("# head\n_target_: x\nhidden: [64, 32]\noutput_dim: 5\n")
    trial = (7, 0.75, {"student_net.hidden": "[32,32,32]"})

    net.write_text(studies.write_net(net, "s_canonical", trial, 200))
    text = studies.write_net(net, "s_canonical", trial, 200)

    assert text.count(studies.COMMENT) == 1
    assert "hidden: [32, 32, 32]\noutput_dim: 5" in text
    assert text.startswith("# head\n_target_: x\n# Selected")


def _teacher(experiment: str, *overrides: str):
    base = ("trainer=cpu", "logger=none", f"experiment={experiment}")
    return compose_root("train.yaml", *base, *overrides)


def test_the_cnn_search_composes_into_the_teacher_study() -> None:
    pytest.importorskip("hydra_plugins.hydra_optuna_sweeper")
    cfg = _teacher(
        "mnist_cnn",
        "hparams_search=teacher/mnist_cnn_optuna",
        "teacher_net.phi_channels=[16,32,64]",
        "teacher_net.kernel_size=5",
        "teacher_net.rho_layers=[128,64]",
        "teacher_lr=0.003",
        "teacher_optimizer.weight_decay=1e-05",
    )
    net, sweeper = instantiate(cfg.algorithm.net), cfg.hydra.sweeper

    assert all(OmegaConf.select(cfg, key) is not None for key in sweeper.params)
    assert net.hint_dim == 64 and net.phi[0].kernel_size == (5, 5)
    assert (net.rho[0].out_features, net.rho[2].out_features) == (128, 64)
    assert cfg.algorithm.optimizer.lr == 0.003 and cfg.hp_tag == "lr0.003"
    assert cfg.algorithm.optimizer.weight_decay == 1e-5
    assert sweeper.study_name == "cnn_circular_canonical"
    assert sweeper.storage.startswith("sqlite:///logs/optuna/teacher.db")
    assert cfg.optimized_metric == "val/acc_best" and cfg.data.batch_size == 128


def test_the_zero_padded_control_takes_the_circular_recipe() -> None:
    circular, zeros = (_teacher(e) for e in ("mnist_cnn", "mnist_cnn_zeros"))
    recipe = ("teacher_net", "teacher_optimizer", "teacher_scheduler", "data")

    for key in recipe:
        a, b = (OmegaConf.to_container(c[key], resolve=True) for c in (circular, zeros))
        if key == "teacher_net":
            assert b.pop("padding_mode") == "zeros"
            assert a.pop("padding_mode") == "circular"
        assert a == b
    assert circular.teacher_lr == zeros.teacher_lr
