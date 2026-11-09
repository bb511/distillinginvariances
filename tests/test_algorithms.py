from pathlib import Path

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.classifier import Classifier
from src.algorithms.distillation import DistillationModule
from tests.conftest import RECORD_COLUMNS

ADAM = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}
DEEPSETS = {"_target_": "src.algorithms.components.deepsets.DeepSetsNet"}
LORENTZNET = {
    "_target_": "src.algorithms.components.lorentznet.LorentzNetGraph",
    "nconstituents": 8,
}
MLP = {
    "_target_": "src.algorithms.components.mlp.MLPStudent",
    "in_dim": 24,
    "required_features": ["pt", "etarel", "phirel"],
}


class StubDataModule:
    feature_names = RECORD_COLUMNS
    norm_scales = None


def _classifier(net: dict) -> Classifier:
    return Classifier(net=net, num_classes=5, input_shape=(8, 16), optimizer=ADAM)


def _student(
    objective: str, teacher_net: dict = DEEPSETS, **kwargs
) -> DistillationModule:
    return DistillationModule(
        net=MLP,
        num_classes=5,
        input_shape=(8, 16),
        optimizer=ADAM,
        teacher_net=teacher_net,
        objective=objective,
        **kwargs,
    )


def _batch(module, hint_dim: int = 32, n: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(n, *module.net_input_shape(), generator=generator)
    y = torch.eye(5)[torch.randint(0, 5, (n,), generator=generator)]
    return (
        x,
        y,
        torch.randn(n, 5, generator=generator),
        torch.randn(n, hint_dim, generator=generator),
    )


def _fit_one_batch(module, batch: tuple[torch.Tensor, ...], tmp_path: Path) -> Path:
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, DataLoader(TensorDataset(*batch), batch_size=len(batch[0])))
    trainer.save_checkpoint(tmp_path / "last.ckpt")
    return tmp_path / "last.ckpt"


@pytest.mark.parametrize(
    "net", [DEEPSETS, LORENTZNET, MLP], ids=["deepsets", "lorentznet", "mlp"]
)
def test_a_classifier_reduces_the_full_width_batch_to_its_columns(net: dict) -> None:
    module = _classifier(net)
    module.bind(StubDataModule())

    assert [RECORD_COLUMNS[i] for i in module.model_feature_indices] == [
        f"j1_{f}" for f in module.net.required_features
    ]
    assert module(torch.randn(4, 8, 16)).shape == (4, 5)
    assert (
        torch.isfinite(module.model_step(_batch(module)[:2])["loss"])
        and module.flops_total > 0
    )


@pytest.mark.parametrize("objective", ["hint", "kd", "relational"])
def test_one_optimiser_step_per_objective(objective: str) -> None:
    module = _student(objective, alpha=0.5, beta=0.25)
    loss = module.model_step(_batch(module))["loss"]

    loss.backward()
    module.configure_optimizers()["optimizer"].step()

    assert torch.isfinite(loss)
    assert module.tracks_accuracy == (objective != "hint")


def test_the_regressor_follows_the_hint_width() -> None:
    assert isinstance(_student("hint").regressor, torch.nn.Identity)  # 32 -> 32
    wide = _student("hint", teacher_net=LORENTZNET).regressor
    assert (wide.in_features, wide.out_features) == (32, 72)


def test_the_hint_stage_leaves_the_layers_above_the_guided_one_alone() -> None:
    module = _student("hint", freeze_above_guided=True)
    frozen = [module.net.hidden[2], module.net.hidden[3], module.net.output]
    before = [layer.weight.clone() for layer in frozen]

    module.model_step(_batch(module))["loss"].backward()
    module.configure_optimizers()["optimizer"].step()

    assert all(torch.equal(layer.weight, was) for layer, was in zip(frozen, before))
    assert module.net.hidden[1].weight.grad is not None


def test_checkpoints_round_trip_and_leave_the_teacher_out(tmp_path: Path) -> None:
    teacher = _classifier(DEEPSETS)
    teacher_path = _fit_one_batch(teacher, _batch(teacher)[:2], tmp_path / "t")
    hint = _student("hint", freeze_above_guided=True)
    hint_path = _fit_one_batch(hint, _batch(hint), tmp_path / "h")
    student = _student(
        "kd", teacher_ckpt=str(teacher_path), warm_start_ckpt=str(hint_path)
    )
    x = _batch(student)[0]
    assert torch.equal(student.net.hidden[0].weight, hint.net.hidden[0].weight)

    loaded = DistillationModule.load_from_checkpoint(
        _fit_one_batch(student, _batch(student), tmp_path / "s"), weights_only=False
    )

    assert not any("teacher" in key for key in student.state_dict())
    assert torch.allclose(student.eval()(x), loaded.eval()(x), atol=1e-6)
    assert not student.teacher().training and student.teacher() is student.teacher()
