import os

# Before hgq imports Keras, and before anything starts JAX.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer

# The optional Poetry group hgq; the suite passes without it.
pytest.importorskip("hgq")

import keras
from hgq.layers import QDense

from src.algorithms.classifier import Classifier
from src.algorithms.components.hgq_port import _wrap_lanes
from src.algorithms.mlp_hgq import MLPHGQ
from tests.conftest import ROOT, TinyJets, tiny_jets

MLP = {
    "_target_": "src.algorithms.components.mlp.MLPStudent",
    "in_dim": 24,
    "hidden": [16, 8],
    "output_dim": 3,
    "required_features": ["px", "py", "pz"],
}
KERAS_ADAM = {
    "_target_": "keras.optimizers.Adam",
    "_partial_": True,
    "learning_rate": 1e-3,
}
TORCH_ADAM = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}


def _module(**overrides) -> MLPHGQ:
    settings = {
        "net": MLP,
        "num_classes": 3,
        "input_shape": (8, 4),
        "optimizer": KERAS_ADAM,
        "beta": 1e-6,
        "calibration_rows": 16,
    }
    return MLPHGQ(**(settings | overrides))


def _bound(tmp_path: Path, **overrides) -> tuple[MLPHGQ, TinyJets]:
    datamodule = tiny_jets(tmp_path, fold_idx=None)
    module = _module(**overrides)
    module.bind(datamodule)
    return module, datamodule


def _float_checkpoint(tmp_path: Path, net: dict) -> tuple[Classifier, Path]:
    classifier = Classifier(
        net=net, num_classes=3, input_shape=(8, 4), optimizer=TORCH_ADAM
    )
    path = tmp_path / "float.ckpt"
    torch.save({"state_dict": classifier.state_dict()}, path)
    return classifier, path


def _fit(module: MLPHGQ, datamodule, tmp_path: Path, epochs: int) -> Trainer:
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule)
    return trainer


def _array(value) -> np.ndarray:
    return keras.ops.convert_to_numpy(value)


def _dense_units(module: MLPHGQ) -> list[int]:
    layers = module.net.keras_model.layers
    return [layer.units for layer in layers if isinstance(layer, QDense)]


def _synthesis_script():
    path = ROOT / "scripts" / "synthesis" / "synthesise.py"
    spec = importlib.util.spec_from_file_location("synthesise", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_port_has_a_quantised_dense_layer_per_linear_of_the_float_student(
    tmp_path: Path,
) -> None:
    module, datamodule = _bound(tmp_path, net=MLP | {"depth": 1, "width_factor": 2})

    logits = module(datamodule.split("test")[0])

    assert _dense_units(module) == [32, 3]
    assert logits.shape == (8, 3) and logits.dtype == torch.float64
    assert module.flops_total > 0 and list(module.parameters()) == []


def test_training_steps_keras_once_per_batch_and_checkpoints_its_weights(
    tmp_path: Path,
) -> None:
    datamodule = tiny_jets(tmp_path, fold_idx=None)
    module = _module(beta_schedule=[[0, 1e-6, "linear"], [2, 1e-5, "constant"]])
    before = module.net.keras_model.get_weights()

    trainer = _fit(module, datamodule, tmp_path, epochs=2)
    after = module.net.keras_model.get_weights()
    restored = MLPHGQ.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, weights_only=False
    )

    assert trainer.global_step == 2 * len(datamodule.train_dataloader())
    assert any(not np.array_equal(old, new) for old, new in zip(before, after))
    assert trainer.callback_metrics["train/ebops"] > 0
    loaded = restored.net.keras_model.get_weights()
    for saved, restored_weight in zip(after, loaded, strict=True):
        np.testing.assert_array_equal(saved, restored_weight)


def test_init_from_reproduces_the_float_student_in_float_mode(tmp_path: Path) -> None:
    classifier, path = _float_checkpoint(tmp_path, MLP)
    module, datamodule = _bound(
        tmp_path, init_from=str(path), quantisation={"float_mode": True}
    )
    classifier.bind(datamodule)
    x = datamodule.split("test")[0]

    module._initialise(datamodule)
    with torch.no_grad():
        expected = classifier(x).double()

    # The float student sums in float32, the port in float64.
    tolerance = 1e-5 * expected.abs().max().item()
    torch.testing.assert_close(module(x), expected, rtol=0.0, atol=tolerance)


def test_init_from_refuses_a_float_student_of_another_stack(tmp_path: Path) -> None:
    _, path = _float_checkpoint(tmp_path, MLP | {"hidden": [8]})
    module = _module(init_from=str(path))

    with pytest.raises(RuntimeError, match="state_dict"):
        module._copy_float_network(module.net.keras_model)


def test_a_copied_student_is_seeded_at_the_configured_widths(tmp_path: Path) -> None:
    _, path = _float_checkpoint(tmp_path, MLP)
    module, datamodule = _bound(tmp_path, init_from=str(path))
    model = module.net.keras_model

    module._initialise(datamodule)
    lanes = _wrap_lanes(model)
    dense = [layer for layer in model.layers if isinstance(layer, QDense)]

    assert lanes and all(np.all(_array(x.i) + _array(x._f) == 14) for x in lanes)
    assert all(np.all(_array(layer.kq.quantizer._b) == 8) for layer in dense)


def test_the_export_predicts_its_reference_logits_from_the_stored_words(
    tmp_path: Path,
) -> None:
    module, datamodule = _bound(tmp_path)
    out = tmp_path / "export"

    meta = module.export(out, datamodule, {"row": "tiny"})
    model = keras.models.load_model(out / "model.keras", compile=False)
    with np.load(out / "inputs.npz") as inputs, np.load(out / "reference.npz") as ref:
        words, logits = inputs["inputs_clean"], ref["logits_clean"]
        labels = ref["labels"]

    predicted = model.predict(words.astype(np.float64), verbose=0)
    np.testing.assert_array_equal(predicted, logits)
    assert words.shape == (8, 24)
    assert labels.tolist() == datamodule.split("test")[1].argmax(dim=1).tolist()
    assert meta["ebops"] == module.ebops_total > 0
    assert (meta["port"], meta["row"]) == ("mlp_hgq", "tiny")
    assert meta["stack"] == [24, 16, 8, 3]
    assert json.loads((out / "export.json").read_text()) == meta


def test_synthesis_emulates_the_export_bit_for_bit(tmp_path: Path) -> None:
    pytest.importorskip("alkaid")
    module, datamodule = _bound(tmp_path)
    module.export(tmp_path / "export", datamodule, {"row": "tiny"})
    out = tmp_path / "synthesis"
    argv = ["--export-dir", str(tmp_path / "export"), "--out", str(out)]

    status = _synthesis_script().main([*argv, "--no-cosim", "--no-vivado"])
    check = json.loads((out / "check.json").read_text())

    assert status == 0 and check["project"] == "mlp_hgq"
    assert check["n_mismatch"] == check["n_mismatch_keras"] == 0
    assert (out / "synthesis.csv").is_file()


def test_the_keras_kd_objective_is_the_torch_one() -> None:
    from src.algorithms.components.hgq_port import distillation_loss
    from src.algorithms.losses.kd import kd_loss

    generator = torch.Generator().manual_seed(3)
    logits, teacher = (torch.randn(8, 3, generator=generator).double() for _ in "ab")
    labels = torch.eye(3, dtype=torch.float64)[torch.arange(8) % 3]
    targets = torch.cat([labels, teacher], dim=1)

    keras_loss = distillation_loss(3, 4.0, 0.5)(targets.numpy(), logits.numpy())
    torch_loss = kd_loss(logits, teacher, 4.0)
    torch_loss = torch_loss + 0.5 * torch.nn.functional.cross_entropy(
        logits, labels.argmax(1)
    )

    assert float(np.mean(_array(keras_loss))) == pytest.approx(
        float(torch_loss), rel=1e-10
    )


def test_a_distilled_port_steps_on_the_cached_teacher_logits(tmp_path: Path) -> None:
    module, datamodule = _bound(tmp_path, temperature=4.0, alpha=1.0)
    x, y, _ = datamodule.split("train")
    teacher = torch.randn(len(x), 3)
    rows = module._keras_inputs(x)

    targets = module._keras_targets((x, y, teacher, teacher), rows.dtype)
    module.configure_optimizers()
    logs = module.net.train_on_batch(rows, targets)

    assert targets.shape == (len(x), 6)
    assert all(np.isfinite(logs[key]) for key in ("loss", "ce", "acc"))
    with pytest.raises(ValueError, match="teacher cache"):
        module._keras_targets((x, y), rows.dtype)


def test_the_distilled_quantised_stage_composes_on_the_recipe() -> None:
    from tests.conftest import compose_root

    cfg = compose_root("train.yaml", "experiment=jets_deepsets", "method=hgq_kd")

    assert cfg.stage == "hgq_kd" and cfg.algorithm.temperature == 4.0
    assert cfg.data.teacher_cache.endswith("main.npz")
    assert cfg.trainer.max_epochs == 1000 and cfg.data.batch_size == 1024
    assert cfg.callbacks.model_checkpoint.every_n_epochs == 10
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert list(cfg.algorithm.beta_schedule[-1]) == [1000, 1.0e-6, "constant"]
