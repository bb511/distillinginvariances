"""Components of porting a network to HGQ2, after jet_tagging_gdl's module of that name.

The quantisers, the Keras network behind the interface of a torch ``FeatureNet``, and
the steps that seed, trace, price and export a port. Importing this module selects
Keras's JAX backend, so it is imported before Keras.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
# JAX reserves most of a GPU's memory at its first allocation otherwise, beside torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import keras
import numpy as np
import torch
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.constraints import Min, MinMax
from hgq.layers import QDense, Quantizer
from hgq.layers.core import QLayerBase
from hgq.quantizer.internal import (
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
    TrainableQuantizerBase,
)
from hgq.regularizers import MonoL1
from hgq.utils import trace_minmax
from jax._src.xla_bridge import backends_are_initialized
from keras import ops
from keras.src import backend as keras_backend

from src.algorithms.components import FeatureNet

if keras.backend.backend() != "jax":
    raise ImportError(
        f"Keras runs on the {keras.backend.backend()!r} backend, but the HGQ2 port "
        "needs 'jax'. Unset KERAS_BACKEND, or set it to jax, before anything imports "
        "keras."
    )


def _float64_kept(name: str):
    """Keras's elementwise *name* on JAX, computed in float64 for a float64 tensor."""
    keras_fn, jax_fn = getattr(keras_backend.numpy, name), getattr(jax.numpy, name)

    def fn(x):
        x = ops.convert_to_tensor(x)
        if keras.backend.standardize_dtype(x.dtype) == "float64":
            return jax_fn(x)
        return keras_fn(x)

    return fn


# Keras 3.15 casts a float64 tensor to float32 in floor and ceil on JAX, so HGQ2's
# rounding floor(x 2^f + 1/2) misrounds values within float32 resolution of a tie.
for _name in ("floor", "ceil"):
    setattr(keras_backend.numpy, _name, _float64_kept(_name))

# The least integer bits of a weight or bias: alkaid 0.8.1 emulates wrongly a layer
# whose weights carry about 30 fraction bits (jet_tagging_gdl, 2026-10-10).
WEIGHT_I_MIN = -12
# The initial bits of every weight (b0) and fraction bits of every datalane (f0), and
# the L1 weight of every bit count: those of the published JEDI-linear
# (arXiv:2508.15468).
INITIAL_BITS = 7
BITS_L1 = 1e-8
# The range within which HGQ2 traces the integer bits of a datalane.
DATALANE_I_RANGE = (0, 12)
# Rows per Keras call on a whole split.
PREDICT_BATCH = 16384
# A GPU run keeps the CPU backend beside CUDA, the default: HGQ2's trace_minmax stages
# its data under keras.device("cpu:0").
_PLATFORMS = {"gpu": "cuda,cpu", "cuda": "cuda,cpu", "cpu": "cpu"}


def _quantiser_scopes() -> list:
    """HGQ2's defaults as JEDI-linear's ``get_model`` and ``get_gnn`` set them, every
    datalane quantised per element, followed by the floor ``WEIGHT_I_MIN`` on the
    integer bits of every weight and bias; jet_tagging_gdl's ``_quantiser_scopes`` at
    ``per_slot``."""
    l1 = MonoL1(BITS_L1)
    kbi = dict(default_q_type="kbi", b0=INITIAL_BITS, i0=0, fr=l1, ir=l1)
    kif = dict(default_q_type="kif", place="datalane", f0=INITIAL_BITS, fr=l1)
    return [
        QuantizerConfigScope(overflow_mode="wrap", **kbi),
        QuantizerConfigScope(overflow_mode="wrap", ic=MinMax(*DATALANE_I_RANGE), **kif),
        LayerConfigScope(beta0=0),
        QuantizerConfigScope(place=("weight", "bias"), overflow_mode="SAT_SYM"),
        QuantizerConfigScope(place="datalane", heterogeneous_axis=None),
        QuantizerConfigScope("kbi", ("weight", "bias"), ic=Min(WEIGHT_I_MIN)),
    ]


@dataclass
class Quantisation:
    """The quantisers of the port.

    :param data_bits: i + f of every datalane once a copied float network is seeded, i
        traced on the training rows.
    :param weight_bits: the bits of every weight and bias once a copied float network is
        seeded, the integer bits those its value needs.
    :param float_mode: replaces every quantiser by an identity, for the comparison with
        the float network.
    """

    data_bits: int = 14
    weight_bits: int = 8
    float_mode: bool = False

    def scopes(self) -> list:
        """The scopes the layers are built in.

        In float mode the identities are passed to each layer instead (``dense_confs``),
        as jet_tagging_gdl's vector-neuron port builds its float mode.
        """
        if self.float_mode:
            return [LayerConfigScope(beta0=0.0, enable_ebops=True)]
        return _quantiser_scopes()

    def dense_confs(self) -> dict[str, QuantizerConfig]:
        """The quantisers a ``QDense`` takes beside the scopes' defaults: none, or the
        identities of float mode."""
        if not self.float_mode:
            return {}
        places = {"iq_conf": "datalane", "kq_conf": "weight", "bq_conf": "bias"}
        return {key: QuantizerConfig("dummy", place) for key, place in places.items()}


def _cross_entropy(labels, logits):
    return keras.losses.categorical_crossentropy(labels, logits, from_logits=True)


def _accuracy(labels, logits):
    return keras.metrics.categorical_accuracy(labels, logits)


def _labels_of(metric, classes: int):
    """*metric* of the targets' first *classes* columns, the one-hot labels."""

    def fn(targets, logits):
        return metric(targets[:, :classes], logits)

    return fn


def distillation_loss(classes: int, temperature: float, alpha: float):
    """The Keras twin of ``DistillationModule``'s KD objective, T^2 KL(p_T || p_S) plus
    alpha times the cross entropy, both averaged over the batch.

    The targets are (batch, 2 classes): the one-hot labels, then the teacher's logits.
    The KL takes 0 log 0 as 0, as torch's ``kl_div`` does.
    """

    def loss(targets, logits):
        labels, teacher = targets[:, :classes], targets[:, classes:]
        log_p = ops.log_softmax(logits / temperature, axis=-1)
        q = ops.softmax(teacher / temperature, axis=-1)
        log_q = ops.log(ops.where(q > 0, q, 1.0))
        kl = ops.sum(q * (log_q - log_p), axis=-1) * temperature**2
        return kl + alpha * _cross_entropy(labels, logits)

    return loss


class KerasNetwork(FeatureNet):
    """The Keras port behind the interface of the float network it replaces.

    It holds no torch parameter: Keras owns the weights, the optimiser and the step, and
    ``forward`` serves validation and evaluation as the float network's would. The
    samples are flattened to (samples, in_dim) rows, as ``MLPStudent`` flattens them.
    """

    def __init__(
        self,
        keras_model: keras.Model,
        inputs_model: keras.Model,
        required_features: list[str] | None,
    ) -> None:
        super().__init__()
        self.keras_model = keras_model
        self.inputs_model = inputs_model
        self.required_features = required_features

    def rows(self, x: torch.Tensor) -> np.ndarray:
        """The (samples, in_dim) rows of prepared samples, in Keras's float type."""
        flat = x.detach().flatten(start_dim=1).cpu().numpy()
        return flat.astype(self.keras_model.inputs[0].dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (samples, classes) logits of prepared samples, in Keras's float
        type, on the device of *x*."""
        logits = self.keras_model.predict_on_batch(self.rows(x))
        return torch.from_numpy(np.array(logits)).to(x.device)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """The (rows, classes) logits of a whole split's rows."""
        logits = self.keras_model.predict(rows, batch_size=PREDICT_BATCH, verbose=0)
        return np.asarray(logits)

    def quantised_inputs(self, rows: np.ndarray) -> np.ndarray:
        """The (rows, in_dim) words the network reads of *rows*: the output of its first
        input quantiser."""
        words = self.inputs_model.predict(rows, batch_size=PREDICT_BATCH, verbose=0)
        return np.asarray(words)

    def compile_keras(
        self, optimizer: keras.optimizers.Optimizer, loss, classes: int
    ) -> None:
        """Compile with *loss* of the targets and the logits; the layers add beta times
        their EBOPs and the L1 penalties on their bits.

        The targets are the one-hot labels, followed under distillation by the teacher's
        logits, so the metrics read their first *classes* columns alone.
        ``train_on_batch`` then reports ``loss`` (the total), ``ce`` and ``acc``.
        """
        metrics = [
            keras.metrics.MeanMetricWrapper(
                _labels_of(_cross_entropy, classes), name="ce"
            ),
            keras.metrics.MeanMetricWrapper(_labels_of(_accuracy, classes), name="acc"),
        ]
        self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_on_batch(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """One optimiser step on rows *x* and one-hot labels *y*."""
        logs = self.keras_model.train_on_batch(x, y, return_dict=True)
        return {key: float(value) for key, value in logs.items()}

    def learning_rate(self) -> float:
        """The learning rate of the next step."""
        return float(ops.convert_to_numpy(self.keras_model.optimizer.learning_rate))


def configure_jax(platform: str, devices: int | Sequence[int], floatx: str) -> None:
    """Set JAX's precision and Keras's float type, the platform, and the devices.

    :param platform: Lightning's accelerator, ``gpu`` (or ``cuda``) or ``cpu``.
    :param devices: Lightning's devices, a count or a list of CUDA indices.
    :param floatx: Keras's float type, float32 or float64; float64 also switches on
        JAX's x64.
    """
    jax.config.update("jax_enable_x64", floatx == "float64")
    # TF32 would round the quantised inputs of every product on a GPU.
    jax.config.update("jax_default_matmul_precision", "highest")
    keras.config.set_floatx(floatx)
    # The device list is read once, when the first backend starts.
    if backends_are_initialized():
        return
    ids = range(devices) if isinstance(devices, int) else devices
    jax.config.update("jax_platforms", _PLATFORMS.get(platform, platform))
    jax.config.update("jax_cuda_visible_devices", ",".join(str(i) for i in ids))


def require_cpu_jax() -> None:
    """Refuse to trace or export unless JAX runs on the CPU.

    :raises ValueError: on any other backend, where XLA's float64 2**f is off by an ulp
        for some f, so that the port would round some rows otherwise than alkaid.
    """
    backend = jax.default_backend()
    if backend != "cpu":
        raise ValueError(
            f"JAX runs on {backend!r}, where XLA's float64 2**f is inexact, so the "
            "exported port and its reference logits would not match alkaid bit for "
            "bit. Load the module with jax_platform='cpu' and JAX_PLATFORMS=cpu."
        )


def _weight_values(layer: keras.Layer) -> list[tuple[Quantizer, Any]]:
    """Pair each weight and bias quantiser of a dense *layer* with the value it
    quantises; no pair for any other layer."""
    if not isinstance(layer, QDense):
        return []
    biases = [] if layer.bias is None else [(layer.bq, layer.bias)]
    return [(layer.kq, layer.kernel), *biases]


def seed_weight_bits(model: keras.Model, bits: int) -> None:
    """Give every trainable weight and bias quantiser *bits* bits.

    The integer bits receive what its present value needs, at least ``WEIGHT_I_MIN``,
    such that no copied weight saturates or wraps.
    """
    for layer in model.layers:
        for quantiser, value in _weight_values(layer):
            inner = quantiser.quantizer
            if not (isinstance(inner, FixedPointQuantizerKBI) and inner.trainable):
                continue
            inner._b.assign(ops.full(inner._b.shape, bits, inner._b.dtype))
            inner._i.assign(ops.maximum(inner.get_minimal_i(value), WEIGHT_I_MIN))


def _wrap_lanes(model: keras.Model) -> list[FixedPointQuantizerKIF]:
    """The trainable datalane quantisers, whose integer bits are traced."""
    # Case-blind: the scopes spell the mode "wrap", jet_tagging_gdl's configs "WRAP".
    return [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerKIF)
        and layer.overflow_mode.upper() == "WRAP"
        and layer.trainable
    ]


def calibrate(model: keras.Model, inputs: np.ndarray, batch_size: int = 16384) -> int:
    """Trace the integer bits of every WRAP quantiser on rows *inputs* afresh.

    Return the EBOPs of the traced model.
    """
    return int(trace_minmax(model, inputs, batch_size=batch_size))


def seed_datalane_bits(
    model: keras.Model, inputs: np.ndarray, bits: int, batch_size: int = 16384
) -> int:
    """Give every trainable datalane i + f = *bits*, i traced on rows *inputs*.

    :return: the EBOPs of the last trace, within a bit per lane of the seeded model.
    """
    # Twice: the first trace reads each lane through upstream lanes still at their
    # initial f, and a lane whose range the second trace widens would wrap.
    for _ in range(2):
        ebops = calibrate(model, inputs, batch_size)
        for lane in _wrap_lanes(model):
            lane._f.assign(bits - lane.i)
    return ebops


def set_beta(model: keras.Model, beta: float) -> None:
    """Set the EBOPs weight of every quantised layer, as HGQ2's BetaScheduler does."""
    for layer in model._flatten_layers():
        if getattr(layer, "_beta", None) is not None:
            layer._beta.assign(ops.convert_to_tensor(beta, dtype=layer._beta.dtype))


def total_ebops(model: keras.Model) -> int:
    """The EBOPs of the last training step or trace, summed over the layers."""
    layers = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, QLayerBase) and layer.enable_ebops
    ]
    return sum(int(ops.convert_to_numpy(layer._ebops)) for layer in layers)


def quantiser_variables(model: keras.Model) -> list[keras.Variable]:
    """The variables of every quantiser, which weight decay must leave alone."""
    return [
        variable
        for layer in model._flatten_layers()
        if isinstance(layer, TrainableQuantizerBase)
        for variable in layer.variables
    ]


def export_port(
    directory: str | Path,
    model: keras.Model,
    inputs: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Write what synthesis reads: ``model.keras``, ``inputs.npz`` (``inputs_clean``,
    the words the network reads) and ``export.json`` (*meta*).

    The words are stored in float32 where that is exact, else in their own type.

    :return: *directory*, created if missing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.keras")
    words = np.asarray(inputs, dtype=np.float32)
    if not np.array_equal(words, inputs):
        words = np.asarray(inputs)
    np.savez(directory / "inputs.npz", inputs_clean=words)
    (directory / "export.json").write_text(json.dumps(meta, indent=2))
    return directory
