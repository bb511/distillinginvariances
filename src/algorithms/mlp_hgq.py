"""The MLP student trained quantisation-aware with HGQ2, for synthesis by alkaid.

The port is the float student layer for layer: one quantised dense layer per
``nn.Linear`` of the ``MLPStudent`` that the ``net`` config builds, ReLU between them,
no batch normalisation, and logits without softmax or output quantiser. It reads the
student's stored columns as the record's loader normalised them, flattened as the float
student flattens them, so the first layer's trainable input quantiser is the only input
quantisation: the ``standardised`` path of jet_tagging_gdl's JEDI-linear port without
its standardisation. Every quantiser is that port's (``Quantisation.scopes``).
"""

import functools
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate

from src.algorithms.components.mlp import MLPStudent
from src.algorithms.keras_base import KerasQuantisedModule
from src.utils.checkpoints import checkpoint_of

# isort: split
# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.layers import QDense

# The port's name, which synthesise.py makes the top module of the design.
PORT = "mlp_hgq"


def _float_student(init_from: str, net: dict) -> MLPStudent:
    """Rebuild in eval mode the float student of a ``Classifier`` or
    ``DistillationModule`` checkpoint, or of the checkpoint a run directory's manifest
    names, at the architecture of the config *net*.

    :raises RuntimeError: from ``load_state_dict``, when the checkpoint's student has
        another architecture.
    """
    path = checkpoint_of(init_from) if Path(init_from).is_dir() else init_from
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    student = instantiate(net)
    prefix = "net."
    own = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    student.load_state_dict(own)
    return student.eval()


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().double().numpy()


def _dense_layers(model: keras.Model) -> list[QDense]:
    """The quantised dense layers of *model*, input first."""
    return [layer for layer in model.layers if isinstance(layer, QDense)]


class MLPHGQ(KerasQuantisedModule):
    """The cell's MLP student as an HGQ2 port, trained on the labels."""

    def _build_port(self, twin: MLPStudent) -> tuple[keras.Model, keras.Model]:
        """Build the port of *twin* and the model of the words it reads.

        Call ``configure_jax`` first: the layers take Keras's floatx at construction.

        :return: the network, (rows, in_dim) to (rows, classes) logits, and the model of
            the output of its first input quantiser, the words alkaid reads.
        """
        linears = [*twin.hidden, twin.output]
        x = keras.Input((linears[0].in_features,), name="features")
        with ExitStack() as scopes:
            for scope in self.quantisation.scopes():
                scopes.enter_context(scope)
            layers = self._dense_stack([linear.out_features for linear in linears])
            logits = functools.reduce(lambda h, layer: layer(h), layers, x)
        # alkaid's trace of the first quantiser is the identity on its own output.
        words = layers[0].iq(x)
        return keras.Model(x, logits, name=PORT), keras.Model(x, words, name="words")

    def _dense_stack(self, widths: list[int]) -> list[QDense]:
        """One quantised dense layer per width, ReLU on every one but the last, whose
        output is the logits."""
        confs, last = self.quantisation.dense_confs(), len(widths) - 1
        return [
            QDense(
                width,
                activation=None if j == last else "relu",
                name="logits" if j == last else f"dense{j}",
                **confs,
            )
            for j, width in enumerate(widths)
        ]

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy every ``nn.Linear`` of the float student of ``init_from`` into its dense
        layer: the kernel is the transposed weight."""
        student = _float_student(self.hparams.init_from, self.hparams.net)
        linears = [*student.hidden, student.output]
        for layer, linear in zip(_dense_layers(model), linears, strict=True):
            layer.kernel.assign(_numpy(linear.weight).T)
            layer.bias.assign(_numpy(linear.bias))

    def _export_meta(
        self, datamodule, logits: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """The base's record, with the widths from the input to the logits and the
        columns the student reads."""
        model = self.net.keras_model
        widths = [layer.units for layer in _dense_layers(model)]
        stack = [int(model.inputs[0].shape[-1]), *widths]
        meta = super()._export_meta(datamodule, logits, labels)
        return meta | {"stack": stack, "features": self.net.required_features}
