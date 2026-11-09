"""The base of the networks trained quantisation-aware with HGQ2, after the module of
that name in jet_tagging_gdl.

Lightning keeps the loop, the data and validation; Keras on JAX owns the network, its
optimiser and its training step.
"""

import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn

from src.algorithms.classifier import Classifier
from src.algorithms.components.hgq_port import (
    KerasNetwork,
    Quantisation,
    calibrate,
    configure_jax,
    export_port,
    quantiser_variables,
    require_cpu_jax,
    seed_datalane_bits,
    seed_weight_bits,
    set_beta,
    total_ebops,
)
from src.utils.pylogger import RankedLogger

# isort: split
# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.utils.sugar import PieceWiseSchedule

log = RankedLogger(__name__, rank_zero_only=True)

# Rows per chunk of a whole split on its way to Keras.
CHUNK = 65536
# Rows per batch of HGQ2's trace.
CALIBRATION_BATCH = 16384
VERSIONED = ("keras", "hgq2", "jax")
# Beside an export: the reference logits and labels that synthesise.py checks against.
REFERENCE_FILE = "reference.npz"


class KerasQuantisedModule(Classifier):
    """A network trained quantisation-aware with HGQ2, for synthesis by alkaid.

    ``BaseModule`` builds the float network of ``net`` and prices its FLOPs on it; the
    port that ``_build_port`` builds from it then takes its place as ``net``, a
    ``KerasNetwork`` without torch parameters, through which ``Classifier``'s step
    validates. Training is manual: each batch is one Keras ``train_on_batch`` and one
    step of the mock optimiser that Lightning puts in place of a torch one, which
    advances ``global_step``, without which ``ModelCheckpoint`` never saves. The Keras
    weights go inside the Lightning checkpoint and the Keras optimiser does not, so a
    run cannot resume mid-way.

    ``trace`` sets the integer bits of every wrapping quantiser on the run's whole
    training split, the range the hardware is built for, and ``export`` writes the
    traced port; ``ebops_total`` is NaN until a trace.

    :param init_from: a float checkpoint, or its run directory, that
        ``_copy_float_network`` copies at the start of a fit; None trains from scratch.
    :param quantisation: the fields of ``Quantisation`` that the config sets.
    :param beta: the constant weight of the EBOPs in the loss, read when
        ``beta_schedule`` is empty.
    :param beta_schedule: (epoch, beta, interpolation) points of HGQ2's
        PieceWiseSchedule.
    :param calibration_rows: the training rows on which the datalane bits of a copied
        network are seeded.
    :param floatx: Keras's float type; float64 reproduces the fixed-point arithmetic
        exactly, since the products of quantised values exceed float32's 24 bits.
    :param jax_platform: Lightning's accelerator, on which JAX trains; ``trace`` and
        ``export`` need ``cpu``.
    :param jax_devices: Lightning's devices.
    """

    def __init__(
        self,
        net: dict,
        num_classes: int,
        input_shape: tuple[int, ...],
        optimizer: dict,
        scheduler: dict | None = None,
        init_from: str | None = None,
        quantisation: dict | None = None,
        beta: float | None = None,
        beta_schedule: list[list] | None = None,
        calibration_rows: int = 100_000,
        floatx: str = "float64",
        jax_platform: str = "cpu",
        jax_devices: int | list[int] = 1,
    ) -> None:
        super().__init__(net, num_classes, input_shape, optimizer, scheduler)
        configure_jax(jax_platform, jax_devices, floatx)
        # Keras draws the initial weights from its own generator, seeded from torch.
        keras.utils.set_random_seed(torch.initial_seed() % 2**32)
        self.automatic_optimization = False
        self.quantisation = Quantisation(**(quantisation or {}))
        self.ebops_total = math.nan
        twin = self.net
        self.net = KerasNetwork(*self._build_port(twin), twin.required_features)

    def _build_port(self, twin: nn.Module) -> tuple[keras.Model, keras.Model]:
        """The port of the float network *twin* and the model of the words it reads."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_port()."
        )

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the trained float network that ``init_from`` names into the port."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _copy_float_network()."
        )

    def _keras_loss(self) -> keras.losses.Loss:
        """The objective of the Keras step, and the one place to change it: the cross
        entropy of the logits and the one-hot labels."""
        return keras.losses.CategoricalCrossentropy(from_logits=True)

    def _keras_inputs(self, x: torch.Tensor) -> np.ndarray:
        """The (rows, in_dim) rows the port reads of a batch or a whole split at the
        full stored width, as ``forward`` hands them to the network."""
        selected = (self._select(chunk) for chunk in x.split(CHUNK))
        prepared = (self.net.prepare(chunk, self.norm_scales) for chunk in selected)
        return np.concatenate([self.net.rows(chunk) for chunk in prepared])

    def setup(self, stage: str) -> None:
        """Bind to the datamodule and, before a fit, initialise the port."""
        super().setup(stage)
        datamodule = getattr(self._trainer, "datamodule", None)
        if stage == "fit" and datamodule is not None:
            self._initialise(datamodule)

    def _initialise(self, datamodule) -> None:
        """Copy the float network ``init_from`` names and, outside float mode, seed its
        bits.

        From scratch nothing is copied or seeded: the weights start at the scopes' bits,
        and HGQ2 traces the datalanes' integer bits in training.
        """
        if not self.hparams.init_from:
            return
        log.info(f"Copying the float network of {self.hparams.init_from}")
        self._copy_float_network(self.net.keras_model)
        if not self.quantisation.float_mode:
            self._seed_bits(datamodule)

    def _seed_bits(self, datamodule) -> None:
        """Seed the bits of the copied weights and, on the first ``calibration_rows``
        training rows, those of the datalanes."""
        model, q = self.net.keras_model, self.quantisation
        seed_weight_bits(model, q.weight_bits)
        x = datamodule.split("train")[0][: self.hparams.calibration_rows]
        rows = self._keras_inputs(x)
        ebops = seed_datalane_bits(model, rows, q.data_bits, CALIBRATION_BATCH)
        log.info(f"{ebops} EBOPs at {q.data_bits} data and {q.weight_bits} weight bits")

    def configure_optimizers(self) -> None:
        """Compile the port with the Keras optimiser of the config.

        :return: None, for which Lightning substitutes its mock optimiser.
        """
        optimizer = instantiate(self.hparams.optimizer)(**self._learning_rate())
        variables = quantiser_variables(self.net.keras_model)
        optimizer.exclude_from_weight_decay(var_list=variables)
        self.net.compile_keras(optimizer, self._keras_loss())
        return None

    def _learning_rate(self) -> dict[str, Any]:
        """The optimiser's ``learning_rate``: the scheduler of the config, stepped per
        batch over the whole run, or nothing without one."""
        if not self.hparams.scheduler:
            return {}
        steps = max(int(self.trainer.estimated_stepping_batches), 1)
        return {"learning_rate": instantiate(self.hparams.scheduler)(decay_steps=steps)}

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """One Keras step, logging the epoch means of the cross entropy, of Keras's
        (micro) accuracy and of the loss with the EBOPs and bit penalties."""
        x, y = batch[0], batch[1]
        rows = self._keras_inputs(x)
        logs = self.net.train_on_batch(rows, y.cpu().numpy().astype(rows.dtype))
        # The mock optimiser's step does nothing but advance global_step.
        self.optimizers().step()
        # Keras resets its metrics at every train_on_batch, so acc is the batch's own.
        self._epoch_hits += logs["acc"] * len(x)
        self._epoch_rows += len(x)
        means = {"train/loss": logs["ce"], "train/acc": logs["acc"]}
        means["train/loss_total"] = logs["loss"]
        self.log_dict(means, on_step=False, on_epoch=True, batch_size=len(x))

    def _beta_at(self, epoch: int) -> float:
        """The value at *epoch* of HGQ2's PieceWiseSchedule over ``beta_schedule``, or
        the constant ``beta`` without one."""
        points = self.hparams.beta_schedule
        if not points:
            return float(self.hparams.beta or 0.0)
        return PieceWiseSchedule([tuple(point) for point in points])(epoch)

    def on_train_epoch_start(self) -> None:
        """Give every quantised layer this epoch's beta and start counting the epoch's
        correct predictions."""
        set_beta(self.net.keras_model, self._beta_at(self.current_epoch))
        self._epoch_hits, self._epoch_rows = 0.0, 0

    def on_train_epoch_end(self) -> None:
        """Log the EBOPs of the epoch's last step, its beta, the next learning rate and
        the epoch's Keras accuracy, the mean over its rows.

        ``BaseModule``'s line would read training metrics that the Keras step never
        updates.
        """
        epoch, ebops = self.current_epoch, total_ebops(self.net.keras_model)
        beta, lr = self._beta_at(epoch), self.net.learning_rate()
        acc = self._epoch_hits / max(self._epoch_rows, 1)
        self.log_dict({"train/ebops": float(ebops), "train/beta": beta, "train/lr": lr})
        log.info(
            f"[epoch {epoch:>3d}] train | EBOPs {ebops}, beta {beta:.3g}, lr {lr:.3g}, "
            f"acc {acc:.4f}"
        )

    def on_fit_start(self) -> None:
        """Send the count of Keras variables to the loggers; the torch count is zero."""
        count = self.net.keras_model.count_params()
        for logger in self.loggers:
            logger.log_hyperparams({"algorithm/keras_params": count})

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep the Keras weights, the quantisers' bits among them, beside the torch
        state."""
        checkpoint["keras_weights"] = self.net.keras_model.get_weights()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the Keras weights, cast to this port's float type, which may differ
        from the training's."""
        variables = self.net.keras_model.weights
        weights = zip(checkpoint["keras_weights"], variables, strict=True)
        self.net.keras_model.set_weights([np.asarray(w, v.dtype) for w, v in weights])

    def trace(self, datamodule) -> int:
        """Trace the integer bits of every wrapping quantiser afresh on the run's whole
        training split, the range the hardware is built for, and keep the EBOPs of the
        traced port as ``ebops_total``.

        :return: those EBOPs.
        :raises ValueError: when JAX does not run on the CPU.
        """
        require_cpu_jax()
        start = time.perf_counter()
        rows = self._keras_inputs(datamodule.split("train")[0])
        ebops = calibrate(self.net.keras_model, rows, CALIBRATION_BATCH)
        self.ebops_total, seconds = float(ebops), time.perf_counter() - start
        log.info(f"Traced {len(rows)} training rows in {seconds:.0f} s: {ebops} EBOPs")
        return ebops

    def export(
        self,
        directory: str | Path,
        datamodule,
        provenance: dict[str, Any],
        split: str = "test",
    ) -> dict[str, Any]:
        """Trace the port on the run's training split, then write what synthesis reads
        of *split*: the files of ``export_port`` and ``reference.npz``, which holds
        ``logits_clean``, the (rows, classes) logits in Keras's float type, and
        ``labels``, the (rows,) class indices.

        :param provenance: entries of ``export.json`` beside the port's own; its ``row``
            names the design in the CSV of synthesise.py.
        :param split:``test`` for a design to synthesise, ``fold`` (the run's validation
            rows) for a latency estimate that must not read the test split.
        :return: the contents of ``export.json``.
        :raises ValueError: when JAX does not run on the CPU.
        """
        self.trace(datamodule)
        x, y, _ = datamodule.split(split)
        rows = self._keras_inputs(x)
        logits, labels = self.net.predict(rows), y.argmax(dim=1).numpy()
        meta = self._export_meta(datamodule, logits, labels) | {"split": split}
        meta |= provenance
        words = self.net.quantised_inputs(rows)
        export_port(directory, self.net.keras_model, words, meta)
        np.savez(Path(directory) / REFERENCE_FILE, logits_clean=logits, labels=labels)
        log.info(
            f"Exported the traced port and {len(labels)} {split} rows to {directory}"
        )
        return meta

    def _export_meta(
        self, datamodule, logits: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """The port behind an export, as synthesise.py reads it; the port's name is the
        top module of its design."""
        keys = ("beta", "beta_schedule", "floatx", "init_from")
        return {
            "port": self.net.keras_model.name,
            "nconstituents": getattr(datamodule, "nconstituents", None),
            **{key: self.hparams[key] for key in keys},
            "ebops": int(self.ebops_total),
            "quantisation": asdict(self.quantisation),
            "n_rows": len(labels),
            # The micro accuracy on the exported split, as results.csv's accuracy column.
            "accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
            "versions": {name: version(name) for name in VERSIONED},
        }
