"""One evaluation pass: every metric column of the results contract for one
checkpoint."""

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, ListConfig

from src.data.components.augmentation import apply_group
from src.metrics.calibration import ECEAccumulator, class_indices
from src.metrics.divergence import entropy
from src.metrics.fidelity import fidelity
from src.metrics.invariance import InvarianceAccumulator
from src.metrics.temperature import fit_temperature, scale

N_ECE_BINS = 15
# The keyword arguments of the probe, which name the width of the group's draws.
PROBE_KWARGS = ("lorentz_eta_max", "max_shift", "n_swaps")
# Carried out of the pass for the caller's temperature fit, never written to the table.
TENSOR_KEYS = ("logits", "targets")
_FIDELITY_KEYS = ("fid_top1", "fid_1mjsd", "fid_top1_g", "fid_1mjsd_g")
_INVARIANCE_KEYS = ("inv_top1", "inv_1mjsd")


def takes_matrix(module: LightningModule) -> bool:
    """Whether the module's network reads the per-jet Lorentz matrix; LorentzNet's
    does."""
    return "lorentz_matrix" in inspect.signature(module.net.forward).parameters


def net_kwargs(
    module: LightningModule, matrices: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """What the module's forward takes beside the batch, given the split's matrices."""
    if matrices is None or not takes_matrix(module):
        return {}
    return {"lorentz_matrix": matrices}


def resolve_device(cfg: DictConfig) -> torch.device:
    """Where the pass runs; no Trainer is involved, so the device is chosen here."""
    if cfg.trainer.get("accelerator") != "gpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    devices = cfg.trainer.get("devices")
    index = devices[0] if isinstance(devices, (list, ListConfig)) else 0
    return torch.device(f"cuda:{int(index)}")


def evaluate_checkpoint(
    module: LightningModule,
    teacher: LightningModule,
    datamodule: LightningDataModule,
    split: str,
    probe: DictConfig | dict[str, Any],
    temperature_ts: float | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Every metric of the results contract for *module* on one split.

    One fp32 pass over the split: the student's logits, the teacher's on the same rows,
    then ``probe.n_draws`` independent per-sample transforms, the first of which also
    serves the fidelity under the group action (the ``_g`` columns).

    :param teacher: the reference of the fidelity columns; pass *module* itself for a
        teacher run, which puts the fidelity columns at 1.
    :param probe: ``n_draws`` and the transform's keyword arguments.
    :param temperature_ts: the temperature fitted on the validation fold; None fits one on
        this split's own logits, which is in-sample.
    :return: the contract's metric keys, plus the CPU ``logits`` and ``targets`` the caller
        carries to the next split.
    """
    pass_ = _Pass(module, teacher, _drawer(datamodule, probe), int(probe["n_draws"]))
    _run(pass_, datamodule, split, batch_size)
    parts = pass_.collected()
    metrics = _quality(parts["logits"], parts["targets"])
    metrics |= _fidelities(parts) | _invariance(pass_)
    metrics |= _temperature(parts["logits"], parts["targets"], temperature_ts)
    return metrics | {key: parts[key] for key in TENSOR_KEYS}


def constant_row(
    datamodule: LightningDataModule, split: str, batch_size: int = 1024
) -> dict[str, Any]:
    """The metrics of the predictor that always returns the training class frequencies.

    The floor every trained row is read against. Its fidelity and its invariance are 1
    by construction, and ``inv_M`` records that a constant predictor needs no probe
    draw.
    """
    targets = torch.cat(
        [y.cpu() for _, y, _ in datamodule.eval_batches(split, batch_size)]
    )
    frequencies = datamodule.train_class_frequencies().cpu().double()
    logits = frequencies.log().expand(targets.shape[0], -1)
    metrics = _quality(logits, targets) | _temperature(logits, targets, None)
    trivial = dict.fromkeys(_FIDELITY_KEYS + _INVARIANCE_KEYS, 1.0)
    return metrics | trivial | {"inv_M": 0}


class _Pass:
    """The per-batch work of one evaluation pass, holding its accumulators.

    The probe's first draw is kept rather than redrawn: the transformed-input fidelity
    and the first invariance draw must see the same jets, or the two columns are
    measured on different transforms.
    """

    def __init__(
        self,
        module: LightningModule,
        teacher: LightningModule,
        draw: Callable,
        n_draws: int,
    ) -> None:
        self.module = module
        self.teacher = teacher
        self.draw = draw
        self.n_draws = n_draws
        self.parts: dict[str, list[torch.Tensor]] = {
            name: [] for name in (*TENSOR_KEYS, "teacher", "student_g", "teacher_g")
        }
        self.invariance = InvarianceAccumulator()

    def batch(
        self, x: torch.Tensor, y: torch.Tensor, matrices: torch.Tensor | None
    ) -> None:
        logits = _forward(self.module, x, matrices)
        self._keep(logits=logits, targets=y)
        self._keep(teacher=_forward(self.teacher, x, matrices))
        for index in range(self.n_draws):
            self._probe(x, matrices, logits, first=index == 0)

    def collected(self) -> dict[str, torch.Tensor]:
        return {name: torch.cat(values) for name, values in self.parts.items()}

    def _probe(
        self,
        x: torch.Tensor,
        matrices: torch.Tensor | None,
        logits: torch.Tensor,
        first: bool,
    ) -> None:
        x_t, params = self.draw(x)
        probed = _compose(params, matrices)
        transformed = _forward(self.module, x_t, probed)
        self.invariance.update(logits, transformed)
        if first:
            self._keep(student_g=transformed)
            self._keep(teacher_g=_forward(self.teacher, x_t, probed))

    def _keep(self, **tensors: torch.Tensor) -> None:
        for name, value in tensors.items():
            self.parts[name].append(value.detach().cpu())


@torch.no_grad()
def _run(
    pass_: _Pass, datamodule: LightningDataModule, split: str, batch_size: int
) -> None:
    """Feed every batch of the split through the pass, in fp32 on the module's
    device."""
    device = next(pass_.module.parameters()).device
    for x, y, matrices in datamodule.eval_batches(split, batch_size):
        moved = None if matrices is None else matrices.to(device).float()
        pass_.batch(x.to(device).float(), y, moved)


def _drawer(
    datamodule: LightningDataModule, probe: DictConfig | dict[str, Any]
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]:
    """One independent transform per sample, of the group the run is judged on."""
    kwargs = {key: probe[key] for key in PROBE_KWARGS if key in probe}
    return lambda x: apply_group(
        datamodule.group, x, datamodule.feature_names, datamodule.norm_scales, **kwargs
    )


def _forward(
    module: LightningModule, x: torch.Tensor, matrices: torch.Tensor | None
) -> torch.Tensor:
    return module(x, **net_kwargs(module, matrices))


def _compose(
    params: torch.Tensor | None, matrices: torch.Tensor | None
) -> torch.Tensor | None:
    """The matrix a network must see under the probe: the probe's boost after the
    data's.

    The permutation and translation groups return index tensors rather than matrices,
    and leave the data's own matrices untouched.
    """
    if params is None or params.ndim != 3 or tuple(params.shape[-2:]) != (4, 4):
        return matrices
    return params if matrices is None else params @ matrices


def _quality(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Accuracy, likelihood, calibration and the two spread columns of one logit set."""
    nll, ece = _nll_and_ece(logits, targets)
    probability = logits.double().softmax(-1)
    return {
        "accuracy": float(logits.argmax(-1).eq(class_indices(targets)).double().mean()),
        "nll": nll,
        "ece": ece,
        "ece_bins": N_ECE_BINS,
        "mean_confidence": float(probability.max(-1).values.mean()),
        "mean_entropy": float(entropy(probability).mean()),
    }


def _nll_and_ece(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    labels = class_indices(targets)
    accumulator = ECEAccumulator(N_ECE_BINS)
    accumulator.update(logits, targets)
    # Summed then divided, so the accumulation stays float64 whatever the split size.
    total = F.cross_entropy(logits.double(), labels, reduction="sum")
    return float(total) / labels.shape[0], accumulator.compute()


def _fidelities(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    """Fidelity on the clean rows and under one draw of the group action."""
    clean = fidelity(parts["logits"], parts["teacher"])
    transformed = fidelity(parts["student_g"], parts["teacher_g"])
    return dict(zip(_FIDELITY_KEYS, clean + transformed))


def _invariance(pass_: _Pass) -> dict[str, Any]:
    top1, similarity = pass_.invariance.compute()
    return {"inv_top1": top1, "inv_1mjsd": similarity, "inv_M": pass_.n_draws}


def _temperature(
    logits: torch.Tensor, targets: torch.Tensor, temperature_ts: float | None
) -> dict[str, float]:
    temperature = (
        fit_temperature(logits, targets)
        if temperature_ts is None
        else float(temperature_ts)
    )
    nll, ece = _nll_and_ece(scale(logits, temperature), targets)
    return {"nll_ts": nll, "ece_ts": ece, "temperature_ts": temperature}
