"""One evaluation pass: the metric columns of the results contract for one checkpoint on
one split, per probe, and the per-sample arrays the decomposition reads."""

import inspect
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, ListConfig

from src.data.components.augmentation import apply_group
from src.metrics.calibration import ECEAccumulator, class_indices
from src.metrics.divergence import entropy
from src.metrics.fidelity import pair_means
from src.metrics.invariance import invariance
from src.metrics.magnitude import magnitudes
from src.metrics.temperature import fit_temperature, scale

N_ECE_BINS = 15
# Per group, the keyword of apply_group that sets the width of its draws, the result
# column that records it and its short name in a probe's tag. The other groups are drawn
# over their whole range and have no knob.
PROBE_KNOBS = {
    "lorentz": ("lorentz_eta_max", "probe_eta_max", "eta"),
    "lorentz_iso": ("lorentz_eta_max", "probe_eta_max", "eta"),
    "z": ("lorentz_eta_max", "probe_eta_max", "eta"),
    "translation": ("max_shift", "probe_max_shift", "shift"),
    "permutation": ("n_swaps", "probe_n_swaps", "swaps"),
}
KNOB_COLUMNS = ("probe_eta_max", "probe_max_shift", "probe_n_swaps")
# Named in the order pair_means returns (top-1, sqrt(JSD), legacy 1 - JSD); zip drops
# the legacy value where a column set has none.
_FID = ("fid_top1", "fid_sqrtjsd", "fid_1mjsd")
_FID_G = ("fid_top1_g", "fid_sqrtjsd_g", "fid_1mjsd_g")
_INV = ("inv_top1", "inv_sqrtjsd", "inv_1mjsd")
_INV_G = ("inv_top1_g", "inv_sqrtjsd_g")
_TINV_G = ("tinv_top1_g", "tinv_sqrtjsd_g")


@dataclass(frozen=True)
class Probe:
    """A group, the draws per sample and the value of the group's knob (None: the whole
    group for ``max_shift`` and ``n_swaps``)."""

    group: str
    n_draws: int
    knob: float | int | None = None

    @property
    def kwargs(self) -> dict[str, Any]:
        spec = PROBE_KNOBS.get(self.group)
        return {} if spec is None else {spec[0]: self.knob}

    @property
    def tag(self) -> str:
        """The probe's name in file names and in the seed of its draws."""
        spec = PROBE_KNOBS.get(self.group)
        if spec is None:
            return self.group
        return f"{self.group}_{spec[2]}{'all' if self.knob is None else self.knob}"

    def columns(self) -> dict[str, Any]:
        cells = {"probe_group": self.group} | dict.fromkeys(KNOB_COLUMNS)
        spec = PROBE_KNOBS.get(self.group)
        if spec is not None:
            cells[spec[1]] = self.knob
        return cells


def make_probe(group: str, settings: DictConfig | dict[str, Any]) -> Probe:
    """The probe of *group* under the eval config's ``probe`` settings.

    :raises ValueError: below one draw, since the first draw serves the ``_g`` columns.
    """
    n_draws = int(settings["n_draws"])
    if n_draws < 1:
        raise ValueError(f"A probe needs n_draws >= 1, got {n_draws}.")
    spec = PROBE_KNOBS.get(group)
    return Probe(group, n_draws, None if spec is None else settings.get(spec[0]))


@dataclass
class Evaluation:
    """One split's results.

    ``metrics`` holds one dict per probe, in the order of the probes, each with the
    clean columns too. ``clean`` holds the student's and the teacher's ``(rows,
    classes)`` logits at x and the class indices; ``probed`` holds per probe every draw
    of the student, ``(draws, rows, classes)``, the teacher's logits on the first draw
    and that draw's magnitudes. ``temperatures`` is (student, teacher).
    """

    metrics: list[dict[str, Any]]
    temperatures: tuple[float, float]
    clean: dict[str, torch.Tensor]
    probed: list[dict[str, Any]]

    def samples(self, index: int) -> dict[str, np.ndarray]:
        """The per-sample arrays of probe *index*: what a dump holds."""
        kept = self.probed[index]
        logits = {
            "student": self.clean["student"],
            "teacher": self.clean["teacher"],
            "student_g": kept["draws"][0],
            "teacher_g": kept["teacher_g"],
        }
        arrays = {name: value.float().numpy() for name, value in logits.items()}
        sizes = {f"mag_{n}": v.float().numpy() for n, v in kept["magnitude"].items()}
        targets = self.clean["targets"].to(torch.int16).numpy()
        return arrays | sizes | {"targets": targets}


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
    probes: list[Probe],
    seed: int,
    temperatures: tuple[float, float] | None = None,
    batch_size: int = 1024,
) -> Evaluation:
    """Every metric of the results contract for *module* on one split, per probe.

    One fp32 pass: the student's and the teacher's logits at x, then for each probe
    ``n_draws`` independent per-sample transforms through the student, the first of
    which also goes through the teacher (the ``_g`` and ``tinv`` columns).

    :param teacher: the reference of the fidelity columns; pass *module* itself for a
        teacher run, which puts them at agreement 1 and distance 0.
    :param seed: with the split, the probe and the batch index, fixes every draw.
    :param temperatures: (student, teacher), fitted on the validation fold; None fits
        both by NLL on this split, which is in-sample.
    """
    pass_ = _Pass(module, teacher, datamodule, probes, (seed, split))
    _run(pass_, datamodule, split, batch_size)
    clean, probed = pass_.collected()
    temperatures = temperatures or _fit_temperatures(clean, teacher is module)
    shared = _clean_metrics(clean, temperatures)
    metrics = [shared | _probe_metrics(clean, kept, temperatures) for kept in probed]
    return Evaluation(metrics, temperatures, clean, probed)


def constant_row(
    datamodule: LightningDataModule, split: str, batch_size: int = 1024
) -> dict[str, Any]:
    """The metrics of the predictor that always returns the training class frequencies.

    The floor every trained row is read against. It is its own fidelity reference and
    needs no probe draw (``inv_M`` 0), so its agreements are 1 and its distances 0 by
    construction; it has no teacher, so the ``tinv`` columns stay blank.
    """
    targets = torch.cat(
        [y.cpu() for _, y, _ in datamodule.eval_batches(split, batch_size)]
    )
    frequencies = datamodule.train_class_frequencies().cpu().double()
    logits = frequencies.log().expand(targets.shape[0], -1)
    temperatures = (fit_temperature(logits, targets), None)
    metrics = _quality(logits, targets) | _scaled(logits, targets, temperatures)
    names = (*_FID, *_FID_G, *_INV, *_INV_G)
    trivial = {name: 0.0 if "sqrtjsd" in name else 1.0 for name in names}
    return metrics | trivial | {"inv_M": 0}


class _Pass:
    """The per-batch work of one evaluation pass, holding what it keeps on the CPU.

    Each probe draws under a seed of its own, fixed by (seed, split, probe tag, batch
    index), so two checkpoints evaluated at the same ``batch_size`` on the same device
    type meet the same transforms, whatever probes run beside them.
    """

    def __init__(
        self,
        module: LightningModule,
        teacher: LightningModule,
        datamodule: LightningDataModule,
        probes: list[Probe],
        stream: tuple[int, str],
    ) -> None:
        self.module, self.teacher, self.datamodule = module, teacher, datamodule
        self.probes, self.stream, self.index = probes, stream, 0
        self.clean = {name: [] for name in ("student", "teacher", "targets")}
        self.probed = [{"draws": [], "teacher_g": [], "magnitude": []} for _ in probes]

    def batch(
        self, x: torch.Tensor, y: torch.Tensor, matrices: torch.Tensor | None
    ) -> None:
        student = _forward(self.module, x, matrices)
        teacher = self._teacher(student, x, matrices)
        _keep(self.clean, student=student, teacher=teacher, targets=class_indices(y))
        for probe, kept in zip(self.probes, self.probed):
            with probe_rng(self.stream, probe.tag, self.index, x.device):
                self._probe(probe, kept, x, matrices)
        self.index += 1

    def collected(self) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        clean = {name: torch.cat(values) for name, values in self.clean.items()}
        return clean, [_concatenated(kept) for kept in self.probed]

    def _probe(
        self,
        probe: Probe,
        kept: dict[str, list],
        x: torch.Tensor,
        matrices: torch.Tensor | None,
    ) -> None:
        draws = []
        for index in range(probe.n_draws):
            x_t, params = apply_group(probe.group, x, *self._layout, **probe.kwargs)
            probed = _compose(params, matrices)
            draws.append(_forward(self.module, x_t, probed))
            if index == 0:
                _keep(kept, teacher_g=self._teacher(draws[0], x_t, probed))
                sizes = magnitudes(probe.group, params, x)
                kept["magnitude"].append({n: v.cpu() for n, v in sizes.items()})
        _keep(kept, draws=torch.stack(draws))

    @property
    def _layout(self) -> tuple[list[str] | None, dict[str, float] | None]:
        return self.datamodule.feature_names, self.datamodule.norm_scales

    def _teacher(
        self, student: torch.Tensor, x: torch.Tensor, matrices: torch.Tensor | None
    ) -> torch.Tensor:
        """The teacher's logits; a teacher run's own, without a second forward pass."""
        if self.teacher is self.module:
            return student
        return _forward(self.teacher, x, matrices)


@contextmanager
def probe_rng(
    stream: tuple[int, str], tag: str, batch: int, device: torch.device
) -> Iterator[None]:
    """Seed the draws of one probe on one batch of a split from (seed, split), its tag
    and the batch index; the ambient RNG is restored on exit."""
    with torch.random.fork_rng(devices=_cuda(device)):
        torch.manual_seed(_probe_seed(*stream, tag, batch))
        yield


def _cuda(device: torch.device) -> list[torch.device]:
    return [device] if device.type == "cuda" else []


def _probe_seed(seed: int, split: str, tag: str, batch: int) -> int:
    # crc32 rather than hash(): Python salts the hash of a str per process.
    return zlib.crc32(f"{seed}/{split}/{tag}/{batch}".encode())


def _keep(store: dict[str, list], **tensors: torch.Tensor) -> None:
    for name, value in tensors.items():
        store[name].append(value.detach().cpu())


def _concatenated(kept: dict[str, list]) -> dict[str, Any]:
    """One probe's batches joined along the rows; ``draws`` has its rows on axis 1."""
    names = kept["magnitude"][0].keys()
    return {
        "draws": torch.cat(kept["draws"], dim=1),
        "teacher_g": torch.cat(kept["teacher_g"]),
        "magnitude": {n: torch.cat([m[n] for m in kept["magnitude"]]) for n in names},
    }


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


def _fit_temperatures(
    clean: dict[str, torch.Tensor], shared: bool
) -> tuple[float, float]:
    """The student's and the teacher's temperatures; one fit when they are one model."""
    student = fit_temperature(clean["student"], clean["targets"])
    teacher = student if shared else fit_temperature(clean["teacher"], clean["targets"])
    return student, teacher


def _clean_metrics(
    clean: dict[str, torch.Tensor], temperatures: tuple[float, float]
) -> dict[str, Any]:
    student, targets = clean["student"], clean["targets"]
    fidelity = pair_means(student, clean["teacher"], *temperatures)
    scaled = _scaled(student, targets, temperatures)
    return _quality(student, targets) | scaled | dict(zip(_FID, fidelity))


def _probe_metrics(
    clean: dict[str, torch.Tensor],
    kept: dict[str, Any],
    temperatures: tuple[float, float],
) -> dict[str, Any]:
    """Fidelity on the first draw, the student's invariance over every draw and on the
    first, and the teacher's on the first: the decomposition's terms share that draw."""
    student_t, teacher_t = temperatures
    first = kept["draws"][0]
    fidelity = pair_means(first, kept["teacher_g"], student_t, teacher_t)
    inv = invariance(clean["student"], kept["draws"], student_t)
    inv_g = pair_means(clean["student"], first, student_t, student_t)
    tinv = pair_means(clean["teacher"], kept["teacher_g"], teacher_t, teacher_t)
    named = zip((_FID_G, _INV, _INV_G, _TINV_G), (fidelity, inv, inv_g, tinv))
    cells = {k: v for names, values in named for k, v in zip(names, values)}
    return cells | {"inv_M": kept["draws"].shape[0]}


def _quality(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Accuracy, likelihood, calibration and the two spread columns of one logit set."""
    nll, ece = _nll_and_ece(logits, targets)
    probability = logits.double().softmax(-1)
    return {
        "accuracy": float(logits.argmax(-1).eq(class_indices(targets)).double().mean()),
        "accuracy_macro": _macro_accuracy(logits, targets),
        "nll": nll,
        "ece": ece,
        "ece_bins": N_ECE_BINS,
        "mean_confidence": float(probability.max(-1).values.mean()),
        "mean_entropy": float(entropy(probability).mean()),
    }


def _macro_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """The mean over classes of the share of each class predicted right, as the training
    loop's `MulticlassAccuracy` logs it (average "macro"): on the validation split of
    the run's own rows it equals that checkpoint's ``val/acc``, which is
    ``val/acc_best``."""
    labels = class_indices(targets).cpu()
    hits = labels[logits.argmax(-1).cpu().eq(labels)]
    classes = logits.shape[-1]
    support = torch.bincount(labels, minlength=classes).double()
    right = torch.bincount(hits, minlength=classes).double()
    return float((right[support > 0] / support[support > 0]).mean())


def _nll_and_ece(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    labels = class_indices(targets)
    accumulator = ECEAccumulator(N_ECE_BINS)
    accumulator.update(logits, labels)
    # Summed then divided, so the accumulation stays float64 whatever the split size.
    total = F.cross_entropy(logits.double(), labels, reduction="sum")
    return float(total) / labels.shape[0], accumulator.compute()


def _scaled(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperatures: tuple[float, float | None],
) -> dict[str, Any]:
    nll, ece = _nll_and_ece(scale(logits, temperatures[0]), targets)
    return {
        "nll_ts": nll,
        "ece_ts": ece,
        "temperature_ts": temperatures[0],
        "temperature_teacher_ts": temperatures[1],
    }
