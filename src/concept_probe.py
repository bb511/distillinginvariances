# Probe one checkpoint's layers for the frame of a group draw and for concepts constant on
# the orbits, and erase the frame with LEACE against random removals of equal rank.

from typing import Any

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate the resolvers.
register_resolvers()

from src import vocabulary
from src.data.components.augmentation import apply_group
from src.metrics.calibration import class_indices
from src.metrics.concepts import concept_labels, frame_labels
from src.metrics.erasure import (
    erase,
    fit_probe,
    forward_from,
    leace,
    r_squared,
    random_removal,
    ridge_predict,
)
from src.metrics.evaluate import Probe, make_probe, probe_rng, resolve_device
from src.metrics.fidelity import top1_agreement
from src.metrics.profile import layer_outputs
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.bootstrap import percentile_interval
from src.utils.evaluation import (
    eval_data,
    eval_datamodule,
    load_module,
    load_teacher,
    reinitialise,
    resolve_checkpoint,
    row_identity,
    row_provenance,
)
from src.utils.results import ResultsWriter

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def concept_probe(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Probe and erase on a seeded subset of ``cfg.eval_split``, transformed once by a
    seeded draw of the cell's group.

    :return: every probe's value, keyed ``<layer>/<kind>/<target>/<statistic>``, and the
        shared objects.
    """
    manifest, checkpoint = resolve_checkpoint(cfg)
    datamodule, device = eval_datamodule(cfg), resolve_device(cfg)
    module = load_module(cfg.stage, checkpoint, datamodule, device)
    teacher = (
        module if cfg.stage == "teacher" else load_teacher(cfg, datamodule, device)
    )
    if cfg.random_init:
        reinitialise(module, cfg.seed)
    probe = make_probe(cfg.data.group, cfg.probe)
    data = _collect(cfg, module, teacher, datamodule, probe)
    rows = _probe_rows(cfg, data) + _erasure_rows(cfg, module, data)
    base = _base(cfg, manifest, checkpoint) | probe.columns()
    _write(cfg, [base | row for row in rows])
    keys = ("layer", "kind", "target", "statistic")
    return {"/".join(str(r[k]) for k in keys): r["value"] for r in rows}, {"cfg": cfg}


def _base(cfg: DictConfig, manifest: Any, checkpoint: str) -> dict[str, Any]:
    init = "random" if cfg.random_init else "trained"
    keys = {"eval_split": cfg.eval_split, "eval_data": eval_data(cfg), "init": init}
    return row_identity(cfg) | row_provenance(cfg, manifest, checkpoint) | keys


@torch.no_grad()
def _collect(
    cfg: DictConfig,
    module: LightningModule,
    teacher: LightningModule,
    datamodule: LightningDataModule,
    probe: Probe,
) -> dict[str, Any]:
    """Every layer at x and at g.x, the teacher's logits at both, the frame labels of g,
    the concepts of x and the labels, over the subset; float64 on the CPU."""
    x, y, _ = datamodule.split(cfg.eval_split)
    generator = torch.Generator().manual_seed(cfg.probe_seed)
    order = torch.randperm(len(x), generator=generator)[: cfg.probing.n_samples]
    device = next(module.parameters()).device
    batches = order.split(cfg.probing.batch_size)
    parts = [
        _batch(
            cfg, (module, teacher), probe, datamodule, x[rows].to(device), y[rows], i
        )
        for i, rows in enumerate(batches)
    ]
    return _joined(parts)


def _batch(
    cfg: DictConfig,
    models: tuple[LightningModule, LightningModule],
    probe: Probe,
    datamodule: LightningDataModule,
    x: torch.Tensor,
    y: torch.Tensor,
    index: int,
) -> dict[str, Any]:
    (module, teacher), x = models, x.float()
    layout = (datamodule.feature_names, datamodule.norm_scales)
    with probe_rng((cfg.probe_seed, cfg.eval_split), probe.tag, index, x.device):
        x_t, params = apply_group(probe.group, x, *layout, **probe.kwargs)
    concepts = concept_labels(cfg.data.dataset, x, layout)
    parts = {"acts": layer_outputs(module, x), "acts_t": layer_outputs(module, x_t)}
    parts |= {"teacher": teacher(x), "teacher_t": teacher(x_t), "concepts": concepts}
    parts |= {"frames": frame_labels(probe.group, params, x_t, layout)}
    return _cpu(parts | {"targets": class_indices(y)})


def _cpu(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _cpu(v) if isinstance(v, dict) else v.detach().cpu().double()
        for k, v in parts.items()
    }


def _joined(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        k: _joined([p[k] for p in parts])
        if isinstance(v, dict)
        else torch.cat([p[k] for p in parts])
        for k, v in parts[0].items()
    }


def _probe_rows(cfg: DictConfig, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Per layer, the frame probe on g.x and one probe per concept on x."""
    cut = int(cfg.probing.train_share * len(data["targets"]))
    rows = []
    for index, (name, acts) in enumerate(data["acts"].items()):
        layer = {"layer": name, "layer_index": index, "width": acts.shape[1]}
        frame = _scored(cfg, data["acts_t"][name], data["frames"], cut)
        rows.append(layer | frame | {"kind": "frame", "target": "frame"})
        for concept, values in data["concepts"].items():
            scored = _scored(cfg, acts, values.unsqueeze(1), cut)
            rows.append(layer | scored | {"kind": "concept", "target": concept})
    return rows


def _scored(
    cfg: DictConfig, x: torch.Tensor, z: torch.Tensor, cut: int
) -> dict[str, Any]:
    """Held-out R^2 of a ridge probe fitted on the rows before *cut*, with a percentile
    95 % interval from resampling the held-out rows."""
    predicted = ridge_predict(fit_probe(x[:cut], z[:cut]), x[cut:])
    generator = torch.Generator().manual_seed(cfg.probe_seed)
    n = len(predicted)
    replicates = [
        float(r_squared(z[cut:][rows], predicted[rows]))
        for rows in (
            torch.randint(n, (n,), generator=generator)
            for _ in range(cfg.probing.n_boot)
        )
    ]
    low, high = percentile_interval(np.array(replicates))
    value = float(r_squared(z[cut:], predicted))
    sizes = {"n_train": cut, "n_test": n, "rank": z.shape[1]}
    return {"statistic": "r2", "value": value, "ci_low": low, "ci_high": high} | sizes


def _erasure_rows(
    cfg: DictConfig, module: LightningModule, data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per hidden layer of a student, the effect of erasing the frame there."""
    if not hasattr(module.net, "forward_with_hidden"):
        return []
    hidden = [name for name in data["acts"] if name.startswith("hidden_")]
    return [
        row
        for index, name in enumerate(hidden)
        for row in _erasure(cfg, module.net, data, name, index)
    ]


def _erasure(
    cfg: DictConfig, net: torch.nn.Module, data: dict[str, Any], name: str, index: int
) -> list[dict[str, Any]]:
    """LEACE fitted on the training rows at g.x, applied at x and g.x of the held-out
    rows; the controls remove random subspaces of the frame's rank."""
    cut = int(cfg.probing.train_share * len(data["targets"]))
    train, frames = data["acts_t"][name][:cut], data["frames"]
    eraser = leace(train, frames[:cut])
    effect = _effect(net, data, name, index, cut, eraser)
    generator = torch.Generator().manual_seed(cfg.probe_seed)
    controls = [
        _effect(
            net,
            data,
            name,
            index,
            cut,
            random_removal(train, frames.shape[1], generator),
        )
        for _ in range(cfg.probing.n_controls)
    ]
    residual = _scored(cfg, erase(data["acts_t"][name], eraser), frames, cut)
    layer = {"layer": name, "layer_index": index, "width": train.shape[1]}
    cells = {"kind": "erasure", "target": "frame", "rank": frames.shape[1]}
    rows = [_effect_row(s, effect[s], [c[s] for c in controls]) for s in effect]
    return [
        layer | cells | row for row in rows + [residual | {"statistic": "r2_after"}]
    ]


def _effect(
    net: torch.nn.Module,
    data: dict[str, Any],
    name: str,
    index: int,
    cut: int,
    eraser: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Changes in accuracy and fidelity at g.x and in top-1 invariance under erasure."""
    erased = [
        forward_from(net, index, erase(data[key][name][cut:], eraser)).double().cpu()
        for key in ("acts", "acts_t")
    ]
    before = [data[key]["logits"][cut:] for key in ("acts", "acts_t")]
    targets, teacher = data["targets"][cut:].long(), data["teacher_t"][cut:]
    accuracy = [
        float(z.argmax(-1).eq(targets).double().mean()) for z in (erased[1], before[1])
    ]
    fidelity = [
        float(top1_agreement(z, teacher).mean()) for z in (erased[1], before[1])
    ]
    invariance = [float(top1_agreement(*pair).mean()) for pair in (erased, before)]
    changes = (accuracy, fidelity, invariance)
    names = ("d_accuracy_g", "d_fidelity_g", "d_invariance")
    return {n: after - prior for n, (after, prior) in zip(names, changes)}


def _effect_row(statistic: str, value: float, controls: list[float]) -> dict[str, Any]:
    low, high = np.quantile(controls, [0.025, 0.975])
    control = {"control_mean": float(np.mean(controls)), "control_low": float(low)}
    return {
        "statistic": statistic,
        "value": value,
        "control_high": float(high),
    } | control


def _write(cfg: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Append the rows, unless this is a throwaway run (``results.path: null``)."""
    path = OmegaConf.select(cfg, "results.path")
    if path is None:
        log.info("results.path is null, so the probes write no row.")
        return
    writer = ResultsWriter(path, columns=vocabulary.PROBE_COLUMNS)
    for row in rows:
        writer.append(row)
    log.info(f"Wrote {len(rows)} rows to {path}")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="concept_probe.yaml"
)
def main(cfg: DictConfig) -> None:
    """Probe the checkpoint ``cfg`` points at."""
    extras(cfg)
    concept_probe(cfg)


if __name__ == "__main__":
    main()
