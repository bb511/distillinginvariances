# Measure how far one probe moves a cell's samples off their support, as the held-out AUC
# of a classifier that tells the student's inputs x from their first probe draw g.x.

from typing import Any

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import fold_value, register_resolvers

# Before Hydra composes: the configs interpolate ${short_hash:}, ${fold_idx:},
# ${fold_label:} and ${regime_of_teacher:}.
register_resolvers()

from src import vocabulary
from src.algorithms import resolve_feature_indices
from src.data.components.augmentation import apply_group
from src.metrics.coverage import auc_interval, fit_classifier, scores
from src.metrics.evaluate import Probe, make_probe, probe_rng, resolve_device
from src.metrics.magnitude import bin_masks, magnitudes, with_derived
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.evaluation import eval_data, eval_datamodule
from src.utils.results import ResultsWriter

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def coverage(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the classifier on the validation fold's pairs and score the test pairs.

    :return: the AUC of every bin, keyed ``<magnitude>/<low>`` (``all`` for the whole
        draw), and the shared objects.
    """
    datamodule, device = eval_datamodule(cfg), resolve_device(cfg)
    probe = make_probe(cfg.get("probe_group") or cfg.data.group, cfg.probe)
    train = _pairs(cfg, datamodule, probe, "fold", device)
    test = _pairs(cfg, datamodule, probe, "test", device)
    settings = OmegaConf.to_container(cfg.classifier)
    net = fit_classifier(train["x"], train["x_t"], **settings, seed=cfg.seed)
    rows = _rows(cfg, probe, test, net)
    _write(cfg, rows)
    metrics = {f"{r['magnitude'] or 'all'}/{r['bin_low']}": r["auc"] for r in rows}
    return metrics, {"cfg": cfg, "datamodule": datamodule}


def _features(cfg: DictConfig, datamodule: LightningDataModule) -> list[int] | None:
    """The student's input columns; None where it reads the whole sample."""
    required = cfg.student_net.get("required_features")
    if not required:
        return None
    return resolve_feature_indices(list(required), list(datamodule.feature_names))


@torch.no_grad()
def _pairs(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    probe: Probe,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    """The student's view of every row of *split* and of its first probe draw, with the
    draw's magnitudes; the draws are those of an evaluation's dump of the same probe."""
    columns = _features(cfg, datamodule)
    layout = (datamodule.feature_names, datamodule.norm_scales)
    parts = []
    batches = datamodule.eval_batches(split, cfg.get("batch_size_eval", 1024))
    for index, (x, _, _) in enumerate(batches):
        x = x.to(device).float()
        x_t, params = _first_draw(probe, x, layout, ((cfg.seed, split), index))
        sizes = magnitudes(probe.group, params, x)
        parts.append((_view(x, columns), _view(x_t, columns), sizes))
    return _joined(parts)


def _first_draw(
    probe: Probe, x: torch.Tensor, layout: tuple, key: tuple[tuple[int, str], int]
) -> tuple[torch.Tensor, torch.Tensor]:
    stream, batch = key
    with probe_rng(stream, probe.tag, batch, x.device):
        return apply_group(probe.group, x, *layout, **probe.kwargs)


def _view(x: torch.Tensor, columns: list[int] | None) -> torch.Tensor:
    return (x if columns is None else x[..., columns]).flatten(1)


def _joined(parts: list[tuple]) -> dict[str, Any]:
    views, views_t, sizes = zip(*parts)
    joined = {n: torch.cat([s[n] for s in sizes]).cpu().numpy() for n in sizes[0]}
    return {"x": torch.cat(views), "x_t": torch.cat(views_t), "sizes": joined}


def _rows(
    cfg: DictConfig, probe: Probe, test: dict[str, Any], net: torch.nn.Module
) -> list[dict[str, Any]]:
    """One row for the whole draw, then one per requested bin that holds a pair."""
    negative, positive = scores(net, test["x"]), scores(net, test["x_t"])
    groups = [("", "", "", np.ones(len(negative), dtype=bool))]
    sizes = with_derived(test["sizes"])
    for name, edges in (cfg.get("coverage_bins") or {}).items():
        groups += [(name, *mask) for mask in bin_masks(sizes[name], edges)]
    identity = _identity(cfg, probe)
    return [
        identity | _bin_row(cfg, group, negative, positive)
        for group in groups
        if group[-1].any()
    ]


def _bin_row(
    cfg: DictConfig, group: tuple, negative: torch.Tensor, positive: torch.Tensor
) -> dict[str, Any]:
    name, low, high, mask = group
    rows = torch.from_numpy(mask).to(negative.device)
    value, ci_low, ci_high = auc_interval(
        negative[rows], positive[rows], cfg.n_boot, cfg.seed
    )
    cells = {"magnitude": name, "bin_low": low, "bin_high": high}
    cells |= {"n_pairs": int(mask.sum()), "auc": value}
    return cells | {"ci_low": ci_low, "ci_high": ci_high}


def _identity(cfg: DictConfig, probe: Probe) -> dict[str, Any]:
    required = cfg.student_net.get("required_features")
    return {
        "dataset": cfg.data.dataset,
        "teacher": cfg.teacher,
        "group": cfg.data.group,
        "features": ",".join(required) if required else "pixels",
        "eval_data": eval_data(cfg),
        "fold": fold_value(cfg.fold),
        "n_boot": cfg.n_boot,
        "seed": cfg.seed,
        "classifier_hidden": "x".join(str(width) for width in cfg.classifier.hidden),
        "classifier_epochs": cfg.classifier.epochs,
    } | probe.columns()


def _write(cfg: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Append the rows, unless this is a throwaway run (``results.path: null``)."""
    path = OmegaConf.select(cfg, "results.path")
    if path is None:
        log.info("results.path is null, so the coverage writes no row.")
        return
    writer = ResultsWriter(path, columns=vocabulary.COVERAGE_COLUMNS)
    for row in rows:
        writer.append(row)
    log.info(f"Wrote {len(rows)} rows to {path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="coverage.yaml")
def main(cfg: DictConfig) -> None:
    """Measure the coverage of the probe ``cfg`` names."""
    extras(cfg)
    coverage(cfg)


if __name__ == "__main__":
    main()
