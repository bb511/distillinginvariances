# Profile one checkpoint: the invariance fraction of every layer under every probe, with
# the torus spectrum or the transposition map of each layer.

from pathlib import Path
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
from src.metrics import profile as orbits
from src.metrics.evaluate import Probe, probe_rng, resolve_device
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.evaluation import (
    eval_data,
    eval_datamodule,
    load_module,
    probes_of,
    reinitialise,
    resolve_checkpoint,
    row_identity,
    row_provenance,
)
from src.utils.results import ResultsWriter
from src.utils.samples import write_samples

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def profile(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Profile the checkpoint ``cfg`` names on a seeded subset of ``cfg.eval_split``.

    :return: the invariance fraction of every layer, keyed ``<probe tag>/<layer>``, and
        the shared objects.
    """
    manifest, checkpoint = resolve_checkpoint(cfg)
    datamodule, device = eval_datamodule(cfg), resolve_device(cfg)
    module = load_module(cfg.stage, checkpoint, datamodule, device)
    if cfg.random_init:
        reinitialise(module, cfg.seed)
    x, base = _subset(cfg, datamodule), _base(cfg, manifest, checkpoint)
    rows = [
        row
        for probe in probes_of(cfg)
        for row in _probe_rows(cfg, module, datamodule, x, probe, base)
    ]
    _write(cfg, rows)
    metrics = {f"{r['probe_group']}/{r['layer']}": r["invariance"] for r in rows}
    return metrics, {"cfg": cfg, "module": module}


def _subset(cfg: DictConfig, datamodule: LightningDataModule) -> torch.Tensor:
    x = datamodule.split(cfg.eval_split)[0]
    order = torch.randperm(
        len(x), generator=torch.Generator().manual_seed(cfg.probe_seed)
    )
    return x[order[: cfg.profile.n_samples]]


def _base(cfg: DictConfig, manifest: Any, checkpoint: str) -> dict[str, Any]:
    init = "random" if cfg.random_init else "trained"
    keys = {"eval_split": cfg.eval_split, "eval_data": eval_data(cfg), "init": init}
    return row_identity(cfg) | row_provenance(cfg, manifest, checkpoint) | keys


def _probe_rows(
    cfg: DictConfig,
    module: LightningModule,
    datamodule: LightningDataModule,
    x: torch.Tensor,
    probe: Probe,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per layer under *probe*; its spectra or maps go to one file."""
    exhaustive = probe.group == "translation" and probe.knob is None
    store = _accumulate(cfg, module, datamodule, x, probe, exhaustive)
    rows, arrays = _layer_rows(cfg, store, probe, exhaustive)
    _save(cfg, probe, arrays, base)
    return [base | probe.columns() | row for row in rows]


@torch.no_grad()
def _accumulate(
    cfg: DictConfig,
    module: LightningModule,
    datamodule: LightningDataModule,
    x: torch.Tensor,
    probe: Probe,
    exhaustive: bool,
) -> dict[str, dict[str, Any]]:
    """Per layer, every sample's orbit moments and the batch sums the maps need."""
    store: dict[str, dict[str, Any]] = {}
    device = next(module.parameters()).device
    for index, batch in enumerate(x.split(cfg.profile.batch_size)):
        batch = batch.to(device).float()
        orbit = _orbit(cfg, probe, batch, datamodule, index, exhaustive)
        for name, acts in orbits.orbit_outputs(module, orbit).items():
            _add(store.setdefault(name, {"moments": []}), acts, exhaustive)
        if probe.group == "permutation" and cfg.profile.transpositions:
            _add_transpositions(store, module, batch)
    return store


def _orbit(
    cfg: DictConfig,
    probe: Probe,
    x: torch.Tensor,
    datamodule: LightningDataModule,
    index: int,
    exhaustive: bool,
) -> torch.Tensor:
    """``(samples, draws, ...)``: the whole torus, or n_draws seeded probe draws."""
    if exhaustive:
        return orbits.torus_orbit(x)
    layout = (datamodule.feature_names, datamodule.norm_scales)
    with probe_rng((cfg.probe_seed, cfg.eval_split), probe.tag, index, x.device):
        draws = [
            apply_group(probe.group, x, *layout, **probe.kwargs)[0]
            for _ in range(probe.n_draws)
        ]
    return torch.stack(draws, 1)


def _add(entry: dict[str, Any], acts: torch.Tensor, exhaustive: bool) -> None:
    entry["moments"].append(orbits.sample_moments(acts))
    entry["draws"] = acts.shape[1]
    if exhaustive:
        side = round(acts.shape[1] ** 0.5)
        entry["sums"] = entry.get("sums", 0) + orbits.shared_draw_sums(acts)
        entry["energy"] = entry.get("energy", 0) + orbits.torus_spectrum(acts, side)


def _add_transpositions(
    store: dict[str, dict[str, Any]], module: LightningModule, x: torch.Tensor
) -> None:
    clean = orbits.layer_outputs(module, x)
    moved = orbits.orbit_outputs(module, orbits.transposition_orbits(x))
    for name, acts in moved.items():
        distances, scale = orbits.transposition_distances(acts, clean[name])
        entry = store[name] | {"slots": x.shape[1]}
        entry["moved"] = entry.get("moved", 0) + distances
        store[name] = entry | {"scale": entry.get("scale", 0) + scale}


def _layer_rows(
    cfg: DictConfig, store: dict[str, dict[str, Any]], probe: Probe, exhaustive: bool
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows, arrays = [], {}
    for index, (name, entry) in enumerate(store.items()):
        moments = np.concatenate(entry["moments"])
        stats = orbits.invariance_fraction(
            moments, entry["draws"], exhaustive, cfg.profile.n_boot, cfg.probe_seed
        )
        nv = orbits.normalised_variance(moments, entry["sums"]) if exhaustive else None
        cells = {"layer": name, "layer_index": index, "n_samples": len(moments)}
        cells |= {"n_draws": entry["draws"], "exhaustive": exhaustive, "nv": nv}
        rows.append(cells | stats)
        arrays |= _maps(name, entry, moments)
    return rows, arrays


def _maps(
    name: str, entry: dict[str, Any], moments: np.ndarray
) -> dict[str, np.ndarray]:
    """The layer's torus spectrum (shares of its total variance) or its transposition
    map (in units of the squared distance between two samples)."""
    maps = {}
    if "energy" in entry:
        maps[f"spectrum_{name}"] = orbits.spectrum_shares(entry["energy"], moments)
    if "moved" in entry:
        normalised = entry["moved"] / entry["scale"]
        maps[f"transpositions_{name}"] = orbits.pair_matrix(normalised, entry["slots"])
    return maps


def _save(
    cfg: DictConfig, probe: Probe, arrays: dict[str, np.ndarray], base: dict[str, Any]
) -> None:
    if not arrays:
        return
    name = f"{base['eval_data']}_{cfg.eval_split}_{probe.tag}_{base['init']}.npz"
    path = Path(cfg.paths.output_dir) / "profile" / name
    write_samples(path, arrays, base | probe.columns())


def _write(cfg: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Append the rows, unless this is a throwaway run (``results.path: null``)."""
    path = OmegaConf.select(cfg, "results.path")
    if path is None:
        log.info("results.path is null, so the profile writes no row.")
        return
    writer = ResultsWriter(path, columns=vocabulary.PROFILE_COLUMNS)
    for row in rows:
        writer.append(row)
    log.info(f"Wrote {len(rows)} rows to {path}")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="orbit_profile.yaml"
)
def main(cfg: DictConfig) -> None:
    """Profile the checkpoint ``cfg`` points at."""
    extras(cfg)
    profile(cfg)


if __name__ == "__main__":
    main()
