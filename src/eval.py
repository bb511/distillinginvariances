# Evaluate one trained checkpoint, append its rows to the results table and dump the
# per-sample arrays of its probes.

from pathlib import Path
from typing import Any

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${short_hash:}, ${fold_idx:},
# ${fold_label:} and ${regime_of_teacher:}.
register_resolvers()

from src import vocabulary
from src.metrics.evaluate import (
    Evaluation,
    Probe,
    constant_row,
    evaluate_checkpoint,
    resolve_device,
)
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.evaluation import (
    eval_data,
    eval_datamodule,
    load_module,
    load_teacher,
    probes_of,
    resolve_checkpoint,
    row_identity,
    row_provenance,
)
from src.utils.results import ResultsWriter
from src.utils.samples import write_samples

log = RankedLogger(__name__, rank_zero_only=True)

# Written blank on the constant-predictor row: it is fitted by nothing and trained on nothing.
CONSTANT_BLANKS = (
    "temperature",
    "alpha",
    "beta",
    "discrepancy",
    "lambda_is",
    "lambda_ic",
    "lr",
    "batch_size",
    "epochs_run",
    "guided_index",
    "checkpoint_path",
)
# How many probes, from the cell's own group on, dump their per-sample arrays.
DUMPED_PROBES = {"none": 0, "cell": 1, "all": None}


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one checkpoint on every split of ``cfg.eval_splits``, under every probe.

    The validation fold is evaluated first and its temperatures scale the test logits,
    so the test rows' calibration and sqrt(JSD) after scaling are out of sample.

    :return: every metric, keyed ``<split>/<probe tag>/<metric>``, and the shared
        objects.
    """
    if cfg.stage not in vocabulary.METHOD_OF_STAGE:
        raise ValueError(
            f"The {cfg.stage} stage is never exported; nothing to evaluate."
        )
    manifest, checkpoint = resolve_checkpoint(cfg)
    datamodule, device = eval_datamodule(cfg), resolve_device(cfg)
    module = load_module(cfg.stage, checkpoint, datamodule, device)
    teacher = (
        module if cfg.stage == "teacher" else load_teacher(cfg, datamodule, device)
    )
    probes = probes_of(cfg)
    results = _evaluate_splits(cfg, module, teacher, datamodule, probes)
    base = row_identity(cfg) | row_provenance(cfg, manifest, checkpoint)
    _dump(cfg, base, probes, results)
    _write(cfg, _rows(cfg, base, datamodule, probes, results))
    return _flat(probes, results), {
        "cfg": cfg,
        "datamodule": datamodule,
        "module": module,
    }


def _evaluate_splits(
    cfg: DictConfig,
    module: LightningModule,
    teacher: LightningModule,
    datamodule: LightningDataModule,
    probes: list[Probe],
) -> dict[str, Evaluation]:
    results: dict[str, Evaluation] = {}
    temperatures = None
    for split in _ordered_splits(cfg):
        log.info(f"Evaluating the {split!r} split under {[p.tag for p in probes]}")
        results[split] = evaluate_checkpoint(
            module,
            teacher,
            datamodule,
            split,
            probes,
            cfg.seed,
            temperatures,
            cfg.get("batch_size_eval", 1024),
        )
        if split == "fold":
            temperatures = results[split].temperatures
    return results


def _ordered_splits(cfg: DictConfig) -> list[str]:
    splits = list(cfg.get("eval_splits") or vocabulary.EVAL_SPLITS)
    # The fold's temperature scales the test logits, so the fold comes first.
    return sorted(splits, key=lambda split: split != "fold")


def _dump(
    cfg: DictConfig,
    base: dict[str, Any],
    probes: list[Probe],
    results: dict[str, Evaluation],
) -> None:
    """Write the per-sample arrays of the first ``dump_samples`` probes, one file per
    (split, probe), under the evaluation's run directory."""
    data = eval_data(cfg)
    for split, evaluation in results.items():
        for index, probe in enumerate(probes[: _dumped(cfg)]):
            meta = base | _row_keys(split, data, probe)
            meta |= _draw_keys(cfg, probe, evaluation)
            name = f"{data}_{split}_{probe.tag}.npz"
            path = Path(cfg.paths.output_dir) / "samples" / name
            write_samples(path, evaluation.samples(index), meta)


def _dumped(cfg: DictConfig) -> int | None:
    setting = cfg.get("dump_samples", "cell")
    if setting not in DUMPED_PROBES:
        raise ValueError(
            f"dump_samples must be one of {list(DUMPED_PROBES)}, got {setting!r}."
        )
    return DUMPED_PROBES[setting]


def _row_keys(split: str, data: str, probe: Probe) -> dict[str, Any]:
    return {"eval_split": split, "eval_data": data} | probe.columns()


def _draw_keys(cfg: DictConfig, probe: Probe, evaluation: Evaluation) -> dict[str, Any]:
    """What a dump needs beside its row keys: what fixed its draws (with the seed in the
    base) and the temperatures its sqrt(JSD) takes."""
    student, teacher = evaluation.temperatures
    return {
        "probe_tag": probe.tag,
        "n_draws": probe.n_draws,
        "batch_size_eval": cfg.get("batch_size_eval", 1024),
        "temperature_ts": student,
        "temperature_teacher_ts": teacher,
    }


def _rows(
    cfg: DictConfig,
    base: dict[str, Any],
    datamodule: LightningDataModule,
    probes: list[Probe],
    results: dict[str, Evaluation],
) -> list[dict[str, Any]]:
    """One row per (split, probe), and the constant-predictor floor beside the teacher's
    rows."""
    data = eval_data(cfg)
    rows = [
        base | _row_keys(split, data, probe) | metrics
        for split, evaluation in results.items()
        for probe, metrics in zip(probes, evaluation.metrics)
    ]
    if cfg.stage != "teacher":
        return rows
    keys = {split: _row_keys(split, data, probes[0]) for split in results}
    return rows + [_constant(cfg, base | keys[s], datamodule, s) for s in results]


def _constant(
    cfg: DictConfig, base: dict[str, Any], datamodule: LightningDataModule, split: str
) -> dict[str, Any]:
    """The class-frequency floor, written once per split under the teacher stage and
    filed under the cell's own probe, under which it is trivially invariant."""
    metrics = constant_row(datamodule, split, cfg.get("batch_size_eval", 1024))
    labels = {"method": "constant", "stage": "constant"}
    return base | dict.fromkeys(CONSTANT_BLANKS, "") | labels | metrics


def _write(cfg: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Append the rows, unless this is a throwaway evaluation (``results.path:

    null``).
    """
    path = OmegaConf.select(cfg, "results.path")
    if path is None:
        log.info("results.path is null, so the evaluation writes no row.")
        return
    writer = ResultsWriter(path)
    for row in rows:
        writer.append(row)
    log.info(f"Wrote {len(rows)} rows to {path}")


def _flat(probes: list[Probe], results: dict[str, Evaluation]) -> dict[str, Any]:
    return {
        f"{split}/{probe.tag}/{name}": value
        for split, evaluation in results.items()
        for probe, metrics in zip(probes, evaluation.metrics)
        for name, value in metrics.items()
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoint ``cfg`` points at."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
