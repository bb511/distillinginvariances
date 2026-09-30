"""Startup work, failure handling and metric retrieval, shared by both entry points."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply the optional conveniences configured under ``cfg.extras``.

    Warning filters, tag prompting, the config tree, the float32 matmul precision
    (``high`` trades a little accuracy for tensor cores on Volta and later; ``null``
    leaves torch alone) and the per-process GPU memory cap (a fraction of the card;
    ``null`` leaves torch alone).
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    if cfg.extras.get("matmul_precision"):
        precision = cfg.extras.get("matmul_precision")
        log.info(f"Setting matmul precision! <cfg.extras.matmul_precision={precision}>")
        torch.set_float32_matmul_precision(precision)

    if cfg.extras.get("gpu_memory_fraction") and torch.cuda.is_available():
        fraction = float(cfg.extras.get("gpu_memory_fraction"))
        log.info(f"Capping GPU memory! <cfg.extras.gpu_memory_fraction={fraction}>")
        torch.cuda.set_per_process_memory_fraction(fraction)


def task_wrapper(task_func: Callable) -> Callable:
    """Log the traceback and the output directory when the task fails, then re-raise.

    Re-raising is deliberate: an Optuna trial must fail loudly rather than report a silent
    objective.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return task_func(cfg=cfg)
        except Exception:
            log.exception("")
            raise
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

    return wrap


def _require_metric(metric_dict: dict[str, Any], metric_name: str) -> None:
    """Fail on a metric the run never logged, which a study would otherwise optimise
    blindly.

    :raises Exception: naming both places the name has to match, the module and the
        search config.
    """
    if metric_name in metric_dict:
        return
    raise Exception(
        f"Metric value not found! <metric_name={metric_name}>\n"
        "Make sure metric name logged in LightningModule is correct!\n"
        "Make sure `optimized_metric` name in `hparams_search` config is correct!"
    )


def get_metric_value(
    metric_dict: dict[str, Any], metric_name: str | None
) -> float | None:
    """Read one scalar metric out of a finished run's metrics.

    :param metric_name: the metric to read, ``None`` outside a hyperparameter search.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    _require_metric(metric_dict, metric_name)
    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
