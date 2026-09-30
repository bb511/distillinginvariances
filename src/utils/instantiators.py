# The callbacks and loggers a composed config names, instantiated.


import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _target_entries(cfg: DictConfig | None, kind: str) -> list[DictConfig]:
    """The entries of one config node that carry a ``_target_``.

    :param kind: names the node in the log messages, "callback" or "logger".
    :raises TypeError: when the node is not a group of configs.
    """
    if not cfg:
        log.warning(f"No {kind} configs found! Skipping...")
        return []
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"The {kind} config must be a DictConfig, not a {type(cfg).__name__}!"
        )
    return [
        entry
        for entry in cfg.values()
        if isinstance(entry, DictConfig) and "_target_" in entry
    ]


def _instantiate_targets(cfg: DictConfig | None, kind: str) -> list[Callback | Logger]:
    """Instantiate every entry of one config node that carries a ``_target_``."""
    objects = []
    for entry in _target_entries(cfg, kind):
        log.info(f"Instantiating {kind} <{entry._target_}>")
        objects.append(hydra.utils.instantiate(entry))
    return objects


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    """Instantiate every entry of the ``callbacks`` node that carries a ``_target_``."""
    return _instantiate_targets(callbacks_cfg, "callback")


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiate every entry of the ``logger`` node that carries a ``_target_``.

    ``logger=none`` composes an empty node, and so yields no loggers at all.
    """
    return _instantiate_targets(logger_cfg, "logger")
