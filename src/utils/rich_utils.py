# Rich rendering of the composed config tree, and the interactive tag prompt.

from collections.abc import Sequence
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _ordered_fields(cfg: DictConfig, print_order: Sequence[str]) -> list[str]:
    """The top-level fields of *cfg* to render, those of *print_order* first.

    A field of *print_order* that the config does not hold is reported: the order is
    written by hand, so a renamed config group would otherwise drop out of the tree
    unnoticed.
    """
    ordered = []
    for field in print_order:
        if field in cfg:
            ordered.append(field)
        else:
            log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' printing..."
            )
    return ordered + [field for field in cfg if field not in ordered]


def _as_yaml(group: object, resolve: bool) -> str:
    """One config group as the tree shows it, or its plain text when it is not a
    group."""
    if isinstance(group, DictConfig):
        return OmegaConf.to_yaml(group, resolve=resolve)
    return str(group)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "algorithm",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Render the composed config as a Rich tree, one branch per top-level field.

    :param print_order: the fields rendered first; anything else follows in the order
        the config holds it.
    :param resolve: resolve the ``${...}`` interpolations before rendering them.
    :param save_to_file: also write the tree to ``config_tree.log`` in the run
        directory.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for field in _ordered_fields(cfg, print_order):
        branch = tree.add(field, style=style, guide_style=style)
        branch.add(rich.syntax.Syntax(_as_yaml(cfg[field], resolve), "yaml"))
    rich.print(tree)
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def _prompt_for_tags() -> list[str]:
    """Ask for the tags on the command line, splitting the answer on commas.

    :raises ValueError: under a multirun, whose jobs would each stop at the prompt.
    """
    if "id" in HydraConfig().cfg.hydra.job:
        raise ValueError("Specify tags before launching a multirun!")
    log.warning("No tags provided in config. Prompting user to input tags...")
    tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
    return [tag.strip() for tag in tags.split(",") if tag != ""]


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt for tags when the config carries none, and write them back into it.

    :param save_to_file: also write the tags to ``tags.log`` in the run directory.
    """
    if not cfg.get("tags"):
        with open_dict(cfg):
            cfg.tags = _prompt_for_tags()
        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
