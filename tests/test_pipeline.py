import csv
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.cache import cache_teacher
from src.eval import evaluate
from src.train import train
from src.vocabulary import PROVENANCE_COLUMNS, RESULT_COLUMNS
from tests.conftest import compose_root, needs_record, point_into
from tests.test_train import one_batch

pytestmark = [needs_record, pytest.mark.slow]

CHEAP = (
    "data.n_folds=2",
    "data.num_workers=0",
    "data.pin_memory=false",
    "extras.print_config=false",
    "extras.enforce_tags=false",
)


def _upstream(tmp_path: Path) -> tuple[str, ...]:
    return (
        f"teacher_run_dir={tmp_path / 'teacher'}",
        f"hint_run_dir={tmp_path / 'hint'}",
        f"teacher_cache={tmp_path / 'cache.npz'}",
    )


def _compose(config: str, tmp_path: Path, stage: str, *overrides: str) -> DictConfig:
    cfg = compose_root(
        config,
        "experiment=mnist_cnn",
        f"method={stage}",
        "trainer=cpu",
        *_upstream(tmp_path),
        *CHEAP,
        *overrides,
    )
    HydraConfig().set_config(point_into(cfg, tmp_path / stage))
    return cfg


def test_the_stages_chain_into_two_result_rows_per_checkpoint(tmp_path: Path) -> None:
    for stage, overrides in (
        ("teacher", ()),
        ("hint", ()),
        ("kd_hint", ("algorithm.temperature=2",)),
    ):
        if stage == "hint":
            cache_teacher(_compose("cache.yaml", tmp_path, "cache"))
        train(
            one_batch(
                "mnist_cnn",
                tmp_path / stage,
                f"method={stage}",
                *_upstream(tmp_path),
                *overrides,
            )
        )
    table = tmp_path / "results.csv"
    manifest = tmp_path / "kd_hint" / "checkpoint_manifest.json"

    evaluate(
        _compose(
            "eval.yaml",
            tmp_path,
            "kd_hint",
            "algorithm.temperature=2",
            f"checkpoint_manifest_path={manifest}",
            f"results.path={table}",
            "probe.n_draws=2",
            "batch_size_eval=256",
        )
    )
    with table.open(newline="") as handle:
        header, *rows = list(csv.reader(handle))

    assert header == list(RESULT_COLUMNS + PROVENANCE_COLUMNS)
    assert [row[header.index("eval_split")] for row in rows] == ["fold", "test"]
    assert {row[header.index("method")] for row in rows} == {"hint"}
