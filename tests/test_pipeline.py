import csv
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.cache import cache_teacher
from src.eval import evaluate
from src.train import train
from src.utils.samples import read_samples
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


@pytest.mark.parametrize(("fold", "exported"), [("null", "main"), ("1", "1")])
def test_the_stages_chain_into_two_result_rows_per_checkpoint(
    fold: str, exported: str, tmp_path: Path
) -> None:
    for stage, overrides in (
        ("teacher", ()),
        ("hint", ()),
        ("kd_hint", ("algorithm.temperature=2",)),
    ):
        if stage == "hint":
            cache_teacher(_compose("cache.yaml", tmp_path, "cache", f"fold={fold}"))
        train(
            one_batch(
                "mnist_cnn",
                tmp_path / stage,
                f"method={stage}",
                f"fold={fold}",
                *_upstream(tmp_path),
                *overrides,
            )
        )
    table = tmp_path / "results.csv"
    manifest = tmp_path / "kd_hint" / "checkpoint_manifest.json"
    student = (
        f"fold={fold}",
        "algorithm.temperature=2",
        f"checkpoint_manifest_path={manifest}",
        "batch_size_eval=256",
    )

    evaluate(
        _compose(
            "eval.yaml",
            tmp_path,
            "kd_hint",
            *student,
            f"results.path={table}",
            "probe.n_draws=2",
        )
    )
    evaluate(
        _compose(
            "eval.yaml",
            tmp_path,
            "kd_hint",
            *student,
            f"results.path={tmp_path / 'transformed.csv'}",
            "eval_data=transformed",
            "eval_splits=[test]",
            f"data.transformed_dir={tmp_path / 'transformed'}",
        )
    )
    header, rows = _table(table)
    _, transformed = _table(tmp_path / "transformed.csv")
    arrays, meta = read_samples(
        tmp_path / "kd_hint" / "samples" / "canonical_test_translation_shiftall.npz"
    )

    assert header == list(RESULT_COLUMNS + PROVENANCE_COLUMNS)
    assert [row["eval_split"] for row in rows] == ["fold", "test"]
    assert {row["method"] for row in rows} == {"hint"}
    assert {row["fold"] for row in rows} == {exported}
    assert {row["probe_group"] for row in rows} == {"translation"}
    assert [(r["regime"], r["eval_data"]) for r in transformed] == [
        ("canonical", "transformed")
    ]
    assert arrays["student_g"].shape == (10_000, 10)
    assert set(arrays) >= {"teacher_g", "mag_shift_row", "mag_shift_col", "targets"}
    assert (meta["eval_split"], meta["probe_tag"]) == ("test", "translation_shiftall")


def _table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        header, *rows = list(csv.reader(handle))
    return header, [dict(zip(header, row)) for row in rows]
