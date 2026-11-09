import json
import math
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from src.data import top_tagging_datamodule as top_tagging
from src.data.components.augmentation import compute_eta, compute_phi, compute_pt
from src.data.top_tagging_datamodule import TopTaggingDataModule
from src.vocabulary import DATASETS, PROBE_SUBGROUPS
from tests.conftest import (
    TOP_TAGGING_N,
    TOP_TAGGING_PLANTED,
    TOP_TAGGING_SIZES,
    compose_root,
    top_tagging_raw,
    write_top_tagging_raw,
)

SPLITS = {"train": "train.h5", "validation": "val.h5", "test": "test.h5"}
# The sibling's columns, in its order: the imported teachers read them by position.
COLUMNS = ["px", "py", "pz", "e", "pt", "eta", "phi", "etarel", "phirel"]
AXIS = [COLUMNS.index(name) for name in ("etarel", "phirel")]


def _cache(root: Path, prefix: str) -> dict[str, np.ndarray]:
    folder = root / f"nconst_{TOP_TAGGING_N}"
    return {split: np.load(folder / f"{prefix}_{split}.npy") for split in SPLITS}


def _meta(root: Path) -> dict:
    return json.loads((root / f"nconst_{TOP_TAGGING_N}" / "meta.json").read_text())


def _leading(name: str) -> np.ndarray:
    # The leading constituents of every jet of raw file *name*, ranked by p_T one jet at
    # a time: [E, px, py, pz] in GeV, (jets, N, 4), zero-padded.
    p4, _ = top_tagging_raw(name)
    kept = np.zeros((len(p4), TOP_TAGGING_N, 4))
    for jet, constituents in enumerate(p4.astype(np.float64)):
        real = [c for c in constituents if c[0] > 0]
        ranked = sorted(real, key=lambda c: -math.hypot(c[1], c[2]))[:TOP_TAGGING_N]
        kept[jet, : len(ranked)] = ranked
    return kept


def _angles(p4: np.ndarray) -> np.ndarray:
    px, py, pz = (torch.from_numpy(p4[..., i]) for i in (1, 2, 3))
    columns = (compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py))
    return torch.stack(columns, dim=-1).numpy()


def _counts(name: str) -> dict[str, object]:
    p4, flags = top_tagging_raw(name)
    multiplicity = (p4[..., 0] > 0).sum(axis=1)
    return {
        "n_jets": len(p4),
        "class_counts": {"qcd": int((flags == 0).sum()), "top": int(flags.sum())},
        "jets_resorted": len(TOP_TAGGING_PLANTED),
        "jets_truncated": int((multiplicity > TOP_TAGGING_N).sum()),
        "mean_multiplicity": pytest.approx(multiplicity.mean()),
        "max_multiplicity": 20,
    }


def _datamodule(cache: Path, tmp_path: Path, **overrides) -> TopTaggingDataModule:
    settings = {
        "cache_dir": str(cache),
        "nconstituents": TOP_TAGGING_N,
        "transformed_dir": str(tmp_path / "transformed"),
        "transform_kwargs": {"lorentz_eta_max": 1.0},
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "n_folds": 3,
    }
    datamodule = TopTaggingDataModule(**(settings | overrides))
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def test_the_conversion_keeps_the_leading_constituents_at_the_records_scales(
    top_tagging_cache: Path,
) -> None:
    meta = _meta(top_tagging_cache)
    x, y = (_cache(top_tagging_cache, prefix) for prefix in ("x", "y"))
    params = meta["norm_params"]
    scales = np.array([params[name][1] for name in COLUMNS])

    assert (meta["format_version"], meta["nconstituents"]) == (1, TOP_TAGGING_N)
    assert meta["columns"] == COLUMNS and meta["class_names"] == ["qcd", "top"]
    for split, name in SPLITS.items():
        expected, (_, flags) = _leading(name), top_tagging_raw(name)
        real = expected[..., 0] > 0
        gev = x[split].astype(np.float64) * scales
        assert meta["splits"][split] == _counts(name)
        assert x[split].shape == (len(flags), TOP_TAGGING_N, len(COLUMNS))
        assert x[split].dtype == np.float32 and not gev[~real].any()
        # Sorted, truncated, and back in GeV to the float32 rounding of the division.
        assert np.allclose(gev[..., [3, 0, 1, 2]], expected, rtol=1e-6, atol=1e-6)
        assert np.allclose(gev[..., 4:7], _angles(expected), rtol=1e-6, atol=1e-6)
        assert np.array_equal(y[split], np.eye(2, dtype=np.float32)[flags])
    # Fitted on the training split, padded zeros included: a non-negative column
    # reaches 1, any other spans 2.
    for i, name in enumerate(COLUMNS):
        stored = x["train"][..., i]
        positive = name in ("pt", "e")
        span = stored.max() - (0.0 if positive else stored.min())
        assert params[name][0] == ("positive" if positive else "centered")
        assert span == pytest.approx(1.0 if positive else 2.0, rel=1e-6)


def test_the_main_run_and_the_folds_cut_the_records_pool_as_the_sibling_does(
    top_tagging_cache: Path, tmp_path: Path
) -> None:
    datamodule = _datamodule(top_tagging_cache, tmp_path, fold_idx=None)
    x, y = (_cache(top_tagging_cache, prefix) for prefix in ("x", "y"))
    pool_x, pool_y = (np.concatenate([a["train"], a["validation"]]) for a in (x, y))
    n_train = TOP_TAGGING_SIZES["train.h5"]
    params = _meta(top_tagging_cache)["norm_params"]
    served = datamodule.split("pool")[0]
    absolute = [i for i in range(len(COLUMNS)) if i not in AXIS]

    assert datamodule.feature_names == [*COLUMNS, "m2"]
    assert datamodule.input_shape == (TOP_TAGGING_N, len(COLUMNS) + 1)
    assert datamodule.num_classes == 2
    assert datamodule.norm_scales == {n: scale for n, (_, scale) in params.items()}
    # The rebase recomputes the axis columns the conversion stored, and nothing else.
    stored = torch.from_numpy(pool_x)
    assert torch.equal(served[..., absolute], stored[..., absolute])
    assert torch.allclose(served[..., AXIS], stored[..., AXIS], atol=1e-5)
    assert torch.equal(datamodule.split("train")[1], torch.from_numpy(y["train"]))
    assert torch.equal(datamodule.split("fold")[0], served[n_train:])
    assert torch.equal(datamodule.split("test")[1], torch.from_numpy(y["test"]))
    datamodule.set_fold(2)
    folds = KFold(n_splits=3, shuffle=True, random_state=42).split(pool_x)
    train_idx, val_idx = list(folds)[2]
    assert torch.equal(datamodule.split("train")[0], served[train_idx])
    assert torch.equal(datamodule.split("fold")[1], torch.from_numpy(pool_y[val_idx]))


def test_a_missing_record_is_downloaded_and_a_corrupt_download_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, fresh = tmp_path / "source", tmp_path / "fresh"
    write_top_tagging_raw(source)
    md5s = {name: top_tagging._md5(source / name) for name in SPLITS.values()}
    monkeypatch.setattr(top_tagging, "RECORD_URL", f"file://{source}/{{name}}")
    monkeypatch.setattr(top_tagging, "RAW_FILES", md5s)
    datamodule = _datamodule(fresh, tmp_path)
    in_place = _datamodule(tmp_path / "other", tmp_path, raw_dir=str(source))

    assert sorted(path.name for path in (fresh / "raw").iterdir()) == sorted(md5s)
    assert len(datamodule.split("test")[0]) == TOP_TAGGING_SIZES["test.h5"]
    # An intact raw directory elsewhere is read in place and left as it was.
    assert sorted(path.name for path in source.iterdir()) == sorted(md5s)
    assert not (tmp_path / "other" / "raw").exists()
    assert torch.equal(in_place.split("pool")[1], datamodule.split("pool")[1])
    monkeypatch.setattr(top_tagging, "RAW_FILES", {"val.h5": "0" * 32})
    with pytest.raises(ValueError, match="md5"):
        top_tagging.download(tmp_path / "fetched")
    assert not any((tmp_path / "fetched").iterdir())


def test_a_transformed_draw_and_an_augmented_batch_keep_the_padding_at_zero(
    top_tagging_cache: Path, tmp_path: Path
) -> None:
    canonical = _datamodule(top_tagging_cache, tmp_path)
    transformed = _datamodule(top_tagging_cache, tmp_path, regime="transformed")
    augmented = _datamodule(top_tagging_cache, tmp_path, regime="augmented")
    padded = canonical.split("pool")[0][..., 3] == 0
    batch = next(iter(augmented.train_dataloader()))[0]
    reference = next(iter(canonical.train_dataloader()))[0]

    for x in (transformed.split("pool")[0], batch):
        assert not x[(x[..., 3] == 0)].any()
    assert torch.equal(transformed.split("pool")[0][..., 3] == 0, padded)
    assert not torch.allclose(transformed.split("pool")[0], canonical.split("pool")[0])
    assert transformed.transform_matrices("pool").shape == (len(padded), 4, 4)
    assert torch.equal(batch[..., 3] == 0, reference[..., 3] == 0)
    assert not torch.allclose(batch, reference)
    # The mass column is carried through both regimes as the canonical rows hold it.
    masses = canonical.split("pool")[0][..., -1]
    assert torch.equal(transformed.split("pool")[0][..., -1], masses)
    assert torch.equal(batch[..., -1], reference[..., -1])


def test_the_data_config_names_the_dataset_its_multiplicity_and_the_raw_files(
    tmp_path: Path,
) -> None:
    cfg = compose_root("train", "data=top_tagging", "fold=2")
    moved = compose_root("train", "data=top_tagging", f"data.raw_dir={tmp_path}")
    datamodule = hydra.utils.instantiate(cfg.data)

    assert isinstance(datamodule, TopTaggingDataModule)
    assert (datamodule.dataset, datamodule.nconstituents) == ("top_tagging", 100)
    assert (datamodule.n_folds, datamodule.fold_idx) == (5, 1)
    assert datamodule.kfold_seed == 42
    assert datamodule.transformed_dir.endswith("/data/transformed/nconst_100")
    assert datamodule.raw_dir == f"{datamodule.cache_dir}/raw"
    assert datamodule.cache_dir.endswith("/data/top_tagging")
    assert hydra.utils.instantiate(moved.data).raw_dir == str(tmp_path)
    assert "top_tagging" in DATASETS and "top_tagging" in PROBE_SUBGROUPS
