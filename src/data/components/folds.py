"""The k-fold split and the datamodule machinery the two datasets share.

A study fold cuts the pool (the record's training rows, plus its validation rows where
the record has them) into a training and a validation half; the record's test split is
never pooled and never folded. Both datamodules serve the same four splits, ``train``
and ``fold`` being the two halves of the current fold, ``pool`` the whole pool in the
stored order the teacher cache is written in, and ``test`` the untouched test set.
"""

from collections.abc import Iterator

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.components.dataset import TensorBatchDataset
from src.data.components.teacher_cache import TeacherCache, read_cache
from src.data.components.transformed import cached_transform, transformed_split
from src.utils.pylogger import RankedLogger
from src.vocabulary import GROUPS, REGIMES

log = RankedLogger(__name__, rank_zero_only=True)

# Loader processes for a pass that never transforms; 0 workers stays 0.
EVAL_WORKERS = 2

SPLITS = ("train", "fold", "pool", "test")


def fold_indices(
    n: int, n_folds: int, fold_idx: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the training and validation row indices of fold *fold_idx*.

    :param n: rows in the pool.
    :param fold_idx: counted from zero, so fold 1 of the study is index 0.
    :param seed: fixes the shuffle, so every stage of a fold sees the same split.
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = list(folds.split(np.arange(n)))[fold_idx]
    return torch.from_numpy(train_idx), torch.from_numpy(val_idx)


class FoldedDataModule(LightningDataModule):
    """Fold, transform and serve a pool and a test set held in memory.

    A subclass supplies ``load_pool()`` and ``load_test()``, each returning the whole
    split as ``(x, y)`` with one-hot labels, and fixes ``num_classes`` and
    ``input_shape``. Everything downstream, the fold split, the transformed regime, the
    loaders and the teacher-cache alignment, is the same for both datasets.

    :param transform_kwargs: the knobs of
        `~src.data.components.augmentation.apply_group` the regime draws with,
        ``lorentz_eta_max``, ``max_shift`` and ``n_swaps``.
    :param teacher_cache: path of the cache written by ``src/cache.py``, or None; when
        set, the training and validation loaders carry the teacher's logits and hints.
    """

    num_classes: int = 0
    input_shape: tuple[int, ...] = ()
    # The jets override these three; MNIST has no named columns and no normalisation.
    feature_names: list[str] | None = None
    norm_scales: dict[str, float] | None = None
    nconstituents: int | None = None

    def __init__(
        self,
        dataset: str,
        regime: str,
        group: str,
        transform_seed: int,
        transform_kwargs: dict[str, float | None] | None,
        transformed_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        n_folds: int,
        kfold_seed: int,
        fold_idx: int,
        teacher_cache: str | None,
    ) -> None:
        super().__init__()
        _validate(regime, group, n_folds)
        self.dataset = dataset
        self.regime = regime
        self.group = group
        self.transform_seed = transform_seed
        self.transform_kwargs = dict(transform_kwargs or {})
        self.transformed_dir = transformed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.n_folds = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx = fold_idx
        self.teacher_cache = teacher_cache

        # Filled by setup(); each split is (x, y, matrices or None).
        self._pool: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        self._test: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        self._folded: dict[str, tuple] = {}
        self._indices: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cache: TeacherCache | None = None

    def load_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        """Load both splits, transform them when the regime asks, and fold the pool."""
        if self._pool is None:
            self._load_splits()

    def set_fold(self, fold_idx: int) -> None:
        """Serve another fold of the same pool and transformed rows."""
        if not 0 <= fold_idx < self.n_folds:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range for n_folds={self.n_folds}."
            )
        self.fold_idx = fold_idx
        self._indices = None
        self._folded = {}

    def transform_matrices(self, split: str) -> torch.Tensor | None:
        """The per-row Lorentz matrices of *split*, shape (rows, 4, 4), or None.

        Only ``lorentz_iso`` in the transformed regime has them; they act on [E, px, py,
        pz] in GeV and are indexed like the rows of the split.
        """
        return self.split(split)[2]

    def eval_batches(
        self, split: str, batch_size: int | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """Walk the rows of *split* once, in stored order, outside any Lightning
        loop."""
        x, y, matrices = self.split(split)
        size = batch_size or self.batch_size
        for start in range(0, len(x), size):
            rows = slice(start, start + size)
            yield x[rows], y[rows], None if matrices is None else matrices[rows]

    def train_class_frequencies(self) -> torch.Tensor:
        """Class shares of the fold's training half, shape (classes,), summing to
        one."""
        y = self.split("train")[1]
        return y.sum(dim=0) / len(y)

    def split(
        self, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The rows of *split* as ``(x, y, matrices or None)``, in stored order."""
        if split == "pool":
            return self._pool
        if split == "test":
            return self._test
        if split not in ("train", "fold"):
            raise ValueError(f"Split must be one of {list(SPLITS)}, got {split!r}.")
        if split not in self._folded:
            self._folded[split] = self._fold_rows(split)
        return self._folded[split]

    def train_dataloader(self) -> DataLoader:
        shuffler = torch.Generator()
        shuffler.manual_seed(torch.initial_seed())
        return self._loader("train", self.num_workers, shuffler=shuffler)

    def val_dataloader(self) -> DataLoader:
        return self._loader("fold", self._eval_workers)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", self._eval_workers)

    @property
    def _eval_workers(self) -> int:
        return min(self.num_workers, EVAL_WORKERS)

    def _load_splits(self) -> None:
        pool_x, pool_y = self.load_pool()
        test_x, test_y = self.load_test()
        pool_x, pool_m = self._maybe_transform("pool", pool_x, self.transform_seed)
        test_x, test_m = self._maybe_transform("test", test_x, self.transform_seed + 1)
        self._pool = (pool_x.float(), pool_y.float(), pool_m)
        self._test = (test_x.float(), test_y.float(), test_m)
        self._cache = self._read_teacher_cache()
        log.info(f"Pool {tuple(pool_x.shape)}, test {tuple(test_x.shape)}")

    def _maybe_transform(
        self, split: str, x: torch.Tensor, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replace *split* by its transformed copy, before any fold split is taken."""
        if self.regime == "augmented":
            raise NotImplementedError(
                "The augmented regime is a reserved hook; nothing implements it yet."
            )
        if self.regime != "transformed":
            return x, None
        key = self._transform_key(split, seed, len(x))
        return cached_transform(self._transform_dir(seed), key, self._builder(x, seed))

    def _transform_dir(self, seed: int) -> str:
        return f"{self.transformed_dir}/{self.dataset}/{self.group}/seed{seed}"

    def _transform_key(self, split: str, seed: int, n: int) -> dict:
        """What a cached draw must agree on to be reused."""
        # An unset n_swaps stays out of the key, so the caches drawn before the key had it
        # (all drawn with it unset) are still reused.
        n_swaps = self.transform_kwargs.get("n_swaps")
        return {
            "group": self.group,
            "seed": seed,
            "n": n,
            "nconstituents": self.nconstituents,
            "lorentz_eta_max": self.transform_kwargs.get("lorentz_eta_max"),
            "max_shift": self.transform_kwargs.get("max_shift"),
            "split": split,
        } | ({} if n_swaps is None else {"n_swaps": n_swaps})

    def _builder(self, x: torch.Tensor, seed: int):
        def build() -> tuple[torch.Tensor, torch.Tensor | None]:
            return transformed_split(
                x,
                self.group,
                seed,
                self.feature_names,
                self.norm_scales,
                **self.transform_kwargs,
            )

        return build

    def _fold_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._indices is None:
            n = len(self._pool[0])
            self._indices = fold_indices(
                n, self.n_folds, self.fold_idx, self.kfold_seed
            )
            log.info(
                f"Fold {self.fold_idx + 1}/{self.n_folds}: "
                f"{len(self._indices[0])} train, {len(self._indices[1])} val rows"
            )
        return self._indices

    def _fold_rows(
        self, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        idx = self._fold_indices()[0 if split == "train" else 1]
        x, y, matrices = self._pool
        return x[idx], y[idx], None if matrices is None else matrices[idx]

    def _read_teacher_cache(self) -> TeacherCache | None:
        if self.teacher_cache is None:
            return None
        expect = {
            "dataset": self.dataset,
            "regime": self.regime,
            "fold": self.fold_idx + 1,
            "n_folds": self.n_folds,
        }
        return read_cache(self.teacher_cache, expect)

    def _teacher_rows(self, split: str) -> tuple[torch.Tensor, ...]:
        """The teacher's logits and hints for *split*, aligned by the fold indices.

        The cache holds the pool in stored order, so the same indices that cut x and y
        cut it too and the alignment needs no key of its own.
        """
        if self._cache is None or split not in ("train", "fold"):
            return ()
        idx = self._fold_indices()[0 if split == "train" else 1]
        rows = (self._cache.pool_logits, self._cache.pool_hints)
        return tuple(torch.from_numpy(row)[idx] for row in rows)

    def _loader(
        self, split: str, num_workers: int, shuffler: torch.Generator | None = None
    ) -> DataLoader:
        """The loader of *split*: the dataset batches, so one element is one batch."""
        x, y, _ = self.split(split)
        dataset = TensorBatchDataset(
            x,
            y,
            *self._teacher_rows(split),
            batch_size=self.batch_size,
            shuffler=shuffler,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: batch[0],
        )


def _validate(regime: str, group: str, n_folds: int) -> None:
    """Reject a configuration the data layer cannot serve, before anything downloads."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}, expected one of {list(REGIMES)}.")
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group!r}, expected one of {list(GROUPS)}.")
    if n_folds < 2:
        raise ValueError(f"KFold needs n_folds >= 2, got {n_folds}.")
