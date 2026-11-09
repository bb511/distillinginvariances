"""The hls4ml jets, loaded through the pipeline the record ships with.

The record, https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p,
contains its own processing pipeline that is being applied here. See the docs there for
more details about the dataset and processing pipeline. Its train and validation splits
are pooled and cut into folds, or kept as the record split them for the main run; its
test split is served untouched.
"""

import sys
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.components.augmentation import (
    MASS_COLUMN,
    append_mass_column,
    rebase_on_kept_axis,
)
from src.data.components.folds import FoldedDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

REPO_ID = "fastmachinelearning/hls4ml_lhc_jets_150p"

NUM_CLASSES = 5

# Jets per rebasing chunk.
REBASE_CHUNK = 65536

warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


class JetsDataModule(FoldedDataModule):
    """Load the record's jets, transform them once when the regime asks, and fold them.

    Every split is rebased on the axis of the constituents kept as it is loaded: the
    record measures etarel, phirel, deltaR and ptrel from the untruncated jet's axis,
    whereas every transform downstream measures them from the batch it is handed.

    Batches carry all 16 record columns and `MASS_COLUMN`, appended as each split is
    loaded, ``(jets, constituents, 17)``; every network reduces them to the columns it
    reads.

    :param revision: string with a HF repository commit pin; None goes to HEAD.
    :param cache_dir: the local cache path ``paths.base_data_dir``;
        the hub snapshot itself is set by ``HF_HUB_CACHE`` env variable.
    :param nconstituents: how many constituents to keep by leading pT for each jet.
    :param regime: ``canonical`` serves the record as it is, ``transformed`` replaces
        every row by one draw from *group*, ``augmented`` trains on the canonical rows
        transformed afresh per batch and validates and tests as ``transformed`` does.
    :param group: the symmetry group of the study, whose draws the transformed and
        augmented regimes and the invariance probe share.
    :param fold_idx: the fold served, counted from zero; None serves the record's own
        train and validation split, the main run.
    """

    num_classes = NUM_CLASSES

    def __init__(
        self,
        revision: str | None = None,
        cache_dir: str = "data/hf",
        dataset: str = "jets",
        nconstituents: int = 50,
        regime: str = "canonical",
        group: str = "permutation",
        transform_seed: int = 20260820,
        transform_kwargs: dict[str, float | None] | None = None,
        transformed_dir: str = "data/transformed",
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_folds: int = 5,
        kfold_seed: int = 42,
        fold_idx: int | None = 0,
        teacher_cache: str | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            regime=regime,
            group=group,
            transform_seed=transform_seed,
            transform_kwargs=transform_kwargs,
            transformed_dir=transformed_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            n_folds=n_folds,
            kfold_seed=kfold_seed,
            fold_idx=fold_idx,
            teacher_cache=teacher_cache,
        )
        self.revision = revision
        self.cache_dir = cache_dir
        self.nconstituents = nconstituents

        # Filled at instantiation.
        self.record = None
        self.feature_names: list[str] | None = None
        # Filled by load_pool(): where the record's validation rows start in the pool.
        self._n_train: int | None = None

    def prepare_data(self) -> None:
        """Fetch the record and run its pipeline; cached, so reruns are cheap."""
        self._ensure_record().prepare()

    @property
    def input_shape(self) -> tuple[int, int]:
        """The stored width of one jet, (constituents, features); needs the record."""
        return (self.nconstituents, len(self._ensure_record().feature_names) + 1)

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        """Get the normalisation parameters used in normalising the data."""
        return self.record.norm_params if self.record is not None else None

    @property
    def norm_scales(self) -> dict[str, float] | None:
        """Per-feature factors that return the normalised values to GeV."""
        params = self.norm_params
        return (
            None
            if params is None
            else {name: scale for name, (_, scale) in params.items()}
        )

    def load_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The record's training and validation splits, pooled, in that order."""
        x_train, y_train = self._load("train")
        x_val, y_val = self._load("validation")
        self._n_train = len(x_train)
        return torch.cat([x_train, x_val]), torch.cat([y_train, y_val])

    def main_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The record's own split: its training rows, then its validation rows."""
        n_pool = len(self.split("pool")[0])
        return torch.arange(self._n_train), torch.arange(self._n_train, n_pool)

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load("test")

    def _record_root(self) -> str:
        """Where the data is stored on the machine."""
        return snapshot_download(REPO_ID, repo_type="dataset", revision=self.revision)

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        # nconstituents None and <= 0 both mean 'keep them all' here.
        return [
            f"paths.root_dir={root}",
            f"data.nconstituents={self.nconstituents if self.nconstituents else 0}",
            f"paths.base_data_dir={self.cache_dir}",
        ]

    def _ensure_record(self):
        """Fetch the record and instantiate the pipeline it comes with.

        Because the record comes with its own dataloader, we need to insert the root
        path of the that dataloader into sys. That's what 'root' refers to here.'
        """
        if self.record is not None:
            return self.record

        root = self._record_root()
        if root not in sys.path:
            sys.path.insert(0, root)

        self.record = instantiate(
            compose_record_config(root, self._record_overrides(root)).data
        )
        self.feature_names = [*self.record.feature_names, MASS_COLUMN]
        log.info(f"Record {REPO_ID} at {root}, cache {self.record.cache_folder}")

        return self.record

    def _load(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The jets of *split*, rebased on the kept axis and with `MASS_COLUMN`
        appended, and their one-hot labels."""
        data = self._ensure_record().load(split)
        x, scales = data.x.clone(), self.norm_scales
        names = list(self.record.feature_names)
        for chunk in x.split(REBASE_CHUNK):
            chunk.copy_(rebase_on_kept_axis(chunk, names, scales))
        return append_mass_column(x, names, scales), data.y


def compose_record_config(root: str, overrides: list) -> DictConfig:
    """Compose the configuration tree that is shipped with the record."""
    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(Path(root) / "configs"), version_base=None
        ):
            return compose("config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
        if outer is not None:
            GlobalHydra.instance().initialize(outer)
