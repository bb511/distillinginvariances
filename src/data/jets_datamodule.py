"""The hls4ml jets, loaded through the pipeline the record ships with.

The record, https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p,
contains its own processing pipeline that is being applied here. See the docs there for
more details about the dataset and processing pipeline. Its train and validation splits
are pooled and cut into folds; its test split is served untouched.
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

from src.data.components.augmentation import rebase_on_kept_axis
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

    Batches carry all 16 record columns, ``(jets, constituents, features)``; every
    network reduces them to the columns it reads.

    :param revision: string with a HF repository commit pin; None goes to HEAD.
    :param cache_dir: the local cache path ``paths.base_data_dir``;
        the hub snapshot itself is set by ``HF_HUB_CACHE`` env variable.
    :param nconstituents: how many constituents to keep by leading pT for each jet.
    :param regime: ``canonical`` serves the record as it is, ``transformed`` replaces
        every row by one draw from *group*, ``augmented`` is reserved.
    :param group: the symmetry group of the study, whose draws the transformed regime
        and the invariance probe share.
    """

    num_classes = NUM_CLASSES

    def __init__(
        self,
        revision: str | None = None,
        cache_dir: str = "data/hf",
        dataset: str = "jets",
        nconstituents: int = 32,
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
        fold_idx: int = 0,
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

    def prepare_data(self) -> None:
        """Fetch the record and run its pipeline; cached, so reruns are cheap."""
        self._ensure_record().prepare()

    @property
    def input_shape(self) -> tuple[int, int]:
        """The stored width of one jet, (constituents, features); needs the record."""
        return (self.nconstituents, len(self._ensure_record().feature_names))

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
        return torch.cat([x_train, x_val]), torch.cat([y_train, y_val])

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
        self.feature_names = list(self.record.feature_names)
        log.info(f"Record {REPO_ID} at {root}, cache {self.record.cache_folder}")

        return self.record

    def _load(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The jets and one-hot labels of *split*, rebased on the kept axis."""
        data = self._ensure_record().load(split)
        x, scales = data.x.clone(), self.norm_scales
        for chunk in x.split(REBASE_CHUNK):
            chunk.copy_(rebase_on_kept_axis(chunk, self.feature_names, scales))
        return x, data.y


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
