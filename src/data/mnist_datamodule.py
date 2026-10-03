"""MNIST from its published record, decoded once to arrays on disk.

The record, https://huggingface.co/datasets/ylecun/mnist, ships two parquet shards whose
image column holds PNG bytes. They are fetched with the hub snapshot like the jets
record, decoded once to uint8 (images, 28, 28) and stored beside the labels, so no
training run depends on Pillow or on the ``datasets`` package. The 60k training split is
the fold pool, of which the main run holds out a seeded 10k, and the 10k test split is
served untouched.
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from src.data.components.folds import FoldedDataModule, holdout_indices
from src.data.components.transformed import savez_atomic
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

NUM_CLASSES = 10
INPUT_SHAPE = (1, 28, 28)

# The pixel mean and standard deviation of the training split, in units of the [0, 1]
# intensity, as every MNIST recipe since LeCun's uses them.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# MNIST has no validation split, so the main run holds one out of the training split with
# the call and the seed the hls4ml record drew its own validation split with,
# train_test_split(..., random_state=42).
MAIN_SPLIT_SEED = 42
MAIN_VALIDATION_ROWS = 10_000


class MNISTDataModule(FoldedDataModule):
    """Serve the MNIST digits, translated once when the regime asks, and folded.

    Batches are ``(images, 1, 28, 28)`` float32, normalised, with one-hot labels.

    :param repo_id: the HF dataset record.
    :param revision: string with a HF repository commit pin; None goes to HEAD.
    :param cache_dir: where the decoded arrays are stored, one npz per record split.
    :param regime:``canonical`` serves the digits as they are, ``transformed`` replaces
        every image by one draw from *group*, ``augmented`` trains on the canonical
        images transformed afresh per batch and validates and tests as ``transformed``
        does.
    """

    num_classes = NUM_CLASSES
    input_shape = INPUT_SHAPE

    def __init__(
        self,
        repo_id: str = "ylecun/mnist",
        revision: str | None = None,
        cache_dir: str = "data/mnist",
        dataset: str = "mnist",
        regime: str = "canonical",
        group: str = "translation",
        transform_seed: int = 20260820,
        transform_kwargs: dict[str, float | None] | None = None,
        transformed_dir: str = "data/transformed",
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
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
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        """Fetch and decode both record splits, unless they are already decoded."""
        if all(self._arrays(split).exists() for split in ("train", "test")):
            return
        root = snapshot_download(
            self.repo_id, repo_type="dataset", revision=self.revision
        )
        for split in ("train", "test"):
            _decode_split(Path(root), split, self._arrays(split))

    def load_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load("train")

    def main_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_pool = len(self.split("pool")[0])
        return holdout_indices(n_pool, MAIN_VALIDATION_ROWS, MAIN_SPLIT_SEED)

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load("test")

    def _arrays(self, split: str) -> Path:
        return Path(self.cache_dir) / f"{split}.npz"

    def _load(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The normalised images (rows, 1, 28, 28) and one-hot labels of *split*."""
        stored = np.load(self._arrays(split))
        x = torch.from_numpy(stored["x"]).float().unsqueeze(1) / 255.0
        labels = torch.from_numpy(stored["y"]).long()
        return (x - MNIST_MEAN) / MNIST_STD, torch.eye(NUM_CLASSES)[labels]


def _decode_split(root: Path, split: str, target: Path) -> None:
    """Decode the parquet shards of *split* to uint8 (images, 28, 28) beside labels."""
    shards = sorted(root.glob(f"mnist/{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"The record at {root} has no {split} parquet shard.")
    table = pq.read_table([str(shard) for shard in shards])
    images = np.stack([_decode(cell) for cell in table.column("image").to_pylist()])
    savez_atomic(target, x=images, y=table.column("label").to_numpy())
    log.info(f"Decoded {len(images)} {split} images to {target}.")


def _decode(cell: dict) -> np.ndarray:
    """One image of the record's struct column, (28, 28) uint8."""
    return np.array(Image.open(io.BytesIO(cell["bytes"])), dtype=np.uint8)
