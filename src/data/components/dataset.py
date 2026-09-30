# In-memory dataset that hands the trainer whole batches of aligned tensors.

from collections.abc import Callable

import torch
from torch.utils.data import IterableDataset


def _worker_slice() -> tuple[int, int]:
    """This worker's index and the worker count, (0, 1) in the main process."""
    info = torch.utils.data.get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


class TensorBatchDataset(IterableDataset):
    """Yield pre-batched rows from aligned tensors held in memory.

    Under the assumption that the whole split fits in RAM, the batching is done here.
    Implement a sampler if otherwise. A batch is a tuple with one entry per tensor: the
    inputs and the labels, and the teacher's logits and hints when a run distils.

    :param tensors: aligned along their first axis, the inputs first.
    :param batch_size: a trailing partial batch is dropped, so ``len(self)`` is exact
        and OneCycleLR can be given a step count.
    :param shuffler: seeded generator for the per-epoch permutation, or None to keep the
        stored order.
    :param max_batches: cap on the batches each worker yields.
    :param transform: online augmentation of the first tensor alone, from a batch to its
        transformed batch of the same shape.
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int,
        shuffler: torch.Generator | None = None,
        max_batches: int | None = None,
        transform: Callable | None = None,
    ):
        if not tensors:
            raise ValueError("At least the inputs and the labels are needed.")
        if any(t.shape[0] != tensors[0].shape[0] for t in tensors):
            raise ValueError("Every tensor must hold the same number of samples.")
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffler = shuffler
        self.max_batches = max_batches
        self.transform = transform
        self.n = tensors[0].shape[0]
        self.num_batches = self.n // batch_size

    def __len__(self):
        if self.max_batches is None:
            return self.num_batches
        return min(self.num_batches, self.max_batches)

    def __iter__(self):
        order = self._order()
        worker_id, num_workers = _worker_slice()
        emitted = 0
        # Workers stride through the batches, so each batch is emitted exactly once.
        for i in range(worker_id, self.num_batches, num_workers):
            if self.max_batches is not None and emitted >= self.max_batches:
                break
            yield self._batch(order[i * self.batch_size : (i + 1) * self.batch_size])
            emitted += 1

    def _order(self) -> torch.Tensor:
        """The row order of one epoch: a fresh permutation, or the stored order."""
        if self.shuffler is None:
            return torch.arange(self.n)
        return torch.randperm(self.n, generator=self.shuffler)

    def _batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One batch, transformed when the dataset carries a transform."""
        x = self.tensors[0][idx]
        x = x if self.transform is None else self.transform(x)
        return (x, *(tensor[idx] for tensor in self.tensors[1:]))
