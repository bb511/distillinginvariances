from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class JetIDNpyDataset(Dataset):

    def __init__(self, x_path: str | Path, y_path: str | Path, indices: np.ndarray | None=None) -> None:
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.x = np.load(self.x_path, mmap_mode='r')
        self.y = np.load(self.y_path, mmap_mode='r')
        if self.x.ndim != 3 or self.x.shape[1:] != (32, 4):
            raise ValueError(f'Expected x shape (samples, 32, 4), got {self.x.shape}')
        if self.x.dtype != np.float32:
            raise ValueError(f'Expected float32 four-vectors, got {self.x.dtype}')
        if len(self.x) != len(self.y):
            raise ValueError('Feature and label lengths differ')
        self.indices = np.arange(len(self.x), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))
        y_value = self.y[index]
        if np.ndim(y_value) > 0:
            y_value = int(np.argmax(y_value))
        return (x, torch.tensor(int(y_value), dtype=torch.long))

    def close(self) -> None:
        for array in (self.x, self.y):
            mmap = getattr(array, '_mmap', None)
            if mmap is not None:
                mmap.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

def cv_indices(n_samples: int, fold_idx: int, n_folds: int=5, fold_seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= fold_idx < n_folds:
        raise ValueError(f'fold_idx must be in [0, {n_folds}), got {fold_idx}')
    generator = torch.Generator().manual_seed(fold_seed)
    permutation = torch.randperm(n_samples, generator=generator).numpy()
    fold_size = n_samples // n_folds
    start = fold_idx * fold_size
    end = start + fold_size
    validation = permutation[start:end]
    training = np.concatenate([permutation[:start], permutation[end:]])
    return (training, validation)

def stratified_limit(indices: np.ndarray, labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or limit <= 0 or limit >= len(indices):
        return indices
    label_values = labels
    if label_values.ndim == 2:
        label_values = np.argmax(label_values, axis=1)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    classes, counts = np.unique(label_values[indices], return_counts=True)
    remaining = limit
    for position, (class_id, count) in enumerate(zip(classes, counts)):
        class_indices = indices[label_values[indices] == class_id]
        if position == len(classes) - 1:
            take = remaining
        else:
            take = int(round(limit * count / len(indices)))
            take = min(take, remaining)
        selected.append(rng.choice(class_indices, size=take, replace=False))
        remaining -= take
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
