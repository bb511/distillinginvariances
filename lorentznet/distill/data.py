import numpy as np
import torch

def cv_indices(n_samples: int, fold_idx: int, n_folds: int=5, fold_seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(fold_seed)
    permutation = torch.randperm(n_samples, generator=generator).numpy()
    fold_size = n_samples // n_folds
    start = fold_idx * fold_size
    end = start + fold_size
    validation = permutation[start:end]
    training = np.concatenate([permutation[:start], permutation[end:]])
    return (training, validation)
