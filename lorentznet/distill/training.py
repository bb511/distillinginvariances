import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
_TENSOR_CACHE = {}

def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1), F.softmax(teacher_logits / temperature, dim=-1), reduction='batchmean') * (temperature * temperature)

def _cached_tensor(path: Path, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if path is None:
        return None
    key = (str(path.resolve()), str(dtype), str(device))
    if key not in _TENSOR_CACHE:
        array = np.load(path, mmap_mode='r')
        tensor = torch.from_numpy(np.array(array, copy=True))
        _TENSOR_CACHE[key] = tensor.to(device=device, dtype=dtype)
    return _TENSOR_CACHE[key]

class TensorBatchLoader:

    def __init__(self, arrays: tuple[torch.Tensor, ...], indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> None:
        self.arrays = arrays
        self.device = arrays[0].device
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.indices), generator=self.generator).to(self.device)
            indices = self.indices[order]
        else:
            indices = self.indices
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start:start + self.batch_size]
            yield tuple((array[selected] if array is not None else None for array in self.arrays))

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

def _scheduler(optimizer: torch.optim.Optimizer, warmup: int, total: int):

    def scale(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = min(1.0, float(epoch - warmup) / float(max(1, total - warmup)))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)
