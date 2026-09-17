from typing import Any, Dict, Optional
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from .hls4ml_dataset import HLS4MLData150

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class PreBatchedDataset(IterableDataset):

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool=False, permute_dynamic: bool=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.permute_dynamic = permute_dynamic

    def __iter__(self):
        n = self.x.size(0)
        if self.shuffle:
            idx = torch.randperm(n, device=self.x.device)
            x, y = (self.x[idx], self.y[idx])
        else:
            x, y = (self.x, self.y)
        for i in range(0, n, self.batch_size):
            xb = x[i:i + self.batch_size]
            yb = y[i:i + self.batch_size]
            if self.permute_dynamic:
                b, m, c = xb.shape
                perm = torch.argsort(torch.rand(b, m, device=xb.device), dim=1)
                xb = torch.gather(xb, 1, perm.unsqueeze(-1).expand(b, m, c))
            yield (xb, yb)

    def __len__(self):
        return (self.x.size(0) + self.batch_size - 1) // self.batch_size

class JetIDDataModule(LightningDataModule):

    def __init__(self, data_dir: str='data/hls4ml/', nconst: int=16, feats: str='ptetaphi', norm: str='robust', batch_size: int=1024, num_workers: int=0, pin_memory: bool=False, n_folds: int=1, fold_idx: int=0, fold_seed: int=0, permute_constituents: bool=False, permute_seed: int=0, permute_dynamic: bool=False, device: Optional[str]=None) -> None:
        super().__init__()
        if permute_constituents and permute_dynamic:
            raise ValueError('permute_constituents (one-shot) and permute_dynamic (per-batch) are mutually exclusive — pick at most one.')
        self.save_hyperparameters(logger=False)
        self._x_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._x_val: Optional[torch.Tensor] = None
        self._y_val: Optional[torch.Tensor] = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 5

    def prepare_data(self) -> None:
        HLS4MLData150(self.hparams.data_dir, self.hparams.nconst, self.hparams.feats, self.hparams.norm, train=True)
        import gc
        gc.collect()
        if self.hparams.n_folds == 1:
            HLS4MLData150(self.hparams.data_dir, self.hparams.nconst, self.hparams.feats, self.hparams.norm, train=False)
        import gc
        gc.collect()

    def setup(self, stage: Optional[str]=None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f'Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).')
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if self._x_train is None:
            if self.trainer is not None:
                device = self.trainer.strategy.root_device
            else:
                device = torch.device(self.hparams.device) if self.hparams.device is not None else _auto_device()
            train_ds = HLS4MLData150(self.hparams.data_dir, self.hparams.nconst, self.hparams.feats, self.hparams.norm, train=True)
            x, y = train_ds.get_torch_tensors()
            del train_ds
            if self.hparams.n_folds > 1:
                n = x.size(0)
                g = torch.Generator()
                g.manual_seed(self.hparams.fold_seed)
                perm = torch.randperm(n, generator=g)
                x, y = (x[perm], y[perm])
                fold_size = n // self.hparams.n_folds
                val_start = self.hparams.fold_idx * fold_size
                val_end = n if self.hparams.fold_idx == self.hparams.n_folds - 1 else val_start + fold_size
                val_mask = torch.zeros(n, dtype=torch.bool)
                val_mask[val_start:val_end] = True
                self._x_train = x[~val_mask].to(device)
                self._y_train = y[~val_mask].to(device)
                self._x_val = x[val_mask].to(device)
                self._y_val = y[val_mask].to(device)
                del x, y
            else:
                self._x_train = x.to(device)
                self._y_train = y.to(device)
                del x, y
                val_ds = HLS4MLData150(self.hparams.data_dir, self.hparams.nconst, self.hparams.feats, self.hparams.norm, train=False)
                x, y = val_ds.get_torch_tensors()
                self._x_val = x.to(device)
                self._y_val = y.to(device)
                del val_ds, x, y
            if self.hparams.permute_constituents:
                g = torch.Generator(device='cpu')
                g.manual_seed(self.hparams.permute_seed)

                def _apply(t: torch.Tensor) -> torch.Tensor:
                    b, n, c = t.shape
                    perm = torch.argsort(torch.rand(b, n, generator=g), dim=1)
                    perm = perm.to(t.device)
                    return torch.gather(t, 1, perm.unsqueeze(-1).expand(b, n, c))
                self._x_train = _apply(self._x_train)
                self._x_val = _apply(self._x_val)

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = PreBatchedDataset(self._x_train, self._y_train, self.batch_size_per_device, shuffle=True, permute_dynamic=self.hparams.permute_dynamic)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self) -> DataLoader[Any]:
        dataset = PreBatchedDataset(self._x_val, self._y_val, self.batch_size_per_device, shuffle=False, permute_dynamic=self.hparams.permute_dynamic)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self) -> DataLoader[Any]:
        return self.val_dataloader()

    def teardown(self, stage: Optional[str]=None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
