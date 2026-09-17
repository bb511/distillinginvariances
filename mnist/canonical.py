import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassCalibrationError

class _GPUBatchLoader:

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = self

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.x.size(0)
        if self.shuffle:
            idx = torch.randperm(n, device=self.x.device)
        else:
            idx = torch.arange(n, device=self.x.device)
        bs = self.batch_size
        for i in range(0, n, bs):
            sel = idx[i:i + bs]
            yield (self.x[sel], self.y[sel])

    def __len__(self) -> int:
        return (self.x.size(0) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.x[idx], self.y[idx])

class MNISTInMemoryGPUDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int=1024, train_val_test_split: Tuple[int, int, int]=(55000, 5000, 10000), mean: float=0.1307, std: float=0.3081, device: str='auto', split_seed: int=42, n_folds: int=0, fold_idx: int=0) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._train_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._val_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._test_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _resolve_device(self) -> torch.device:
        d = self.hparams.device
        if d == 'auto':
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return torch.device(d)

    def prepare_data(self) -> None:
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str]=None) -> None:
        if self._train_xy is not None:
            return
        train_raw = MNIST(self.hparams.data_dir, train=True, download=False)
        test_raw = MNIST(self.hparams.data_dir, train=False, download=False)
        train_x = train_raw.data.float().unsqueeze(1) / 255.0
        test_x = test_raw.data.float().unsqueeze(1) / 255.0
        train_x = (train_x - self.hparams.mean) / self.hparams.std
        test_x = (test_x - self.hparams.mean) / self.hparams.std
        train_y = train_raw.targets.long()
        test_y = test_raw.targets.long()
        all_x = torch.cat([train_x, test_x], dim=0)
        all_y = torch.cat([train_y, test_y], dim=0)
        device = self._resolve_device()
        all_x = all_x.to(device, non_blocking=False).contiguous()
        all_y = all_y.to(device, non_blocking=False).contiguous()
        n_train, n_val, n_test = self.hparams.train_val_test_split
        assert n_train + n_val + n_test <= all_x.size(0)
        g = torch.Generator(device='cpu').manual_seed(self.hparams.split_seed)
        idx = torch.randperm(all_x.size(0), generator=g)
        n_folds = int(self.hparams.n_folds)
        if n_folds > 0:
            fold_idx = int(self.hparams.fold_idx)
            if not 0 <= fold_idx < n_folds:
                raise ValueError(f'fold_idx={fold_idx} out of range for n_folds={n_folds}')
            n_pool = n_train + n_val
            pool = idx[:n_pool]
            te = idx[n_pool:n_pool + n_test]
            chunk = n_pool // n_folds
            start = fold_idx * chunk
            end = (fold_idx + 1) * chunk if fold_idx < n_folds - 1 else n_pool
            va = pool[start:end]
            tr = torch.cat([pool[:start], pool[end:]])
        else:
            tr = idx[:n_train]
            va = idx[n_train:n_train + n_val]
            te = idx[n_train + n_val:n_train + n_val + n_test]
        self._train_xy = (all_x[tr].contiguous(), all_y[tr].contiguous())
        self._val_xy = (all_x[va].contiguous(), all_y[va].contiguous())
        self._test_xy = (all_x[te].contiguous(), all_y[te].contiguous())

    def _loader(self, xy: Tuple[torch.Tensor, torch.Tensor], shuffle: bool) -> _GPUBatchLoader:
        return _GPUBatchLoader(x=xy[0], y=xy[1], batch_size=self.hparams.batch_size, shuffle=shuffle)

    def train_dataloader(self) -> _GPUBatchLoader:
        return self._loader(self._train_xy, shuffle=True)

    def val_dataloader(self) -> _GPUBatchLoader:
        return self._loader(self._val_xy, shuffle=False)

    def test_dataloader(self) -> _GPUBatchLoader:
        return self._loader(self._test_xy, shuffle=False)

    def teardown(self, stage: Optional[str]=None) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

def activ_string_to_torch(activ: str):
    activations = {'relu': lambda: nn.ReLU(inplace=True), 'tanh': lambda: nn.Tanh(), 'sigmoid': lambda: nn.Sigmoid(), 'leaky_relu': lambda: nn.LeakyReLU()}
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(f'Activation {activ} not implemented.')
    return activation

class CNNInvariant(nn.Module):

    def __init__(self, input_channels: int, phi_channels: list, rho_layers: list, activ: str, output_dim: int, padding_mode: str='circular', kernel_size: int=3, use_bn: bool=False):
        super().__init__()
        self.activ = activ
        self.input_channels = input_channels
        self.phi_channels = list(phi_channels)
        self.rho_layers = list(rho_layers)
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self._phi_block = 3 if use_bn else 2
        self.phi = self._construct_phi()
        self.rho = self._construct_rho()

    def _construct_phi(self) -> nn.Sequential:
        phi = nn.Sequential()
        channels = [self.input_channels] + self.phi_channels
        pad = self.kernel_size // 2
        for nlayer in range(len(channels) - 1):
            phi.append(nn.Conv2d(channels[nlayer], channels[nlayer + 1], kernel_size=self.kernel_size, padding=pad, padding_mode=self.padding_mode, stride=1))
            if self.use_bn:
                phi.append(nn.BatchNorm2d(channels[nlayer + 1]))
            phi.append(activ_string_to_torch(self.activ))
        return phi

    def _construct_rho(self) -> nn.Sequential:
        rho = nn.Sequential()
        layers = [self.phi_channels[-1]] + self.rho_layers + [self.output_dim]
        for nlayer in range(len(layers) - 1):
            rho.append(nn.Linear(layers[nlayer], layers[nlayer + 1]))
            if nlayer == len(layers) - 2:
                break
            rho.append(activ_string_to_torch(self.activ))
        return rho

    def _aggregate(self, feature_map: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

    def forward(self, x):
        phi_out = self.phi(x)
        agg = self._aggregate(phi_out)
        return self.rho(agg)

    def forward_with_hint(self, x):
        phi_out = self.phi(x)
        hint = self._aggregate(phi_out)
        logits = self.rho(hint)
        return (logits, hint)

    def forward_with_phi_hint(self, x, phi_depth: int=3):
        h = x
        hint = None
        target_idx = phi_depth * self._phi_block - 1
        for i, layer in enumerate(self.phi):
            h = layer(h)
            if i == target_idx:
                hint = self._aggregate(h)
        full_agg = self._aggregate(h)
        logits = self.rho(full_agg)
        return (logits, hint)

class MLPBasic(nn.Module):

    def __init__(self, input_dim: int, layers: list, output_dim: int, activ: str):
        super().__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.layers = list(layers)
        self.output_dim = output_dim
        self._construct_mlp()

    def _construct_mlp(self):
        all_layers = [self.input_dim] + self.layers + [self.output_dim]
        self.mlp = nn.Sequential()
        for nlayer in range(len(all_layers) - 1):
            self.mlp.add_module(f'linear_{nlayer}', nn.Linear(all_layers[nlayer], all_layers[nlayer + 1]))
            if nlayer < len(all_layers) - 2:
                self.mlp.add_module(f'activation_{nlayer}', activ_string_to_torch(self.activ))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

    def forward_with_guided(self, x, guided_idx: int=3):
        x = torch.flatten(x, start_dim=1)
        guided = None
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == guided_idx:
                guided = x
        return (x, guided)

    def forward_with_two_guided(self, x, guided_idx1: int, guided_idx2: int):
        x = torch.flatten(x, start_dim=1)
        guided1 = guided2 = None
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == guided_idx1:
                guided1 = x
            if i == guided_idx2:
                guided2 = x
        return (x, guided1, guided2)

class MNISTLitModule(LightningModule):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, compile: bool) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return (loss, preds, y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == 'fit':
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss', 'interval': 'epoch', 'frequency': 1}}
        return {'optimizer': optimizer}

class FitNetsMNISTModule(LightningModule):

    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, teacher_ckpt: str, stage: str='hint', temperature: float=4.0, alpha: float=0.0, beta: float=0.0, guided_idx: int=5, hint_depth: int=3, freeze_guided: bool=False, student_ckpt: Optional[str]=None, num_classes: int=10, compile: bool=False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['teacher', 'student'])
        self.teacher = teacher
        self.student = student
        ckpt = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        teacher_weights = {k[4:]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
        self.teacher.load_state_dict(teacher_weights)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        all_layers = [student.input_dim] + student.layers + [student.output_dim]
        guided_dim = all_layers[(guided_idx + 2) // 2]
        hint_dim = teacher.phi_channels[-1]
        if guided_dim != hint_dim:
            self.regressor = nn.Linear(guided_dim, hint_dim)
        else:
            self.regressor = nn.Identity()
        if stage == 'kd' and student_ckpt is not None:
            ckpt = torch.load(student_ckpt, map_location='cpu', weights_only=False)
            student_weights = {k[8:]: v for k, v in ckpt['state_dict'].items() if k.startswith('student.')}
            self.student.load_state_dict(student_weights)
            reg_weights = {k[10:]: v for k, v in ckpt['state_dict'].items() if k.startswith('regressor.')}
            if reg_weights:
                self.regressor.load_state_dict(reg_weights)
        if stage == 'kd' and freeze_guided:
            for i, layer in enumerate(self.student.mlp):
                if i <= guided_idx:
                    for p in layer.parameters():
                        p.requires_grad_(False)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.register_buffer('hint_std', torch.ones(hint_dim))
        self._hint_stats_ready = False
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        if stage == 'kd':
            self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
            self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
            self.val_acc_best = MaxMetric()
            self.val_ti_agree = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    def _hint_mse(self, guided: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        scale = self.hint_std.view(1, -1)
        return F.mse_loss(guided / scale, hint / scale)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        if self.hparams.stage == 'kd':
            self.val_acc.reset()
            self.val_acc_best.reset()
            self.val_ti_agree.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, y = batch
        if self.hparams.stage == 'hint':
            student_logits, guided = self.student.forward_with_guided(x, self.hparams.guided_idx)
            guided = self.regressor(guided)
            with torch.no_grad():
                _, hint = self.teacher.forward_with_phi_hint(x, self.hparams.hint_depth)
            loss = self._hint_mse(guided, hint)
            return (loss, None, None)
        T = self.hparams.temperature
        beta = self.hparams.beta
        student_logits, guided = self.student.forward_with_guided(x, self.hparams.guided_idx)
        guided = self.regressor(guided)
        with torch.no_grad():
            teacher_logits, hint = self.teacher.forward_with_phi_hint(x, self.hparams.hint_depth)
        kd_loss = T ** 2 * self.kl(F.log_softmax(student_logits / T, dim=1), F.log_softmax(teacher_logits / T, dim=1))
        ce_loss = self.ce(student_logits, y)
        hint_loss = self._hint_mse(guided, hint)
        kd_total = kd_loss + self.hparams.alpha * ce_loss
        if beta > 0 and hint_loss > 0:
            scale = kd_total.detach() / hint_loss.detach()
            loss = (1 - beta) * kd_total + beta * scale * hint_loss
        else:
            loss = kd_total
        preds = torch.argmax(student_logits, dim=1)
        return (loss, preds, y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.stage == 'kd':
            self.train_acc(preds, targets)
            self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.stage == 'kd':
            self.val_acc(preds, targets)
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            x, _ = batch
            with torch.no_grad():
                orig_pred = torch.argmax(self.student(x), dim=1)
                H, W = (x.size(-2), x.size(-1))
                agree_sum = 0.0
                for _ in range(3):
                    dx = random.randint(0, H - 1)
                    dy = random.randint(0, W - 1)
                    shifted = torch.roll(x, shifts=(dx, dy), dims=(-2, -1))
                    perm_pred = torch.argmax(self.student(shifted), dim=1)
                    agree_sum += (perm_pred == orig_pred).float().mean()
                agree = agree_sum / 3
            self.val_ti_agree(agree)
            self.log('val/ti_agree', self.val_ti_agree, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.stage == 'kd':
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if stage == 'fit' and (not self._hint_stats_ready):
            self._compute_hint_std()
            self._hint_stats_ready = True
        if self.hparams.compile and stage == 'fit':
            self.student = torch.compile(self.student)

    def _compute_hint_std(self) -> None:
        device = next(self.teacher.parameters()).device
        loader = self.trainer.datamodule.train_dataloader()
        sum_ = torch.zeros_like(self.hint_std)
        sq_sum = torch.zeros_like(self.hint_std)
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, h = self.teacher.forward_with_phi_hint(x, self.hparams.hint_depth)
                sum_ += h.sum(dim=0)
                sq_sum += (h ** 2).sum(dim=0)
                n += h.shape[0]
        mean = sum_ / n
        var = (sq_sum / n - mean ** 2).clamp(min=0)
        std = var.sqrt().clamp(min=1.0)
        self.hint_std.copy_(std)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.freeze_guided and self.hparams.stage == 'kd':
            params = [p for p in self.student.parameters() if p.requires_grad] + list(self.regressor.parameters())
        else:
            params = list(self.student.parameters()) + list(self.regressor.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss', 'interval': 'epoch', 'frequency': 1}}
        return {'optimizer': optimizer}

def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    m_safe = m.clamp(min=torch.finfo(p.dtype).tiny)
    kl_pm = torch.xlogy(p, p / m_safe).sum(dim=1)
    kl_qm = torch.xlogy(q, q / m_safe).sum(dim=1)
    jsd = 0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0, device=p.device))
    return jsd.clamp(min=0.0, max=1.0)

def _translation_inv_stats(model: nn.Module, x: torch.Tensor, num_perms: int=10):
    with torch.no_grad():
        B, C, H, W = x.shape
        device = x.device
        p = F.softmax(model(x), dim=1)
        pred_orig = torch.argmax(p, dim=1)
        jsd_accum = torch.zeros(B, device=device)
        agree_accum = torch.zeros(B, device=device)
        rows = torch.arange(H, device=device).view(1, H, 1)
        cols = torch.arange(W, device=device).view(1, 1, W)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        chan_idx = torch.arange(C, device=device).view(1, C, 1, 1)
        for _ in range(num_perms):
            dxs = torch.randint(0, H, (B,)).to(device).view(B, 1, 1)
            dys = torch.randint(0, W, (B,)).to(device).view(B, 1, 1)
            r_src = (rows - dxs) % H
            c_src = (cols - dys) % W
            r_idx = r_src.unsqueeze(1).expand(B, C, H, W)
            c_idx = c_src.unsqueeze(1).expand(B, C, H, W)
            shifted = x[batch_idx, chan_idx, r_idx, c_idx]
            q = F.softmax(model(shifted), dim=1)
            jsd_accum += _jsd(p, q)
            agree_accum += (torch.argmax(q, dim=1) == pred_orig).float()
    jsd_mean = jsd_accum / num_perms
    agree_mean = agree_accum / num_perms
    return ((1.0 - jsd_mean).clamp(min=0.0, max=1.0), agree_mean)

def _translation_inv_metrics(model: nn.Module, x: torch.Tensor, num_perms: int=10) -> dict:
    one_minus_jsd, agree = _translation_inv_stats(model, x, num_perms)
    return {'ti_1mjsd': one_minus_jsd.mean().item(), 'ti_agree': agree.mean().item()}

def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, ti_seed: int=42) -> dict:
    torch.manual_seed(ti_seed)
    model.to(device)
    model.eval()
    acc_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    ece_metric = MulticlassCalibrationError(num_classes=10).to(device)
    nll_fn = nn.NLLLoss(reduction='sum').to(device)
    total_nll = 0.0
    total_count = 0
    ti_1mjsd_sum = 0.0
    ti_agree_sum = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = (x.to(device), y.to(device))
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            acc_metric.update(logits, y)
            ece_metric.update(logits, y)
            total_nll += nll_fn(log_probs, y).item()
            total_count += y.size(0)
            one_minus_jsd, agree = _translation_inv_stats(model, x)
            ti_1mjsd_sum += one_minus_jsd.sum().item()
            ti_agree_sum += agree.sum().item()
    return {'accu': acc_metric.compute().item(), 'nlll': total_nll / total_count, 'ecel': ece_metric.compute().item(), 'ti_1mjsd': ti_1mjsd_sum / total_count, 'ti_agree': ti_agree_sum / total_count}

def compute_fidelity(student: nn.Module, teacher: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    student.to(device)
    student.eval()
    teacher.to(device)
    teacher.eval()
    kl_fn = nn.KLDivLoss(reduction='sum', log_target=True)
    agree_sum = 0.0
    kl_sum = 0.0
    jsd_sum = 0.0
    total_count = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            s_logits = student(x)
            t_logits = teacher(x)
            agree_sum += (torch.argmax(s_logits, dim=1) == torch.argmax(t_logits, dim=1)).float().sum().item()
            kl_sum += kl_fn(F.log_softmax(s_logits, dim=1), F.log_softmax(t_logits, dim=1)).item()
            p = F.softmax(s_logits, dim=1)
            q = F.softmax(t_logits, dim=1)
            jsd_sum += _jsd(p, q).sum().item()
            total_count += x.size(0)
    return {'top1_agreement': agree_sum / total_count, 'teach_stu_kldiv': kl_sum / total_count, 'teach_stu_jsd': 1.0 - jsd_sum / total_count}

def _load_net_from_ckpt(net: nn.Module, ckpt_path: str, key_prefix: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    prefix_len = len(key_prefix)
    weights = {k[prefix_len:]: v for k, v in ckpt['state_dict'].items() if k.startswith(key_prefix)}
    net.load_state_dict(weights)
    return net

def _build_teacher() -> CNNInvariant:
    return CNNInvariant(input_channels=1, phi_channels=[32, 64, 64], rho_layers=[128], activ='relu', output_dim=10, padding_mode='circular', kernel_size=3)

def _build_mlp() -> MLPBasic:
    return MLPBasic(input_dim=784, layers=[256, 64, 64], output_dim=10, activ='relu')
CFG = dict(seed=42, n_folds=5, split_seed=42, batch_size=128, eval_batch_size=1024, split=(55000, 5000, 10000), teacher_epochs=200, baseline_epochs=200, kd_epochs=200, hint_epochs=600, lr=0.001, weight_decay=0.0, es_patience=30, scheduler_patience=10, temperature=4.0)
DATA_DIR = 'data'
OUTPUT_ROOT = Path('paper_outputs/mnist/canonical_lightning/cv5_w256')
DEVICE = torch.device('cpu')
FOLDS = list(range(5))

def configure(data_root, output_root, folds=(0, 1, 2, 3, 4), device=None):
    global DATA_DIR, OUTPUT_ROOT, FOLDS, DEVICE
    DATA_DIR = str(data_root)
    OUTPUT_ROOT = Path(output_root) / 'canonical_lightning' / 'cv5_w256'
    FOLDS = list(folds)
    DEVICE = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

def datamodule(fold, evaluation=False):
    return MNISTInMemoryGPUDataModule(DATA_DIR, batch_size=CFG['eval_batch_size'] if evaluation else CFG['batch_size'], train_val_test_split=CFG['split'], mean=0.1307, std=0.3081, device=str(DEVICE), split_seed=CFG['split_seed'], n_folds=CFG['n_folds'], fold_idx=fold)

def stage_directory(stage, fold, temperature=None):
    root = OUTPUT_ROOT if temperature is None else OUTPUT_ROOT / f'T{temperature:g}'
    return root / stage / f'fold_{fold}'

def checkpoint(stage, fold, temperature=None):
    return stage_directory(stage, fold, temperature) / 'checkpoints' / 'best.ckpt'

def run_stage(stage, fold, temperature=None):
    directory = stage_directory(stage, fold, temperature)
    completed = directory / 'done.txt'
    if completed.is_file() and checkpoint(stage, fold, temperature).is_file():
        return checkpoint(stage, fold, temperature)
    seed_everything(CFG['seed'], workers=True, verbose=False)
    opt = partial(torch.optim.Adam, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=CFG['scheduler_patience'])
    if stage == 'cnn_teacher':
        model = MNISTLitModule(_build_teacher(), opt, sched, False)
        epochs = CFG['teacher_epochs']
    elif stage == 'mlp_baseline':
        model = MNISTLitModule(_build_mlp(), opt, sched, False)
        epochs = CFG['baseline_epochs']
    else:
        hint = stage == 'hint'
        student_ckpt = str(checkpoint('hint', fold)) if stage in ('kd_b0', 'kd_b025') else None
        model = FitNetsMNISTModule(_build_teacher(), _build_mlp(), opt, sched, teacher_ckpt=str(checkpoint('cnn_teacher', fold)), stage='hint' if hint else 'kd', temperature=CFG['temperature'] if temperature is None else temperature, alpha=0.0, beta=0.25 if stage == 'kd_b025' else 0.0, guided_idx=3, hint_depth=3, freeze_guided=False, student_ckpt=student_ckpt, num_classes=10, compile=False)
        epochs = CFG['hint_epochs'] if hint else CFG['kd_epochs']
    monitor, mode = ('val/loss', 'min') if stage == 'hint' else ('val/acc', 'max')
    saved = ModelCheckpoint(dirpath=directory / 'checkpoints', filename='best', monitor=monitor, mode=mode, save_last=False, save_top_k=1, auto_insert_metric_name=False, enable_version_counter=False)
    latest = ModelCheckpoint(dirpath=directory / 'checkpoints', filename='last', monitor=None, save_top_k=-1, save_last=False, auto_insert_metric_name=False, enable_version_counter=False, every_n_epochs=1, save_on_train_epoch_end=True)
    stop = EarlyStopping(monitor=monitor, mode=mode, patience=CFG['es_patience'], min_delta=0.0)
    logger = CSVLogger(save_dir=str(directory), name='csv')
    trainer = Trainer(default_root_dir=str(directory), min_epochs=1, max_epochs=epochs, accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu', devices=[DEVICE.index or 0] if DEVICE.type == 'cuda' else 1, check_val_every_n_epoch=1, deterministic=False, num_sanity_val_steps=0, log_every_n_steps=200, callbacks=[saved, latest, stop], logger=logger, enable_model_summary=False, enable_progress_bar=False)
    dm = datamodule(fold)
    last = directory / 'checkpoints' / 'last.ckpt'
    trainer.fit(model=model, datamodule=dm, ckpt_path=str(last) if last.is_file() else None, weights_only=False)
    if stage in ('cnn_teacher', 'mlp_baseline'):
        trainer.test(model=model, datamodule=dm, ckpt_path=saved.best_model_path, weights_only=False, verbose=False)
    completed.write_text('complete\n', encoding='utf-8')
    return checkpoint(stage, fold, temperature)

def summarize(temperatures=()):
    results = []
    paths = [(stage, fold, None) for stage in ('cnn_teacher', 'mlp_baseline', 'pureKD', 'kd_b0', 'kd_b025') for fold in FOLDS]
    paths.extend(((stage, fold, float(t)) for t in temperatures for stage in ('pureKD', 'kd_b0', 'kd_b025') for fold in FOLDS))
    for stage, fold, temperature in paths:
        path = checkpoint(stage, fold, temperature)
        if not path.is_file():
            continue
        net = _build_teacher() if stage == 'cnn_teacher' else _build_mlp()
        prefix = 'net.' if stage in ('cnn_teacher', 'mlp_baseline') else 'student.'
        net = _load_net_from_ckpt(net, str(path), prefix)
        dm = datamodule(fold, evaluation=True)
        dm.prepare_data()
        dm.setup('fit')
        loader = dm.val_dataloader()
        metrics = validate(net, loader, DEVICE)
        teacher_path = checkpoint('cnn_teacher', fold)
        if stage != 'cnn_teacher' and teacher_path.is_file():
            teacher = _load_net_from_ckpt(_build_teacher(), str(teacher_path), 'net.')
            fid = compute_fidelity(net, teacher, loader, DEVICE)
            metrics.update(fid_top1=fid['top1_agreement'], fid_1mjsd=fid['teach_stu_jsd'])
        else:
            metrics.update(fid_top1=None, fid_1mjsd=None)
        row = dict(stage=stage, fold=fold, temperature=temperature, **metrics)
        results.append(row)
    summary = {}
    for stage, temperature in sorted({(r['stage'], r['temperature']) for r in results}, key=str):
        rows = [r for r in results if r['stage'] == stage and r['temperature'] == temperature]
        key = stage if temperature is None else f'T{temperature:g}/{stage}'
        summary[key] = {}
        for k in ('accu', 'nlll', 'ecel', 'ti_1mjsd', 'ti_agree', 'fid_top1', 'fid_1mjsd'):
            values = [r[k] for r in rows if r[k] is not None]
            summary[key][k] = {'mean': float(np.mean(values)) if values else None, 'std': float(np.std(values)) if values else None}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / 'summary.json').write_text(json.dumps({'folds': results, 'summary': summary}, indent=2), encoding='utf-8')
    return results

def run(stages=('teacher', 'baseline', 'hint', 'temperature_scan', 'summary'), temperatures=(1, 2, 4, 8, 16)):
    if 'teacher' in stages:
        for fold in FOLDS:
            run_stage('cnn_teacher', fold)
    names = {'baseline': 'mlp_baseline', 'pure_kd': 'pureKD', 'hint': 'hint', 'kd_hint': 'kd_b0', 'kd_hint_beta': 'kd_b025'}
    for fold in FOLDS:
        for stage in stages:
            if stage in names:
                run_stage(names[stage], fold)
    if 'temperature_scan' in stages:
        for temperature in temperatures:
            for fold in FOLDS:
                for stage in ('pureKD', 'kd_b0', 'kd_b025'):
                    run_stage(stage, fold, float(temperature))
    if 'summary' in stages:
        summarize(temperatures)
