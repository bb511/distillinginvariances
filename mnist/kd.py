import os, csv, json, math, random, copy
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import _LRScheduler
DATASET = 'mnist'
VARIANT_TAG = 'w256'
SHIFT_MODE = 'shuffle'
SHIFT_SEED = 0
SHIFT_MAX = None
SEED = 42
SPLIT_SEED = 42
N_FOLDS = 5
Temperature = 1.0
FOLDS = list(range(N_FOLDS))
PRESETS = {'mnist': dict(input_shape=(1, 28, 28), num_classes=10, split=(55000, 5000, 10000), batch_size=128, teacher_kwargs=dict(input_channels=1, phi_channels=[32, 64, 64], rho_layers=[128], activ='relu', output_dim=10, padding_mode='circular', kernel_size=3, use_bn=False), student_kwargs=dict(input_dim=784, layers=[256, 64, 64], output_dim=10, activ='relu'), scheduler='rop', teacher=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), baseline=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), pureKD=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), hint=dict(max_epochs=600, lr=0.001, wd=0.0, monitor='val/loss', mode='min'), kd_b0=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), kd_b025=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), es_patience=30, guided_idx=3, hint_depth=3, temperature=Temperature, alpha=0.0)}
CFG = PRESETS[DATASET]
_DEFAULT_CFG = copy.deepcopy(CFG)
DEVICE = torch.device('cpu')
LOG_DIR = '.'
DATA_DIR = '.'
CV_ROOT = '.'
CV_ROOT_T = '.'
_MNIST_MEAN, _MNIST_STD = (0.1307, 0.3081)
ALL_X = ALL_Y = None

def apply_one_shot_translation(x: torch.Tensor, seed: int, max_shift=None) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f'expected (N,C,H,W); got {tuple(x.shape)}')
    N, C, H, W = x.shape
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    if max_shift is None:
        dx = torch.randint(0, H, (N,), generator=g)
        dy = torch.randint(0, W, (N,), generator=g)
    else:
        dx = torch.randint(-int(max_shift), int(max_shift) + 1, (N,), generator=g)
        dy = torch.randint(-int(max_shift), int(max_shift) + 1, (N,), generator=g)
    dev = x.device
    dx, dy = (dx.to(dev), dy.to(dev))
    row = (torch.arange(H, device=dev).view(1, H, 1).expand(N, H, W) - dx.view(N, 1, 1)) % H
    col = (torch.arange(W, device=dev).view(1, 1, W).expand(N, H, W) - dy.view(N, 1, 1)) % W
    row_e = row.unsqueeze(1).expand(-1, C, -1, -1)
    col_e = col.unsqueeze(1).expand(-1, C, -1, -1)
    flat_idx = (row_e * W + col_e).reshape(N, C, H * W)
    return x.reshape(N, C, H * W).gather(2, flat_idx).reshape(N, C, H, W)

class GPUBatchLoader:

    def __init__(self, x, y, batch_size, shuffle):
        self.x, self.y = (x, y)
        self.batch_size, self.shuffle = (batch_size, shuffle)

    def __iter__(self):
        n = self.x.size(0)
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            sel = idx[i:i + self.batch_size]
            yield (self.x[sel], self.y[sel])

    def __len__(self):
        return (self.x.size(0) + self.batch_size - 1) // self.batch_size

def _load_full_dataset(name, data_dir, shift_mode, shift_seed, shift_max, device):
    if name == 'mnist':
        MNIST(data_dir, train=True, download=True)
        MNIST(data_dir, train=False, download=True)
        tr = MNIST(data_dir, train=True, download=False)
        te = MNIST(data_dir, train=False, download=False)
        tr_x = (tr.data.float().unsqueeze(1) / 255.0 - _MNIST_MEAN) / _MNIST_STD
        te_x = (te.data.float().unsqueeze(1) / 255.0 - _MNIST_MEAN) / _MNIST_STD
        tr_y, te_y = (tr.targets.long(), te.targets.long())
    else:
        raise ValueError(name)
    all_x = torch.cat([tr_x, te_x], dim=0)
    all_y = torch.cat([tr_y, te_y], dim=0)
    if shift_mode == 'shuffle':
        all_x = apply_one_shot_translation(all_x, seed=shift_seed, max_shift=shift_max)
    elif shift_mode != 'none':
        raise ValueError(f'unknown shift_mode={shift_mode!r}')
    return (all_x.to(device).contiguous(), all_y.to(device).contiguous())

def fold_split(all_x, all_y, split, split_seed, n_folds, fold_idx):
    n_tr, n_va, n_te = split
    g = torch.Generator(device='cpu').manual_seed(int(split_seed))
    idx = torch.randperm(all_x.size(0), generator=g)
    n_pool = n_tr + n_va
    pool = idx[:n_pool]
    te = idx[n_pool:n_pool + n_te]
    chunk = n_pool // n_folds
    s = fold_idx * chunk
    e = (fold_idx + 1) * chunk if fold_idx < n_folds - 1 else n_pool
    tr = torch.cat([pool[:s], pool[e:]])
    va = pool[s:e]
    return ((all_x[tr].contiguous(), all_y[tr].contiguous()), (all_x[va].contiguous(), all_y[va].contiguous()), (all_x[te].contiguous(), all_y[te].contiguous()))

def build_loaders_for_fold(fold_idx):
    tr_xy, va_xy, te_xy = fold_split(ALL_X, ALL_Y, CFG['split'], SPLIT_SEED, N_FOLDS, fold_idx)
    bs = CFG['batch_size']
    return (GPUBatchLoader(*tr_xy, batch_size=bs, shuffle=True), GPUBatchLoader(*va_xy, batch_size=bs, shuffle=False), GPUBatchLoader(*te_xy, batch_size=bs, shuffle=False))

def _activation(name):
    return {'relu': lambda: nn.ReLU(inplace=True), 'tanh': lambda: nn.Tanh(), 'sigmoid': lambda: nn.Sigmoid(), 'leaky_relu': lambda: nn.LeakyReLU()}[name]()

class CNNInvariant(nn.Module):

    def __init__(self, input_channels, phi_channels, rho_layers, activ='relu', output_dim=10, padding_mode='circular', kernel_size=3, use_bn=False):
        super().__init__()
        self.input_channels = input_channels
        self.phi_channels = list(phi_channels)
        self.rho_layers = list(rho_layers)
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self._phi_block = 3 if use_bn else 2
        self.phi = self._construct_phi(activ)
        self.rho = self._construct_rho(activ)

    def _construct_phi(self, activ):
        phi = nn.Sequential()
        chs = [self.input_channels] + self.phi_channels
        pad = self.kernel_size // 2
        for i in range(len(chs) - 1):
            phi.append(nn.Conv2d(chs[i], chs[i + 1], kernel_size=self.kernel_size, padding=pad, padding_mode=self.padding_mode, stride=1))
            if self.use_bn:
                phi.append(nn.BatchNorm2d(chs[i + 1]))
            phi.append(_activation(activ))
        return phi

    def _construct_rho(self, activ):
        rho = nn.Sequential()
        layers = [self.phi_channels[-1]] + self.rho_layers + [self.output_dim]
        for i in range(len(layers) - 1):
            rho.append(nn.Linear(layers[i], layers[i + 1]))
            if i == len(layers) - 2:
                break
            rho.append(_activation(activ))
        return rho

    @staticmethod
    def _agg(fm):
        return F.adaptive_avg_pool2d(fm, 1).flatten(1)

    def forward(self, x):
        return self.rho(self._agg(self.phi(x)))

    def forward_with_phi_hint(self, x, phi_depth=3):
        target = phi_depth * self._phi_block - 1
        h = x
        hint = None
        for i, layer in enumerate(self.phi):
            h = layer(h)
            if i == target:
                hint = self._agg(h)
        return (self.rho(self._agg(h)), hint)

class MLPBasic(nn.Module):

    def __init__(self, input_dim, layers, output_dim, activ='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.layers = list(layers)
        self.output_dim = output_dim
        all_layers = [input_dim] + self.layers + [output_dim]
        self.mlp = nn.Sequential()
        for i in range(len(all_layers) - 1):
            self.mlp.add_module(f'linear_{i}', nn.Linear(all_layers[i], all_layers[i + 1]))
            if i < len(all_layers) - 2:
                self.mlp.add_module(f'activation_{i}', _activation(activ))

    def forward(self, x):
        return self.mlp(torch.flatten(x, start_dim=1))

    def forward_with_guided(self, x, guided_idx=3):
        x = torch.flatten(x, start_dim=1)
        guided = None
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == guided_idx:
                guided = x
        return (x, guided)

class NormalizedHintMSE(nn.Module):

    def __init__(self, hint_dim):
        super().__init__()
        self.register_buffer('std', torch.ones(hint_dim))

    @torch.no_grad()
    def compute_train_std(self, teacher, train_loader, hint_depth, device):
        teacher.eval()
        sum_ = torch.zeros_like(self.std)
        sq = torch.zeros_like(self.std)
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            _, h = teacher.forward_with_phi_hint(x, hint_depth)
            sum_ += h.sum(dim=0)
            sq += (h ** 2).sum(dim=0)
            n += h.shape[0]
        mean = sum_ / n
        var = (sq / n - mean ** 2).clamp(min=0)
        self.std.copy_(var.sqrt().clamp(min=1.0))

    def forward(self, guided, hint):
        s = self.std.view(1, -1)
        return F.mse_loss(guided / s, hint / s)

class TemperatureKL(nn.Module):

    def __init__(self, temperature=Temperature):
        super().__init__()
        self.T = float(temperature)
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, sl, tl):
        return self.T ** 2 * self.kl(F.log_softmax(sl / self.T, dim=1), F.log_softmax(tl / self.T, dim=1))

def combine_kd_ce_hint(kd_loss, ce_loss, hint_loss, alpha, beta):
    kd_total = kd_loss + alpha * ce_loss
    if beta > 0 and hint_loss is not None and (hint_loss > 0):
        scale = kd_total.detach() / hint_loss.detach()
        return (1 - beta) * kd_total + beta * scale * hint_loss
    return kd_total

@torch.no_grad()
def translation_agreement(net, loader, device, n_shifts=3):
    net.eval()
    wsum = 0.0
    total = 0
    for x, _ in loader:
        x = x.to(device)
        orig = torch.argmax(net(x), dim=1)
        H, W = (x.size(-2), x.size(-1))
        s = 0.0
        for _ in range(n_shifts):
            dx = random.randint(0, H - 1)
            dy = random.randint(0, W - 1)
            shp = torch.argmax(net(torch.roll(x, shifts=(dx, dy), dims=(-2, -1))), dim=1)
            s += (shp == orig).float().mean().item()
        wsum += s / n_shifts * x.size(0)
        total += x.size(0)
    return wsum / max(1, total)

@torch.no_grad()
def top1_acc(net, loader, device):
    net.eval()
    crit = nn.CrossEntropyLoss(reduction='sum')
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = net(x)
        loss_sum += float(crit(logits, y).item())
        correct += int((torch.argmax(logits, dim=1) == y).sum().item())
        total += int(y.numel())
    return (loss_sum / max(1, total), correct / max(1, total))

def classifier_eval(model, val_loader, device):
    vl, va = top1_acc(model, val_loader, device)
    ti = translation_agreement(model, val_loader, device)
    return {'val/loss': vl, 'val/acc': va, 'val/ti_agree': ti}

class Trainer:

    def __init__(self, run_dir, max_epochs, monitor='val/loss', mode='min', es_patience=30, scheduler_step='epoch'):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.max_epochs = max_epochs
        self.monitor = monitor
        assert mode in ('min', 'max')
        self.mode = mode
        self.es_patience = es_patience
        self.scheduler_step = scheduler_step
        self.last_path = os.path.join(run_dir, 'last.pt')
        self.best_path = os.path.join(run_dir, 'best.pt')
        self.csv_path = os.path.join(run_dir, 'metrics.csv')
        self.done_path = os.path.join(run_dir, 'done.txt')
        self.cols = ['epoch', 'lr', 'train/loss', 'val/loss', 'val/acc', 'val/ti_agree', 'best']

    def _better(self, a, b):
        return a < b if self.mode == 'min' else a > b

    def _init_best(self):
        return math.inf if self.mode == 'min' else -math.inf

    def _maybe_resume(self, model, optimizer, scheduler):
        if not os.path.isfile(self.last_path):
            return (0, self._init_best(), 0)
        ck = torch.load(self.last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state'])
        if optimizer is not None and 'optimizer_state' in ck:
            optimizer.load_state_dict(ck['optimizer_state'])
        if scheduler is not None and ck.get('scheduler_state') is not None:
            scheduler.load_state_dict(ck['scheduler_state'])
        return (int(ck.get('epoch', 0)) + 1, float(ck.get('best', self._init_best())), int(ck.get('patience_counter', 0)))

    def _row(self, row):
        new = not os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.cols)
            if new:
                w.writeheader()
            w.writerow(row)

    def fit(self, model, optimizer, scheduler, train_step_fn, eval_fn, train_loader, val_loader, device, extra_state=None):
        if os.path.isfile(self.done_path):
            return
        model.to(device)
        start_epoch, best, pcount = self._maybe_resume(model, optimizer, scheduler)
        epoch = start_epoch
        for epoch in range(start_epoch, self.max_epochs):
            model.train()
            tl_sum = 0.0
            tl_n = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, n_items, _ = train_step_fn(model, batch, device)
                loss.backward()
                optimizer.step()
                tl_sum += float(loss.item()) * n_items
                tl_n += n_items
            train_loss = tl_sum / max(1, tl_n)
            metrics = eval_fn(model, val_loader, device)
            mv = float(metrics.get(self.monitor, math.nan))
            improved = self._better(mv, best)
            if improved:
                best = mv
                pcount = 0
            else:
                pcount += 1
            cur_lr = optimizer.param_groups[0]['lr']
            self._row({'epoch': epoch, 'lr': cur_lr, 'train/loss': train_loss, 'val/loss': metrics.get('val/loss', ''), 'val/acc': metrics.get('val/acc', ''), 'val/ti_agree': metrics.get('val/ti_agree', ''), 'best': best})
            print(f'{self.run_dir} epoch={epoch} loss={train_loss:.4f} {self.monitor}={mv:.4f} best={best:.4f} lr={cur_lr:.2e}')
            ck = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict() if scheduler is not None else None, 'best': best, 'patience_counter': pcount}
            if extra_state is not None:
                ck['extra_state'] = {k: v.state_dict() if hasattr(v, 'state_dict') else v for k, v in extra_state.items()}
            torch.save(ck, self.last_path)
            if improved:
                torch.save(ck, self.best_path)
            if scheduler is not None:
                if self.scheduler_step == 'plateau':
                    scheduler.step(metrics.get('val/loss', train_loss))
                else:
                    scheduler.step()
            if pcount >= self.es_patience:
                break
        with open(self.done_path, 'w') as f:
            f.write(f'finished_epoch={epoch}\n')

def build_scheduler(opt, kind, max_epochs):
    if kind == 'cosine':
        return (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-05), 'epoch')
    if kind == 'rop':
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10), 'plateau')
    raise ValueError(kind)

def build_regressor(student_kwargs, teacher_kwargs, guided_idx):
    all_layers = [student_kwargs['input_dim']] + list(student_kwargs['layers']) + [student_kwargs['output_dim']]
    guided_dim = all_layers[(guided_idx + 2) // 2]
    hint_dim = teacher_kwargs['phi_channels'][-1]
    if guided_dim == hint_dim:
        return (nn.Identity(), guided_dim, hint_dim)
    return (nn.Linear(guided_dim, hint_dim), guided_dim, hint_dim)

class HintStudent(nn.Module):

    def __init__(self, student, regressor, guided_idx):
        super().__init__()
        self.student = student
        self.regressor = regressor
        self.guided_idx = guided_idx

    def forward(self, x):
        return self.student(x)

    def guided(self, x):
        _, g = self.student.forward_with_guided(x, self.guided_idx)
        return self.regressor(g)

def teacher_dir(fold):
    if DATASET == 'cifar10':
        return os.path.join(LOG_DIR, 'cv5_teachers', f'fold_{fold}')
    return os.path.join(CV_ROOT, 'cnn_teacher', f'fold_{fold}')

def stage_dir(stage, fold):
    if stage in ('pureKD', 'kd_b0', 'kd_b025'):
        return os.path.join(CV_ROOT_T, stage, f'fold_{fold}')
    return os.path.join(CV_ROOT, stage, f'fold_{fold}')

def load_teacher_for_fold(fold):
    path = os.path.join(teacher_dir(fold), 'best.pt')
    t = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    t.load_state_dict(ck['model_state'])
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t

def load_hint_into_kd_model(fold):
    s = MLPBasic(**CFG['student_kwargs'])
    reg, gd, hd = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
    full = HintStudent(s, reg, CFG['guided_idx']).to(DEVICE)
    ck = torch.load(os.path.join(stage_dir('hint', fold), 'best.pt'), map_location='cpu', weights_only=False)
    full.load_state_dict(ck['model_state'])
    return (full, hd)

def cls_train_step(model, batch, device):
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return (F.cross_entropy(model(x), y), y.size(0), {})

class WarmupCosineLR(_LRScheduler):

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_factor=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * self.min_lr_factor + base_lr * (1 - self.min_lr_factor) * cosine_decay for base_lr in self.base_lrs]

def build_custom_scheduler(opt, max_epochs):
    return (WarmupCosineLR(opt, warmup_epochs=3, max_epochs=max_epochs, min_lr_factor=0.01), 'epoch')

def run_teacher():
    for fold in FOLDS:
        rd = teacher_dir(fold)
        teacher = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
        P = CFG['teacher']
        opt = torch.optim.Adam(teacher.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        Trainer(rd, max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(teacher, opt, sched, cls_train_step, classifier_eval, train_loader, val_loader, DEVICE)
        del teacher, opt, sched, train_loader, val_loader

def run_baseline():
    for fold in FOLDS:
        P = CFG['baseline']
        model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        Trainer(stage_dir('mlp_baseline', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, cls_train_step, classifier_eval, train_loader, val_loader, DEVICE)
        del model, opt, sched, train_loader, val_loader

def run_pure_kd():
    for fold in FOLDS:
        P = CFG['pureKD']
        teacher_eval = load_teacher_for_fold(fold)
        model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        kd = TemperatureKL(CFG['temperature']).to(DEVICE)

        def pkd_step(m, batch, device):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            sl = m(x)
            with torch.no_grad():
                tl = teacher_eval(x)
            loss = kd(sl, tl) + CFG['alpha'] * F.cross_entropy(sl, y)
            return (loss, x.size(0), {})
        opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        Trainer(stage_dir('pureKD', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, pkd_step, classifier_eval, train_loader, val_loader, DEVICE)
        del model, opt, sched, kd, teacher_eval, train_loader, val_loader

def run_hint():
    for fold in FOLDS:
        P = CFG['hint']
        teacher_eval = load_teacher_for_fold(fold)
        student = MLPBasic(**CFG['student_kwargs'])
        regressor, gd, hd = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
        model = HintStudent(student, regressor, CFG['guided_idx']).to(DEVICE)
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
        hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)

        def hint_step(m, batch, device):
            x, _ = batch
            x = x.to(device)
            g = m.guided(x)
            with torch.no_grad():
                _, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
            return (hint_loss(g, h), x.size(0), {})

        @torch.no_grad()
        def hint_eval(m, loader, device):
            m.eval()
            s = 0.0
            n = 0
            for x, _ in loader:
                x = x.to(device)
                g = m.guided(x)
                _, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                s += float(hint_loss(g, h).item()) * x.size(0)
                n += x.size(0)
            return {'val/loss': s / max(1, n)}
        opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        Trainer(stage_dir('hint', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, hint_step, hint_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
        del model, opt, sched, hint_loss, teacher_eval, train_loader, val_loader

def run_kd_hint():
    for fold in FOLDS:
        if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
            print(f'  skip: hint ckpt missing')
            continue
        P = CFG['kd_b0']
        teacher_eval = load_teacher_for_fold(fold)
        model, hd = load_hint_into_kd_model(fold)
        kd = TemperatureKL(CFG['temperature']).to(DEVICE)
        hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
        ALPHA = CFG['alpha']
        BETA = 0.0

        def kd_step(m, batch, device):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            sl = m.student(x)
            g = m.guided(x)
            with torch.no_grad():
                tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
            kdL = kd(sl, tl)
            ceL = F.cross_entropy(sl, y)
            htL = hint_loss(g, h)
            return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
        opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        Trainer(stage_dir('kd_b0', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
        del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader

def run_kd_hint_beta():
    for fold in FOLDS:
        if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
            print(f'  skip: hint ckpt missing')
            continue
        P = CFG['kd_b025']
        teacher_eval = load_teacher_for_fold(fold)
        model, hd = load_hint_into_kd_model(fold)
        kd = TemperatureKL(CFG['temperature']).to(DEVICE)
        hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
        train_loader, val_loader, _ = build_loaders_for_fold(fold)
        hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
        ALPHA = CFG['alpha']
        BETA = 0.25

        def kd_step(m, batch, device):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            sl = m.student(x)
            g = m.guided(x)
            with torch.no_grad():
                tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
            kdL = kd(sl, tl)
            ceL = F.cross_entropy(sl, y)
            htL = hint_loss(g, h)
            return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
        opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
        sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
        Trainer(stage_dir('kd_b025', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
        del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader

def run_temperature_scan(temperatures=(1.0, 2.0, 4.0, 8.0, 16.0)):
    global Temperature, CV_ROOT_T
    T_values = temperatures
    for T in T_values:
        Temperature = T
        CFG['temperature'] = Temperature
        CV_ROOT_T = os.path.join(LOG_DIR, f'T{int(Temperature)}_cv5_{VARIANT_TAG}')
        os.makedirs(CV_ROOT_T, exist_ok=True)
        for fold in FOLDS:
            P = CFG['pureKD']
            teacher_eval = load_teacher_for_fold(fold)
            model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)

            def pkd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m(x)
                with torch.no_grad():
                    tl = teacher_eval(x)
                loss = kd(sl, tl) + CFG['alpha'] * F.cross_entropy(sl, y)
                return (loss, x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            Trainer(stage_dir('pureKD', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, pkd_step, classifier_eval, train_loader, val_loader, DEVICE)
            del model, opt, sched, kd, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b0']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            ALPHA = CFG['alpha']
            BETA = 0.0

            def kd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            Trainer(stage_dir('kd_b0', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b025']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            ALPHA = CFG['alpha']
            BETA = 0.25

            def kd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            Trainer(stage_dir('kd_b025', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader

def run_alpha_scan(alphas=(0.0, 2.0), temperature=8.0):
    T_val = temperature
    alphas_to_run = alphas
    for alpha_val in alphas_to_run:
        CFG['temperature'] = T_val
        ALPHA = alpha_val
        curr_cv_root = os.path.join(LOG_DIR, f'T{int(T_val)}_cv5_w256_a{alpha_val}_cos')
        os.makedirs(curr_cv_root, exist_ok=True)

        def curr_stage_dir(stage, fold):
            return os.path.join(curr_cv_root, stage, f'fold_{fold}')
        for fold in FOLDS:
            P = CFG['pureKD']
            teacher_eval = load_teacher_for_fold(fold)
            model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)

            def pkd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m(x)
                with torch.no_grad():
                    tl = teacher_eval(x)
                loss = kd(sl, tl) + ALPHA * F.cross_entropy(sl, y)
                return (loss, x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_custom_scheduler(opt, P['max_epochs'])
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            Trainer(curr_stage_dir('pureKD', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, pkd_step, classifier_eval, train_loader, val_loader, DEVICE)
            del model, opt, sched, kd, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b0']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            BETA = 0.0

            def kd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_custom_scheduler(opt, P['max_epochs'])
            Trainer(curr_stage_dir('kd_b0', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b025']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            BETA = 0.25

            def kd_step_b025(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, ALPHA, BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_custom_scheduler(opt, P['max_epochs'])
            Trainer(curr_stage_dir('kd_b025', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step_b025, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader

def run_alpha_temperature_scan(temperatures=(1.0, 2.0, 4.0, 8.0, 16.0), alpha=2.0):
    T_values = temperatures
    ALPHA_VAL = alpha
    for T in T_values:
        CFG['temperature'] = float(T)
        CFG['alpha'] = ALPHA_VAL
        curr_cv_root = os.path.join(LOG_DIR, f'T{int(T)}_cv5_{VARIANT_TAG}_a{ALPHA_VAL}')
        os.makedirs(curr_cv_root, exist_ok=True)

        def curr_stage_dir(stage, fold):
            return os.path.join(curr_cv_root, stage, f'fold_{fold}')
        for fold in FOLDS:
            P = CFG['pureKD']
            teacher_eval = load_teacher_for_fold(fold)
            model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)

            def pkd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m(x)
                with torch.no_grad():
                    tl = teacher_eval(x)
                loss = kd(sl, tl) + CFG['alpha'] * F.cross_entropy(sl, y)
                return (loss, x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            Trainer(curr_stage_dir('pureKD', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, pkd_step, classifier_eval, train_loader, val_loader, DEVICE)
            del model, opt, sched, kd, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b0']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            BETA = 0.0

            def kd_step(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, CFG['alpha'], BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            Trainer(curr_stage_dir('kd_b0', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader
        for fold in FOLDS:
            if not os.path.isfile(os.path.join(stage_dir('hint', fold), 'best.pt')):
                print(f'  skip: hint ckpt missing')
                continue
            P = CFG['kd_b025']
            teacher_eval = load_teacher_for_fold(fold)
            model, hd = load_hint_into_kd_model(fold)
            kd = TemperatureKL(CFG['temperature']).to(DEVICE)
            hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
            train_loader, val_loader, _ = build_loaders_for_fold(fold)
            hint_loss.compute_train_std(teacher_eval, train_loader, CFG['hint_depth'], DEVICE)
            BETA = 0.25

            def kd_step_b025(m, batch, device):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                sl = m.student(x)
                g = m.guided(x)
                with torch.no_grad():
                    tl, h = teacher_eval.forward_with_phi_hint(x, CFG['hint_depth'])
                kdL = kd(sl, tl)
                ceL = F.cross_entropy(sl, y)
                htL = hint_loss(g, h)
                return (combine_kd_ce_hint(kdL, ceL, htL, CFG['alpha'], BETA), x.size(0), {})
            opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
            sched, step = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
            Trainer(curr_stage_dir('kd_b025', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=step).fit(model, opt, sched, kd_step_b025, classifier_eval, train_loader, val_loader, DEVICE, extra_state={'hint_std': hint_loss})
            del model, opt, sched, kd, hint_loss, teacher_eval, train_loader, val_loader

def _jsd(p, q):
    m = 0.5 * (p + q)
    m_safe = m.clamp(min=torch.finfo(p.dtype).tiny)
    kl_pm = torch.xlogy(p, p / m_safe).sum(dim=1)
    kl_qm = torch.xlogy(q, q / m_safe).sum(dim=1)
    jsd = 0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0, device=p.device))
    return jsd.clamp(min=0.0, max=1.0)

@torch.no_grad()
def _per_sample_ti_stats(model, x, num_perms=10):
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
    return ((1.0 - jsd_accum / num_perms).clamp(0.0, 1.0), agree_accum / num_perms)

@torch.no_grad()
def validate_full(model, loader, device, ti_seed=42, n_bins=15):
    torch.manual_seed(ti_seed)
    model.to(device)
    model.eval()
    nll_sum = 0.0
    correct = 0
    total = 0
    ti1mj_sum = 0.0
    tia_sum = 0.0
    bin_correct = torch.zeros(n_bins, device=device)
    bin_conf = torch.zeros(n_bins, device=device)
    bin_count = torch.zeros(n_bins, device=device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_p = F.log_softmax(logits, dim=1)
        nll_sum += float(F.nll_loss(log_p, y, reduction='sum').item())
        probs = log_p.exp()
        conf, pred = probs.max(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        bins = (conf * n_bins).long().clamp(max=n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if mask.any():
                bin_correct[b] += float((pred[mask] == y[mask]).sum().item())
                bin_conf[b] += float(conf[mask].sum().item())
                bin_count[b] += int(mask.sum().item())
        omj, agr = _per_sample_ti_stats(model, x)
        ti1mj_sum += float(omj.sum().item())
        tia_sum += float(agr.sum().item())
    ece = 0.0
    for b in range(n_bins):
        c = float(bin_count[b].item())
        if c > 0:
            acc_b = float(bin_correct[b].item()) / c
            conf_b = float(bin_conf[b].item()) / c
            ece += c / total * abs(acc_b - conf_b)
    return dict(accu=correct / total, nlll=nll_sum / total, ecel=ece, ti_1mjsd=ti1mj_sum / total, ti_agree=tia_sum / total)

@torch.no_grad()
def fidelity(student, teacher, loader, device):
    student.eval()
    teacher.eval()
    agree = 0.0
    jsd_sum = 0.0
    total = 0
    for x, _ in loader:
        x = x.to(device)
        sl = student(x)
        tl = teacher(x)
        agree += float((torch.argmax(sl, dim=1) == torch.argmax(tl, dim=1)).sum().item())
        p_s = F.softmax(sl, dim=1)
        p_t = F.softmax(tl, dim=1)
        jsd_sum += float(_jsd(p_s, p_t).sum().item())
        total += int(x.size(0))
    return dict(fid_top1=agree / total, fid_1mjsd=1.0 - jsd_sum / total)

def build_wrap():
    s = MLPBasic(**CFG['student_kwargs'])
    r, _, _ = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
    return HintStudent(s, r, CFG['guided_idx'])

def fmt_ext(vals):
    a = np.array([v for v in vals if v is not None and (not (isinstance(v, float) and math.isnan(v)))])
    return '    nan    ' if a.size == 0 else f'{a.mean():.4f}±{a.std():.4f}'

def configure(data_root, output_root, regime='transformed', folds=(0, 1, 2, 3, 4), device=None):
    global CFG, DATA_DIR, LOG_DIR, CV_ROOT, CV_ROOT_T, SHIFT_MODE, FOLDS, DEVICE, ALL_X, ALL_Y, Temperature
    CFG = copy.deepcopy(_DEFAULT_CFG)
    SHIFT_MODE = {'canonical': 'none', 'transformed': 'shuffle'}[regime]
    FOLDS = list(folds)
    DEVICE = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    DATA_DIR = str(data_root)
    LOG_DIR = os.path.join(str(output_root), f'mnist_cv5_{VARIANT_TAG}_{SHIFT_MODE}_s{SHIFT_SEED}')
    CV_ROOT = os.path.join(LOG_DIR, f'cv5_{VARIANT_TAG}')
    Temperature = float(CFG['temperature'])
    CV_ROOT_T = os.path.join(LOG_DIR, f'T{int(Temperature)}_cv5_{VARIANT_TAG}')
    os.makedirs(LOG_DIR, exist_ok=True)
    ALL_X = ALL_Y = None
    torch.manual_seed(SEED)
    random.seed(SEED)

def prepare_data():
    global ALL_X, ALL_Y
    if ALL_X is None:
        ALL_X, ALL_Y = _load_full_dataset(DATASET, DATA_DIR, SHIFT_MODE, SHIFT_SEED, SHIFT_MAX, DEVICE)

def summarize():
    records = []
    for path in sorted(Path(LOG_DIR).rglob('best.pt')):
        stage = path.parent.parent.name
        fold = int(path.parent.name.split('_')[-1])
        if fold not in FOLDS or stage not in ('cnn_teacher', 'mlp_baseline', 'pureKD', 'kd_b0', 'kd_b025'):
            continue
        if stage == 'cnn_teacher':
            net = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
        elif stage in ('kd_b0', 'kd_b025'):
            net = build_wrap().to(DEVICE)
        else:
            net = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'])
        teacher = load_teacher_for_fold(fold)
        _, loader, _ = build_loaders_for_fold(fold)
        metrics = validate_full(net, loader, DEVICE)
        metrics.update(fidelity(net, teacher, loader, DEVICE))
        record = dict(run=str(path.parent.parent.relative_to(LOG_DIR)), fold=fold, **metrics)
        records.append(record)
    summary = {}
    for run in sorted({r['run'] for r in records}):
        rows = [r for r in records if r['run'] == run]
        summary[run] = {k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))} for k in ('accu', 'nlll', 'ecel', 'ti_1mjsd', 'ti_agree', 'fid_top1', 'fid_1mjsd')}
    with open(os.path.join(LOG_DIR, 'summary.json'), 'w') as f:
        json.dump({'folds': records, 'summary': summary}, f, indent=2)
    return records

def run(stages=('teacher', 'baseline', 'hint', 'temperature_scan', 'alpha_scan', 'alpha_temperature_scan', 'summary'), temperatures=(1, 2, 4, 8, 16)):
    prepare_data()
    runners = {'teacher': run_teacher, 'baseline': run_baseline, 'pure_kd': run_pure_kd, 'hint': run_hint, 'kd_hint': run_kd_hint, 'kd_hint_beta': run_kd_hint_beta, 'summary': summarize}
    for stage in stages:
        if stage == 'temperature_scan':
            run_temperature_scan(temperatures)
        elif stage == 'alpha_scan':
            run_alpha_scan()
        elif stage == 'alpha_temperature_scan':
            run_alpha_temperature_scan(temperatures)
        else:
            runners[stage]()
