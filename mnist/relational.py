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
VARIANT_TAG = 'vrm_w256'
SHIFT_MODES = ['none', 'shuffle']
SHIFT_SEED = 0
SHIFT_MAX = None
SEED = 42
SPLIT_SEED = 42
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))
SWEEP_FOLD = 0
SWEEP_BASE_LR = 0.001
SWEEP_EPOCHS = 80
SWEEP_PATIENCE = 15
SWEEP_CELLS = [(32, 0), (256, 8), (128, 8), (64, 8), (32, 8), (32, 16), (32, 32), (32, 64), (32, 128), (32, 256)]
SWEEP_LRS = [0.0001, 0.0003, 0.001, 0.003]
CFG = dict(input_shape=(1, 28, 28), num_classes=10, split=(55000, 5000, 10000), batch_size=128, teacher_kwargs=dict(input_channels=1, phi_channels=[32, 64, 64], rho_layers=[128], activ='relu', output_dim=10, padding_mode='circular', kernel_size=3, use_bn=False), student_kwargs=dict(input_dim=784, layers=[256, 64, 64], output_dim=10, activ='relu'), scheduler='rop', teacher=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), baseline=dict(max_epochs=200, lr=0.001, wd=0.0, monitor='val/acc', mode='max'), hint=dict(max_epochs=600, lr=0.001, wd=0.0, monitor='val/loss', mode='min'), vrm=dict(max_epochs=200, wd=0.0, monitor='val/acc', mode='max'), es_patience=30, guided_idx=3, hint_depth=3)
BEST = {}
_DEFAULT_CFG = copy.deepcopy(CFG)
DEVICE = torch.device('cpu')
LOG_DIR = '.'
DATA_DIR = '.'
_MNIST_MEAN, _MNIST_STD = (0.1307, 0.3081)
DATA_CACHE = {}
KD_OUTPUT_ROOT = None
MAX_PAIRS = 256

def apply_one_shot_translation(x, seed, max_shift=None):
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

def _load_full_dataset(data_dir, shift_mode, shift_seed, shift_max, device):
    MNIST(data_dir, train=True, download=True)
    MNIST(data_dir, train=False, download=True)
    tr = MNIST(data_dir, train=True, download=False)
    te = MNIST(data_dir, train=False, download=False)
    tr_x = (tr.data.float().unsqueeze(1) / 255.0 - _MNIST_MEAN) / _MNIST_STD
    te_x = (te.data.float().unsqueeze(1) / 255.0 - _MNIST_MEAN) / _MNIST_STD
    tr_y, te_y = (tr.targets.long(), te.targets.long())
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

def prepare_mode_data(mode):
    if mode not in DATA_CACHE:
        ax, ay = _load_full_dataset(DATA_DIR, mode, SHIFT_SEED, SHIFT_MAX, DEVICE)
        DATA_CACHE[mode] = (ax, ay)
    return DATA_CACHE[mode]

def get_loaders(mode, fold):
    ax, ay = prepare_mode_data(mode)
    tr_xy, va_xy, te_xy = fold_split(ax, ay, CFG['split'], SPLIT_SEED, N_FOLDS, fold)
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

def mode_root(mode):
    return os.path.join(LOG_DIR, mode)

def cv_root(mode):
    return os.path.join(mode_root(mode), f'cv5_{VARIANT_TAG}')

def sweep_dir(mode, phase, name):
    return os.path.join(mode_root(mode), 'vrm_sweep', phase, name)

def best_hp_path(mode):
    return os.path.join(mode_root(mode), 'vrm_sweep', 'best_hparams.json')

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

def load_teacher_for_fold(mode, fold):
    path = os.path.join(teacher_dir(mode, fold), 'best.pt')
    t = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    t.load_state_dict(ck['model_state'])
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t

def load_hint_into_kd_model(mode, fold):
    s = MLPBasic(**CFG['student_kwargs'])
    reg, gd, hd = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
    full = HintStudent(s, reg, CFG['guided_idx']).to(DEVICE)
    ck = torch.load(os.path.join(stage_dir(mode, 'hint', fold), 'best.pt'), map_location='cpu', weights_only=False)
    full.load_state_dict(ck['model_state'])
    return (full, hd)

def cls_train_step(model, batch, device):
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return (F.cross_entropy(model(x), y), y.size(0), {})

def run_teacher(mode, fold):
    P = CFG['teacher']
    teacher = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
    opt = torch.optim.Adam(teacher.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
    tr, va, _ = get_loaders(mode, fold)
    Trainer(teacher_dir(mode, fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=sstep).fit(teacher, opt, sched, cls_train_step, classifier_eval, tr, va, DEVICE)
    del teacher, opt, sched

def run_baseline(mode, fold):
    P = CFG['baseline']
    model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
    tr, va, _ = get_loaders(mode, fold)
    Trainer(stage_dir(mode, 'mlp_baseline', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=sstep).fit(model, opt, sched, cls_train_step, classifier_eval, tr, va, DEVICE)
    del model, opt, sched

def run_hint(mode, fold):
    P = CFG['hint']
    teacher = load_teacher_for_fold(mode, fold)
    student = MLPBasic(**CFG['student_kwargs'])
    regressor, gd, hd = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
    model = HintStudent(student, regressor, CFG['guided_idx']).to(DEVICE)
    tr, va, _ = get_loaders(mode, fold)
    hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
    hint_loss.compute_train_std(teacher, tr, CFG['hint_depth'], DEVICE)
    hdp = CFG['hint_depth']

    def hint_step(m, batch, device):
        x, _ = batch
        x = x.to(device)
        g = m.guided(x)
        with torch.no_grad():
            _, h = teacher.forward_with_phi_hint(x, hdp)
        return (hint_loss(g, h), x.size(0), {})

    @torch.no_grad()
    def hint_eval(m, loader, device):
        m.eval()
        s = 0.0
        n = 0
        for x, _ in loader:
            x = x.to(device)
            g = m.guided(x)
            _, h = teacher.forward_with_phi_hint(x, hdp)
            s += float(hint_loss(g, h).item()) * x.size(0)
            n += x.size(0)
        return {'val/loss': s / max(1, n)}
    opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
    Trainer(stage_dir(mode, 'hint', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=sstep).fit(model, opt, sched, hint_step, hint_eval, tr, va, DEVICE, extra_state={'hint_std': hint_loss})
    del teacher, model, opt, sched, hint_loss

def run_vrm_stage(mode, fold, stage, hp, ce_weight, hint_weight, warm_from_hint):
    P = CFG['vrm']
    teacher = load_teacher_for_fold(mode, fold)
    if warm_from_hint:
        if not os.path.isfile(os.path.join(stage_dir(mode, 'hint', fold), 'best.pt')):
            print(f'    skip {stage} fold={fold}: hint ckpt missing')
            return
        model, hd = load_hint_into_kd_model(mode, fold)
    else:
        model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        hd = CFG['teacher_kwargs']['phi_channels'][-1]
    hint_loss = None
    extra_state = None
    if hint_weight > 0:
        hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
        tr_std, _, _ = get_loaders(mode, fold)
        hint_loss.compute_train_std(teacher, tr_std, CFG['hint_depth'], DEVICE)
        extra_state = {'hint_std': hint_loss}
    step_fn = make_vrm_step(teacher, hp, ce_weight, hint_weight, hint_loss)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=P['wd'])
    sched, sstep = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
    tr, va, _ = get_loaders(mode, fold)
    Trainer(stage_dir(mode, stage, fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=sstep).fit(model, opt, sched, step_fn, classifier_eval, tr, va, DEVICE, extra_state=extra_state)
    del teacher, model, opt, sched

def best_val_acc_from_csv(run_dir):
    csvp = os.path.join(run_dir, 'metrics.csv')
    if not os.path.isfile(csvp):
        return float('nan')
    best = float('-inf')
    with open(csvp) as f:
        for row in csv.DictReader(f):
            v = row.get('val/acc')
            if v in (None, ''):
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if fv > best:
                best = fv
    return best if best != float('-inf') else float('nan')

def run_sweep_cell(mode, run_dir, hp):
    teacher = load_teacher_for_fold(mode, SWEEP_FOLD)
    model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
    step_fn = make_vrm_step(teacher, hp, ce_weight=1.0, hint_weight=0.0, hint_loss=None)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=0.0)
    sched, sstep = build_scheduler(opt, CFG['scheduler'], SWEEP_EPOCHS)
    tr, va, _ = get_loaders(mode, SWEEP_FOLD)
    Trainer(run_dir, max_epochs=SWEEP_EPOCHS, monitor='val/acc', mode='max', es_patience=SWEEP_PATIENCE, scheduler_step=sstep).fit(model, opt, sched, step_fn, classifier_eval, tr, va, DEVICE)
    del teacher, model, opt, sched

def vrm_sweep(mode):
    bp = best_hp_path(mode)
    if os.path.isfile(bp):
        with open(bp) as f:
            best = json.load(f)
        return best
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    p1 = []
    for li, lc in SWEEP_CELLS:
        rd = sweep_dir(mode, 'phase1', f'a{li}_b{lc}')
        hp = dict(lam_is=float(li), lam_ic=float(lc), lr=float(SWEEP_BASE_LR))
        run_sweep_cell(mode, rd, hp)
        acc = best_val_acc_from_csv(rd)
        p1.append((li, lc, acc))
    p1f = [r for r in p1 if r[2] == r[2]]
    best_is, best_ic, _ = max(p1f, key=lambda r: r[2])
    p2 = []
    for lr in SWEEP_LRS:
        rd = sweep_dir(mode, 'phase2', f'a{best_is}_b{best_ic}_lr{lr}')
        hp = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(lr))
        run_sweep_cell(mode, rd, hp)
        acc = best_val_acc_from_csv(rd)
        p2.append((lr, acc))
    p2f = [r for r in p2 if r[1] == r[1]]
    best_lr, best_acc = max(p2f, key=lambda r: r[1])
    best = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(best_lr), val_acc=float(best_acc), ce_weight=1.0)
    with open(bp, 'w') as f:
        json.dump(best, f, indent=2)
    return best

def get_best(mode):
    if mode in BEST and BEST[mode] is not None:
        return BEST[mode]
    bp = best_hp_path(mode)
    if os.path.isfile(bp):
        with open(bp) as f:
            BEST[mode] = json.load(f)
        return BEST[mode]
    raise RuntimeError(f'no tuned hp for mode={mode}; run the sweep cell first')

def run_sweep_cell_ce(mode, run_dir, hp, ce_weight):
    teacher = load_teacher_for_fold(mode, SWEEP_FOLD)
    model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
    step_fn = make_vrm_step(teacher, hp, ce_weight=ce_weight, hint_weight=0.0, hint_loss=None)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=0.0)
    sched, sstep = build_scheduler(opt, CFG['scheduler'], SWEEP_EPOCHS)
    tr, va, _ = get_loaders(mode, SWEEP_FOLD)
    Trainer(run_dir, max_epochs=SWEEP_EPOCHS, monitor='val/acc', mode='max', es_patience=SWEEP_PATIENCE, scheduler_step=sstep).fit(model, opt, sched, step_fn, classifier_eval, tr, va, DEVICE)
    del teacher, model, opt, sched

def vrm_sweep_ce(mode, ce_weight):
    bp = os.path.join(mode_root(mode), f'vrm_sweep_ce{ce_weight}', 'best_hparams.json')
    if os.path.isfile(bp):
        with open(bp) as f:
            best = json.load(f)
        return best
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    p1 = []
    for li, lc in SWEEP_CELLS:
        rd = os.path.join(mode_root(mode), f'vrm_sweep_ce{ce_weight}', 'phase1', f'a{li}_b{lc}')
        hp = dict(lam_is=float(li), lam_ic=float(lc), lr=float(SWEEP_BASE_LR))
        run_sweep_cell_ce(mode, rd, hp, ce_weight)
        acc = best_val_acc_from_csv(rd)
        p1.append((li, lc, acc))
    p1f = [r for r in p1 if r[2] == r[2]]
    best_is, best_ic, _ = max(p1f, key=lambda r: r[2])
    p2 = []
    for lr in SWEEP_LRS:
        rd = os.path.join(mode_root(mode), f'vrm_sweep_ce{ce_weight}', 'phase2', f'a{best_is}_b{best_ic}_lr{lr}')
        hp = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(lr))
        run_sweep_cell_ce(mode, rd, hp, ce_weight)
        acc = best_val_acc_from_csv(rd)
        p2.append((lr, acc))
    p2f = [r for r in p2 if r[1] == r[1]]
    best_lr, best_acc = max(p2f, key=lambda r: r[1])
    best = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(best_lr), val_acc=float(best_acc), ce_weight=ce_weight)
    with open(bp, 'w') as f:
        json.dump(best, f, indent=2)
    return best

def run_vrm_stage_custom(mode, fold, stage, hp, ce_weight, hint_weight, warm_from_hint, out_dir):
    P = CFG['vrm']
    teacher = load_teacher_for_fold(mode, fold)
    if warm_from_hint:
        model, hd = load_hint_into_kd_model(mode, fold)
    else:
        model = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        hd = CFG['teacher_kwargs']['phi_channels'][-1]
    hint_loss = None
    extra_state = None
    if hint_weight > 0:
        hint_loss = NormalizedHintMSE(hint_dim=hd).to(DEVICE)
        tr_std, _, _ = get_loaders(mode, fold)
        hint_loss.compute_train_std(teacher, tr_std, CFG['hint_depth'], DEVICE)
        extra_state = {'hint_std': hint_loss}
    step_fn = make_vrm_step(teacher, hp, ce_weight, hint_weight, hint_loss)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=P['wd'])
    sched, sstep = build_scheduler(opt, CFG['scheduler'], P['max_epochs'])
    tr, va, _ = get_loaders(mode, fold)
    Trainer(out_dir, max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=CFG['es_patience'], scheduler_step=sstep).fit(model, opt, sched, step_fn, classifier_eval, tr, va, DEVICE, extra_state=extra_state)
    del teacher, model, opt, sched

def build_scheduler(opt, kind, max_epochs):
    if kind == 'warmup_cosine_1pct':
        initial_lr = opt.param_groups[0]['lr']
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=3)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs - 3, eta_min=initial_lr * 0.01)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[3])
        return (sched, 'epoch')
    elif kind == 'cosine':
        return (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-05), 'epoch')
    elif kind == 'rop':
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3), 'plateau')
    raise ValueError(kind)

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
        p_s = F.softmax(sl, dim=1)
        p_t = F.softmax(tl, dim=1)
        agree += float((torch.argmax(sl, dim=1) == torch.argmax(tl, dim=1)).sum().item())
        jsd_sum += float(_jsd(p_s, p_t).sum().item())
        total += int(x.size(0))
    return dict(fid_top1=agree / total, fid_1mjsd=1.0 - jsd_sum / max(1, total))

def build_wrap():
    s = MLPBasic(**CFG['student_kwargs'])
    r, _, _ = build_regressor(CFG['student_kwargs'], CFG['teacher_kwargs'], CFG['guided_idx'])
    return HintStudent(s, r, CFG['guided_idx'])

def vrm_loss(z_s_real, z_t_real, eps=1e-08):
    B = z_s_real.size(0)
    ss = z_s_real @ z_s_real.t()
    tt = z_t_real @ z_t_real.t()
    st = z_s_real @ z_t_real.t()
    s_sq = (z_s_real * z_s_real).sum(-1)
    t_sq = (z_t_real * z_t_real).sum(-1)
    pair = (z_s_real * z_t_real).sum(-1)
    norm_s_sq = (s_sq.unsqueeze(1) + s_sq.unsqueeze(0) - 2.0 * ss).clamp_min(eps)
    norm_t_sq = (t_sq.unsqueeze(1) + t_sq.unsqueeze(0) - 2.0 * tt).clamp_min(eps)
    cross = pair.unsqueeze(1) + pair.unsqueeze(0) - st - st.t()
    cos = (cross / torch.sqrt(norm_s_sq * norm_t_sq)).clamp(-1.0, 1.0)
    is_err = 2.0 - 2.0 * cos
    L_IS = (is_err.sum() - is_err.diagonal().sum()) / float(max(1, B * (B - 1)))
    Ws = z_s_real.t().contiguous()
    Wt = z_t_real.t().contiguous()
    w_sq = (Ws * Ws).sum(-1)
    wt_sq = (Wt * Wt).sum(-1)
    W_dot = Ws @ Ws.t()
    WT_dot = Wt @ Wt.t()
    norm_w_sq = (w_sq.unsqueeze(1) + w_sq.unsqueeze(0) - 2.0 * W_dot).clamp_min(eps)
    norm_wt_sq = (wt_sq.unsqueeze(1) + wt_sq.unsqueeze(0) - 2.0 * WT_dot).clamp_min(eps)
    W_WT_diag = (Ws * Wt).sum(-1)
    W_WT = Ws @ Wt.t()
    cross_ic = W_WT_diag.unsqueeze(1) + W_WT_diag.unsqueeze(0) - W_WT - W_WT.t()
    cos_ic = (cross_ic / torch.sqrt(norm_w_sq * norm_wt_sq)).clamp(-1.0, 1.0)
    ic_err = 2.0 - 2.0 * cos_ic
    C = ic_err.size(0)
    L_IC = (ic_err.sum() - ic_err.diagonal().sum()) / float(max(1, C * (C - 1)))
    return (L_IS, L_IC)

def make_vrm_step(teacher, hp, ce_weight, hint_weight, hint_loss=None):
    lam_is = float(hp['lam_is'])
    lam_ic = float(hp['lam_ic'])
    guided_idx = int(CFG['guided_idx'])
    hint_depth = int(CFG['hint_depth'])
    ce_weight = float(ce_weight)
    hint_weight = float(hint_weight)

    def step(model, batch, device):
        x, y = batch
        x, y = (x.to(device), y.to(device))
        if hint_weight > 0:
            sl, guided_raw = model.student.forward_with_guided(x, guided_idx)
            guided = model.regressor(guided_raw)
            with torch.no_grad():
                tl, hint = teacher.forward_with_phi_hint(x, hint_depth)
        else:
            sl = model(x)
            with torch.no_grad():
                tl = teacher(x)
        if 0 < MAX_PAIRS < x.size(0):
            idx = torch.randint(x.size(0), (MAX_PAIRS,), device=x.device)
            sr, tr = sl[idx], tl[idx]
        else:
            sr, tr = sl, tl
        L_IS, L_IC = vrm_loss(sr, tr)
        L_CE = F.cross_entropy(sl, y) if ce_weight > 0 else sl.new_zeros(())
        L_main = lam_is * L_IS + lam_ic * L_IC + ce_weight * L_CE
        if hint_weight > 0:
            L_hint = hint_loss(guided, hint)
            scale = L_main.detach() / L_hint.detach().clamp_min(1e-12)
            loss = (1 - hint_weight) * L_main + hint_weight * scale * L_hint
        else:
            loss = L_main
        return (loss, x.size(0), {})
    return step

def old_cv_root(mode):
    if KD_OUTPUT_ROOT is None:
        return cv_root(mode)
    return os.path.join(str(KD_OUTPUT_ROOT), f'mnist_cv5_w256_{mode}_s{SHIFT_SEED}', 'cv5_w256')

def teacher_dir(mode, fold):
    return os.path.join(old_cv_root(mode), 'cnn_teacher', f'fold_{fold}')

def stage_dir(mode, stage, fold):
    root = old_cv_root(mode) if stage == 'mlp_baseline' else cv_root(mode)
    return os.path.join(root, stage, f'fold_{fold}')

def configure(data_root, output_root, regime='transformed', folds=(0, 1, 2, 3, 4), device=None, kd_output_root=None):
    global CFG, DATA_DIR, LOG_DIR, SHIFT_MODES, FOLDS, DEVICE, DATA_CACHE, BEST, KD_OUTPUT_ROOT
    CFG = copy.deepcopy(_DEFAULT_CFG)
    SHIFT_MODES = [{'canonical': 'none', 'transformed': 'shuffle'}[regime]]
    FOLDS = list(folds)
    DEVICE = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    DATA_DIR = str(data_root)
    LOG_DIR = os.path.join(str(output_root), f'mnist_{VARIANT_TAG}_s{SHIFT_SEED}')
    KD_OUTPUT_ROOT = kd_output_root
    os.makedirs(LOG_DIR, exist_ok=True)
    DATA_CACHE = {}
    BEST = {}
    torch.manual_seed(SEED)
    random.seed(SEED)

def summarize(mode):
    records = []
    paths = []
    for fold in FOLDS:
        paths.append(('teacher', fold, os.path.join(teacher_dir(mode, fold), 'best.pt')))
        paths.append(('mlp_baseline', fold, os.path.join(stage_dir(mode, 'mlp_baseline', fold), 'best.pt')))
    for root in (Path(cv_root(mode)), Path(cv_root(mode) + '_ce0')):
        for path in root.glob('*/fold_*/best.pt'):
            if path.parent.parent.name in ('vrm', 'vrm_hint', 'vrm_hint_b025'):
                paths.append((str(path.parent.parent.relative_to(mode_root(mode))), int(path.parent.name.split('_')[-1]), str(path)))
    for stage, fold, path in paths:
        if fold not in FOLDS or not os.path.isfile(path):
            continue
        if stage == 'teacher':
            net = CNNInvariant(**CFG['teacher_kwargs']).to(DEVICE)
        elif 'hint' in stage:
            net = build_wrap().to(DEVICE)
        else:
            net = MLPBasic(**CFG['student_kwargs']).to(DEVICE)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'])
        teacher = load_teacher_for_fold(mode, fold)
        _, loader, _ = get_loaders(mode, fold)
        metrics = validate_full(net, loader, DEVICE)
        metrics.update(fidelity(net, teacher, loader, DEVICE))
        record = dict(stage=stage, fold=fold, **metrics)
        records.append(record)
    summary = {}
    for stage in sorted({r['stage'] for r in records}):
        rows = [r for r in records if r['stage'] == stage]
        summary[stage] = {k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))} for k in ('accu', 'nlll', 'ecel', 'ti_1mjsd', 'ti_agree', 'fid_top1', 'fid_1mjsd')}
    os.makedirs(mode_root(mode), exist_ok=True)
    with open(os.path.join(mode_root(mode), 'summary.json'), 'w') as f:
        json.dump({'folds': records, 'summary': summary}, f, indent=2)
    return records

def extend_search(mode, ce_weight):
    if ce_weight == 0:
        best = vrm_sweep_ce(mode, 0.0)
        lrs = (0.008, 0.01, 0.03)
        bp = os.path.join(mode_root(mode), 'vrm_sweep_ce0.0', 'best_hparams.json')
        root = os.path.join(mode_root(mode), 'vrm_sweep_ce0.0', 'phase2')
    else:
        best = get_best(mode)
        if mode != 'shuffle':
            return best
        lrs = (0.005, 0.008)
        bp = best_hp_path(mode)
        root = sweep_dir(mode, 'phase2', '')
    best_is, best_ic = int(best['lam_is']), int(best['lam_ic'])
    for lr in lrs:
        rd = os.path.join(root, f'a{best_is}_b{best_ic}_lr{lr}_custom')
        hp = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(lr))
        run_sweep_cell_ce(mode, rd, hp, ce_weight)
        acc = best_val_acc_from_csv(rd)
        if acc > best['val_acc']:
            best = dict(hp, val_acc=acc, ce_weight=ce_weight)
    with open(bp, 'w') as f:
        json.dump(best, f, indent=2)
    if ce_weight != 0:
        BEST[mode] = best
    return best

def run(stages=('teacher', 'search', 'baseline', 'hint', 'relational', 'search_ce0', 'relational_ce0', 'summary')):
    for stage in stages:
        CFG['scheduler'] = 'rop' if stage in ('teacher', 'search') else 'warmup_cosine_1pct'
        for mode in SHIFT_MODES:
            if stage in ('teacher', 'baseline', 'hint'):
                for fold in FOLDS:
                    {'teacher': run_teacher, 'baseline': run_baseline, 'hint': run_hint}[stage](mode, fold)
            elif stage == 'search':
                BEST[mode] = vrm_sweep(mode)
                extend_search(mode, 1.0)
            elif stage == 'search_ce0':
                vrm_sweep_ce(mode, 0.0)
                extend_search(mode, 0.0)
            elif stage == 'relational':
                hp = get_best(mode)
                for fold in FOLDS:
                    run_vrm_stage(mode, fold, 'vrm', hp, 1.0, 0.0, False)
                    run_vrm_stage(mode, fold, 'vrm_hint', hp, 1.0, 0.0, True)
                    run_vrm_stage(mode, fold, 'vrm_hint_b025', hp, 1.0, 0.25, True)
            elif stage == 'relational_ce0':
                hp = vrm_sweep_ce(mode, 0.0)
                for fold in FOLDS:
                    for name, beta, warm in (('vrm', 0.0, False), ('vrm_hint', 0.0, True), ('vrm_hint_b025', 0.25, True)):
                        out = os.path.join(cv_root(mode) + '_ce0', name, f'fold_{fold}')
                        run_vrm_stage_custom(mode, fold, name, hp, 0.0, beta, warm, out)
            elif stage == 'summary':
                summarize(mode)
            else:
                raise ValueError(stage)
