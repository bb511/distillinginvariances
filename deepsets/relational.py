import os, csv, json, math, random, tarfile, glob, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import validate_full, compute_fidelity, save_results
VARIANT_TAG = 'vrm_d3w64h2'
STUDENT_LAYERS = [64, 32, 32, 32]
GUIDED_IDX = 3
SEED = 325
FOLD_SEED = 0
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))
PERM_MODES = ['perm', 'noperm']
PERMUTE_SEED = 0
X_TRAIN_FILE = 'x_train_robust_fast_32const_ptetaphi.npy'
Y_TRAIN_FILE = 'y_train_robust_fast_32const_ptetaphi.npy'
ZENODO_TRAIN_URL = 'https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz'
SWEEP_FOLD = 0
SWEEP_BASE_LR = 0.008
SWEEP_EPOCHS = 100
SWEEP_PATIENCE = 20
SWEEP_CELLS = [(256, 8), (128, 8), (64, 8), (32, 8), (128, 16), (64, 16), (32, 16), (32, 16), (32, 32), (32, 64), (32, 128), (32, 256)]
SWEEP_LRS = [0.001, 0.003, 0.008, 0.01, 0.05]
CFG = dict(nconst=32, num_classes=5, input_dim=96, student_layers=STUDENT_LAYERS, teacher_kwargs=dict(input_dim=3, phi_layers=[32, 32, 32], rho_layers=[32], activ='relu', aggreg='mean', dropout=0.0, output_dim=5), guided_idx=GUIDED_IDX, hint_depth=3, teacher=dict(max_epochs=200, lr=0.0032, wd=0.0, batch_size=128, sched=('plateau', 'min', 30, 'val/loss'), monitor='val/acc', mode='max', es_patience=30), baseline=dict(max_epochs=200, lr=0.01, wd=0.0, batch_size=1024, sched=('plateau', 'min', 30, 'val/loss'), monitor='val/acc', mode='max', es_patience=30), hint=dict(max_epochs=200, lr=0.001, wd=0.0, batch_size=1024, sched=('plateau', 'min', 30, 'val/loss'), monitor='val/loss', mode='min', es_patience=30), vrm=dict(max_epochs=300, wd=0.0, batch_size=1024, monitor='val/acc', mode='max', es_patience=60))
BEST = {}
VRM_FIXED = dict(max_pairs=4096, warmup_steps=10, eta_min_ratio=0.01)
DEVICE = torch.device('cpu')
JET_DATA_DIR = 'data/jetid/processed'
RAW_DIR = 'data/jetid/raw/train'
LOG_ROOT = 'paper_runs/deepsets/relational'
BEST_CE0 = {}
_NCONST = 32
_MIN_PT = 2
_FEAT_IDX = [5, 8, 11]
_JET_CACHE = {}
_FOLD_CACHE = {}

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

def _proc_paths():
    return (os.path.join(JET_DATA_DIR, X_TRAIN_FILE), os.path.join(JET_DATA_DIR, Y_TRAIN_FILE))

def _download(url, dst):
    if not os.path.isfile(dst):
        urllib.request.urlretrieve(url, dst)

def _cut_pt(x_data, y_data):
    mask = x_data[:, :, 5] > _MIN_PT
    structure = mask.sum(axis=1)
    x_split = np.split(x_data[mask, :], np.cumsum(structure)[:-1])
    x_list = [jc for jc in x_split if jc.size > 0]
    y_data = y_data[structure > 0]
    return (x_list, y_data)

def _restrict(x_list):
    out = []
    for jet in x_list:
        if jet.shape[0] >= _NCONST:
            out.append(jet[:_NCONST, :])
        else:
            out.append(np.pad(jet, ((0, _NCONST - jet.shape[0]), (0, 0))))
    return np.array(out)

def _build_processed():
    import h5py
    os.makedirs(RAW_DIR, exist_ok=True)
    tar_path = os.path.join(RAW_DIR, 'hls4ml_LHCjet_150p_train.tar.gz')
    h5files = sorted(glob.glob(os.path.join(RAW_DIR, '**', '*.h5'), recursive=True))
    if not h5files:
        _download(ZENODO_TRAIN_URL, tar_path)
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(RAW_DIR, filter='data')
        h5files = sorted(glob.glob(os.path.join(RAW_DIR, '**', '*.h5'), recursive=True))
    x_parts, y_parts = ([], [])
    for fp in h5files:
        with h5py.File(fp, 'r') as d:
            xr = np.array(d['jetConstituentList'])
            yr = np.array(d['jets'])[:, -6:-1]
        xl, yd = _cut_pt(xr, yr)
        xp = _restrict(xl)
        x_parts.append(xp[:, :, _FEAT_IDX].astype(np.float32))
        y_parts.append(yd.astype(np.float32))
        del xr, yr, xl, xp
    x_sel = np.concatenate(x_parts, axis=0)
    y_data = np.concatenate(y_parts, axis=0)
    iqr = np.array([np.subtract(*np.nanpercentile(x_sel[:, :, f].flatten(), [95, 5])) for f in range(x_sel.shape[-1])], dtype=np.float32)
    x_std = (x_sel / np.where(iqr > 0, iqr, 1.0)).astype(np.float32)
    xp_path, yp_path = _proc_paths()
    np.save(xp_path, x_std)
    np.save(yp_path, y_data)

def ensure_jet_data():
    xp, yp = _proc_paths()
    if os.path.isfile(xp) and os.path.isfile(yp):
        return
    _build_processed()

def _load_jet_train():
    ensure_jet_data()
    xp, yp = _proc_paths()
    x = torch.from_numpy(np.load(xp).astype(np.float32))
    y = torch.from_numpy(np.load(yp).astype(np.float32))
    return (x, y)

def prepare_jet_data():
    if 'xy' not in _JET_CACHE:
        _JET_CACHE['xy'] = _load_jet_train()
    return _JET_CACHE['xy']

def jet_fold_split(all_x, all_y, fold_seed, n_folds, fold_idx):
    n = all_x.size(0)
    g = torch.Generator().manual_seed(int(fold_seed))
    perm = torch.randperm(n, generator=g)
    x, y = (all_x[perm], all_y[perm])
    fold_size = n // n_folds
    vs = fold_idx * fold_size
    ve = n if fold_idx == n_folds - 1 else vs + fold_size
    mask = torch.zeros(n, dtype=torch.bool)
    mask[vs:ve] = True
    return ((x[~mask].contiguous(), y[~mask].contiguous()), (x[mask].contiguous(), y[mask].contiguous()))

def _apply_perm(t, g):
    b, n, c = t.shape
    perm = torch.argsort(torch.rand(b, n, generator=g), dim=1)
    return torch.gather(t, 1, perm.unsqueeze(-1).expand(b, n, c))

def get_fold(mode, fold):
    key = (mode, fold)
    if key not in _FOLD_CACHE:
        ax, ay = prepare_jet_data()
        (trx, tryy), (vax, vay) = jet_fold_split(ax, ay, FOLD_SEED, N_FOLDS, fold)
        if mode == 'perm':
            g = torch.Generator().manual_seed(int(PERMUTE_SEED))
            trx = _apply_perm(trx, g)
            vax = _apply_perm(vax, g)
        elif mode != 'noperm':
            raise ValueError(f'unknown mode {mode!r}')
        _FOLD_CACHE[key] = (trx.contiguous(), tryy.contiguous(), vax.contiguous(), vay.contiguous())
    return _FOLD_CACHE[key]

def get_loaders(mode, fold, batch_size):
    trx, tryy, vax, vay = get_fold(mode, fold)
    return (GPUBatchLoader(trx.to(DEVICE), tryy.to(DEVICE), batch_size, True), GPUBatchLoader(vax.to(DEVICE), vay.to(DEVICE), batch_size, False))

def _activation(name):
    return {'relu': lambda: nn.ReLU(inplace=True), 'tanh': lambda: nn.Tanh(), 'sigmoid': lambda: nn.Sigmoid(), 'leaky_relu': lambda: nn.LeakyReLU()}[name]()

class DeepSetsInvariant(nn.Module):

    def __init__(self, input_dim, phi_layers, rho_layers, activ, aggreg, dropout, output_dim):
        super().__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = list(phi_layers)
        self.rho_layers = list(rho_layers)
        self.phi = self._construct_phi()
        self.agg = {'mean': torch.mean, 'max': torch.max}[aggreg]
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self):
        phi = nn.Sequential()
        layers = [self.input_dim] + self.phi_layers
        for i in range(len(layers) - 1):
            phi.append(nn.Linear(layers[i], layers[i + 1]))
            phi.append(_activation(self.activ))
        return phi

    def _construct_rho(self, dropout):
        rho = nn.Sequential()
        layers = [self.phi_layers[-1]] + self.rho_layers + [self.output_dim]
        for i in range(len(layers) - 1):
            if 0 < dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(nn.Linear(layers[i], layers[i + 1]))
            if i == len(layers) - 2:
                break
            rho.append(_activation(self.activ))
        return rho

    def _agg(self, t):
        out = self.agg(t, dim=1)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x):
        return self.rho(self._agg(self.phi(x)))

    def forward_with_phi_hint(self, x, phi_depth=3):
        h = x
        hint = None
        target_idx = phi_depth * 2 - 1
        for i, layer in enumerate(self.phi):
            h = layer(h)
            if i == target_idx:
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

def build_teacher():
    return DeepSetsInvariant(**CFG['teacher_kwargs'])

def build_student():
    return MLPBasic(input_dim=CFG['input_dim'], layers=CFG['student_layers'], output_dim=CFG['num_classes'], activ='relu')

@torch.no_grad()
def cls_eval_jet(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    loss_sum = 0.0
    correct = 0
    total = 0
    agree_sum = 0.0
    nb = 0
    for x, y in loader:
        x, y = (x.to(device), y.to(device))
        logits = model(x)
        loss_sum += float(ce(logits, y).item())
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == torch.argmax(y, dim=1)).sum().item())
        total += int(y.size(0))
        a = 0.0
        for _ in range(3):
            perm = torch.randperm(x.size(1), device=x.device)
            a += (torch.argmax(model(x[:, perm]), dim=1) == preds).float().mean().item()
        agree_sum += a / 3
        nb += 1
    return {'val/loss': loss_sum / max(1, total), 'val/acc': correct / max(1, total), 'val/pi_agree': agree_sum / max(1, nb)}

class Trainer:

    def __init__(self, run_dir, max_epochs, monitor='val/loss', mode='min', es_patience=30, scheduler_step='epoch', plateau_monitor='val/loss'):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.max_epochs = max_epochs
        self.monitor = monitor
        assert mode in ('min', 'max')
        self.mode = mode
        self.es_patience = es_patience
        self.scheduler_step = scheduler_step
        self.plateau_monitor = plateau_monitor
        self.last_path = os.path.join(run_dir, 'last.pt')
        self.best_path = os.path.join(run_dir, 'best.pt')
        self.csv_path = os.path.join(run_dir, 'metrics.csv')
        self.done_path = os.path.join(run_dir, 'done.txt')
        self.cols = ['epoch', 'lr', 'train/loss', 'val/loss', 'val/acc', 'val/pi_agree', 'best']

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
            self._row({'epoch': epoch, 'lr': cur_lr, 'train/loss': train_loss, 'val/loss': metrics.get('val/loss', ''), 'val/acc': metrics.get('val/acc', ''), 'val/pi_agree': metrics.get('val/pi_agree', ''), 'best': best})
            print(f'{os.path.relpath(self.run_dir, LOG_ROOT)} epoch={epoch} loss={train_loss:.4f} {self.monitor}={mv:.4f} lr={cur_lr:.2g}')
            if scheduler is not None:
                if self.scheduler_step == 'plateau':
                    scheduler.step(metrics.get(self.plateau_monitor, train_loss))
                else:
                    scheduler.step()
            ck = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict() if scheduler is not None else None, 'best': best, 'patience_counter': pcount}
            if extra_state is not None:
                ck['extra_state'] = {k: v.state_dict() if hasattr(v, 'state_dict') else v for k, v in extra_state.items()}
            torch.save(ck, self.last_path)
            if improved:
                torch.save(ck, self.best_path)
            if pcount >= self.es_patience:
                break
        with open(self.done_path, 'w') as f:
            f.write(f'finished_epoch={epoch}\n')

def mode_root(mode):
    return os.path.join(LOG_ROOT, mode)

def cv_root(mode):
    return os.path.join(mode_root(mode), f'cv5_{VARIANT_TAG}')

def teacher_dir(mode, fold):
    return os.path.join(cv_root(mode), 'teacher', f'fold_{fold}')

def stage_dir(mode, stage, fold):
    return os.path.join(cv_root(mode), stage, f'fold_{fold}')

def sweep_dir(mode, phase, name):
    return os.path.join(mode_root(mode), 'vrm_sweep', phase, name)

def best_hp_path(mode):
    return os.path.join(mode_root(mode), 'vrm_sweep', 'best_hparams.json')

def build_plateau(opt, cfg_sched):
    _, mode, patience, monitor = cfg_sched
    s = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.1, patience=patience)
    return (s, 'plateau', monitor)

def build_warmup_cosine(opt, peak_lr, warmup_epochs, max_epochs, eta_min=0.0):
    we = max(1, int(warmup_epochs))
    floor = eta_min / peak_lr if peak_lr > 0 else 0.0

    def fn(epoch):
        if epoch < we:
            return float(epoch + 1) / float(we)
        progress = float(epoch - we) / float(max(1, max_epochs - we))
        progress = min(1.0, progress)
        cosv = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosv
    s = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=fn)
    return (s, 'epoch', 'val/loss')

def build_regressor(guided_idx):
    all_layers = [CFG['input_dim']] + list(CFG['student_layers']) + [CFG['num_classes']]
    guided_dim = all_layers[(guided_idx + 2) // 2]
    hint_dim = CFG['teacher_kwargs']['phi_layers'][-1]
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
    t = build_teacher().to(DEVICE)
    ck = torch.load(os.path.join(teacher_dir(mode, fold), 'best.pt'), map_location='cpu', weights_only=False)
    t.load_state_dict(ck['model_state'])
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t

def load_hint_into_kd_model(mode, fold):
    s = build_student()
    reg, gd, hd = build_regressor(CFG['guided_idx'])
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
    model = build_teacher().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep, pm = build_plateau(opt, P['sched'])
    tr, va = get_loaders(mode, fold, P['batch_size'])
    Trainer(teacher_dir(mode, fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=P['es_patience'], scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, cls_train_step, cls_eval_jet, tr, va, DEVICE)
    del model, opt, sched, tr, va

def run_baseline(mode, fold):
    P = CFG['baseline']
    model = build_student().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep, pm = build_plateau(opt, P['sched'])
    tr, va = get_loaders(mode, fold, P['batch_size'])
    Trainer(stage_dir(mode, 'baseline', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=P['es_patience'], scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, cls_train_step, cls_eval_jet, tr, va, DEVICE)
    del model, opt, sched, tr, va

def run_hint(mode, fold):
    P = CFG['hint']
    teacher = load_teacher_for_fold(mode, fold)
    student = build_student()
    regressor, gd, hd = build_regressor(CFG['guided_idx'])
    model = HintStudent(student, regressor, CFG['guided_idx']).to(DEVICE)
    mse = nn.MSELoss()
    hdp = CFG['hint_depth']

    def hint_step(m, batch, device):
        x, _ = batch
        x = x.to(device)
        g = m.guided(x)
        with torch.no_grad():
            _, h = teacher.forward_with_phi_hint(x, hdp)
        return (mse(g, h), x.size(0), {})

    @torch.no_grad()
    def hint_eval(m, loader, device):
        m.eval()
        s = 0.0
        n = 0
        for x, _ in loader:
            x = x.to(device)
            g = m.guided(x)
            _, h = teacher.forward_with_phi_hint(x, hdp)
            s += float(mse(g, h).item()) * x.size(0)
            n += x.size(0)
        return {'val/loss': s / max(1, n)}
    opt = torch.optim.Adam(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched, sstep, pm = build_plateau(opt, P['sched'])
    tr, va = get_loaders(mode, fold, P['batch_size'])
    Trainer(stage_dir(mode, 'hint', fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=P['es_patience'], scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, hint_step, hint_eval, tr, va, DEVICE)
    del teacher, model, opt, sched, tr, va

def run_vrm_stage(mode, fold, stage, hp, ce_weight, hint_weight, warm_from_hint):
    P = CFG['vrm']
    teacher = load_teacher_for_fold(mode, fold)
    if warm_from_hint:
        if not os.path.isfile(os.path.join(stage_dir(mode, 'hint', fold), 'best.pt')):
            print(f'    skip {stage} fold={fold}: hint ckpt missing')
            return
        model, hd = load_hint_into_kd_model(mode, fold)
    else:
        model = build_student().to(DEVICE)
    hint_loss = nn.MSELoss() if hint_weight > 0 else None
    step_fn = make_vrm_step(teacher, hp, ce_weight, hint_weight, hint_loss)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=P['wd'])
    warmup_steps = int(VRM_FIXED['warmup_steps'])
    eta_min = float(hp['lr']) * float(VRM_FIXED['eta_min_ratio'])
    sched, sstep, pm = build_warmup_cosine(opt, float(hp['lr']), warmup_steps, P['max_epochs'], eta_min)
    tr, va = get_loaders(mode, fold, P['batch_size'])
    Trainer(stage_dir(mode, stage, fold), max_epochs=P['max_epochs'], monitor=P['monitor'], mode=P['mode'], es_patience=P['es_patience'], scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, step_fn, cls_eval_jet, tr, va, DEVICE)
    del teacher, model, opt, sched, tr, va

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
    model = build_student().to(DEVICE)
    step_fn = make_vrm_step(teacher, hp, ce_weight=1.0, hint_weight=0.0, hint_loss=None)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=0.0)
    warmup_steps = int(VRM_FIXED['warmup_steps'])
    eta_min = float(hp['lr']) * float(VRM_FIXED['eta_min_ratio'])
    sched, sstep, pm = build_warmup_cosine(opt, float(hp['lr']), warmup_steps, SWEEP_EPOCHS, eta_min)
    tr, va = get_loaders(mode, SWEEP_FOLD, CFG['vrm']['batch_size'])
    Trainer(run_dir, max_epochs=SWEEP_EPOCHS, monitor='val/acc', mode='max', es_patience=SWEEP_PATIENCE, scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, step_fn, cls_eval_jet, tr, va, DEVICE)
    del teacher, model, opt, sched, tr, va

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
        run_sweep_cell(mode, rd, dict(lam_is=float(li), lam_ic=float(lc), lr=float(SWEEP_BASE_LR)))
        acc = best_val_acc_from_csv(rd)
        p1.append((li, lc, acc))
    p1f = [r for r in p1 if r[2] == r[2]]
    best_is, best_ic, _ = max(p1f, key=lambda r: r[2])
    p2 = []
    for lr in SWEEP_LRS:
        rd = sweep_dir(mode, 'phase2', f'a{best_is}_b{best_ic}_lr{lr}')
        run_sweep_cell(mode, rd, dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(lr)))
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

def run_sweep_cell_ce0(mode, run_dir, hp):
    teacher = load_teacher_for_fold(mode, SWEEP_FOLD)
    model = build_student().to(DEVICE)
    step_fn = make_vrm_step(teacher, hp, ce_weight=0.0, hint_weight=0.0, hint_loss=None)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp['lr']), weight_decay=0.0)
    warmup_steps = int(VRM_FIXED['warmup_steps'])
    eta_min = float(hp['lr']) * float(VRM_FIXED['eta_min_ratio'])
    sched, sstep, pm = build_warmup_cosine(opt, float(hp['lr']), warmup_steps, SWEEP_EPOCHS, eta_min)
    tr, va = get_loaders(mode, SWEEP_FOLD, CFG['vrm']['batch_size'])
    Trainer(run_dir, max_epochs=SWEEP_EPOCHS, monitor='val/acc', mode='max', es_patience=SWEEP_PATIENCE, scheduler_step=sstep, plateau_monitor=pm).fit(model, opt, sched, step_fn, cls_eval_jet, tr, va, DEVICE)
    del teacher, model, opt, sched, tr, va

def vrm_sweep_ce0(mode):
    bp = os.path.join(mode_root(mode), 'vrm_sweep_ce0', 'best_hparams_ce0.json')
    if os.path.isfile(bp):
        with open(bp) as f:
            best = json.load(f)
        return best
    p1 = []
    for li, lc in SWEEP_CELLS:
        rd = os.path.join(mode_root(mode), 'vrm_sweep_ce0', 'phase1', f'a{li}_b{lc}')
        run_sweep_cell_ce0(mode, rd, dict(lam_is=float(li), lam_ic=float(lc), lr=float(SWEEP_BASE_LR)))
        acc = best_val_acc_from_csv(rd)
        p1.append((li, lc, acc))
    p1f = [r for r in p1 if r[2] == r[2]]
    if not p1f:
        raise RuntimeError(f'{mode}: phase 1 produced no valid results')
    best_is, best_ic, _ = max(p1f, key=lambda r: r[2])
    p2 = []
    for lr in SWEEP_LRS:
        rd = os.path.join(mode_root(mode), 'vrm_sweep_ce0', 'phase2', f'a{best_is}_b{best_ic}_lr{lr}')
        run_sweep_cell_ce0(mode, rd, dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(lr)))
        acc = best_val_acc_from_csv(rd)
        p2.append((lr, acc))
    p2f = [r for r in p2 if r[1] == r[1]]
    if not p2f:
        raise RuntimeError(f'{mode}: phase 2 produced no valid results')
    best_lr, best_acc = max(p2f, key=lambda r: r[1])
    best = dict(lam_is=float(best_is), lam_ic=float(best_ic), lr=float(best_lr), val_acc=float(best_acc), ce_weight=0.0)
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    with open(bp, 'w') as f:
        json.dump(best, f, indent=2)
    return best


def vrm_loss(student, teacher, eps=1e-8):
    def compare(s, t):
        count = len(s)
        if count < 2:
            return s.sum() * 0.0
        s_sq, t_sq = (s * s).sum(-1), (t * t).sum(-1)
        norm_s = (s_sq[:, None] + s_sq[None, :] - 2.0 * (s @ s.T)).clamp_min(eps)
        norm_t = (t_sq[:, None] + t_sq[None, :] - 2.0 * (t @ t.T)).clamp_min(eps)
        diagonal = (s * t).sum(-1)
        product = s @ t.T
        cross = diagonal[:, None] + diagonal[None, :] - product - product.T
        cosine = (cross / (norm_s * norm_t).sqrt()).clamp(-1.0, 1.0)
        error = 2.0 - 2.0 * cosine
        return (error.sum() - error.diagonal().sum()) / float(count * (count - 1))
    return compare(student, teacher), compare(student.T, teacher.T)


def make_vrm_step(teacher, hp, ce_weight, hint_weight, hint_loss=None):
    lam_is, lam_ic = float(hp['lam_is']), float(hp['lam_ic'])
    max_pairs = int(VRM_FIXED['max_pairs'])
    guided_idx, hint_depth = int(CFG['guided_idx']), int(CFG['hint_depth'])
    def step(model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        if hint_weight > 0:
            student, guided_raw = model.student.forward_with_guided(x, guided_idx)
            guided = model.regressor(guided_raw)
            with torch.no_grad():
                reference, hint = teacher.forward_with_phi_hint(x, hint_depth)
        else:
            student = model(x)
            with torch.no_grad():
                reference = teacher(x)
        if 0 < max_pairs < len(x):
            indices = torch.randint(len(x), (max_pairs,), device=x.device)
            loss_is, loss_ic = vrm_loss(student[indices], reference[indices])
        else:
            loss_is, loss_ic = vrm_loss(student, reference)
        loss = lam_is * loss_is + lam_ic * loss_ic
        if ce_weight > 0:
            loss = loss + ce_weight * F.cross_entropy(student, y)
        if hint_weight > 0:
            hint_value = hint_loss(guided, hint)
            scale = loss.detach() / hint_value.detach().clamp_min(1e-12)
            loss = (1 - hint_weight) * loss + hint_weight * scale * hint_value
        return loss, len(x), {}
    return step


def configure(data_root, output_root, device):
    global JET_DATA_DIR, RAW_DIR, LOG_ROOT, DEVICE
    root = os.path.abspath(data_root)
    JET_DATA_DIR = os.path.join(root, 'processed') if os.path.isdir(os.path.join(root, 'processed')) else root
    RAW_DIR = os.path.join(root, 'raw', 'train')
    LOG_ROOT = os.fspath(output_root)
    DEVICE = torch.device(device)
    os.makedirs(JET_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    _JET_CACHE.clear()
    _FOLD_CACHE.clear()
    torch.manual_seed(SEED)
    random.seed(SEED)


def paper_hparams(mode, ce_weight):
    values = {
        ('perm', 0): dict(lam_is=64.0, lam_ic=16.0, lr=0.003, ce_weight=0.0),
        ('perm', 1): dict(lam_is=16.0, lam_ic=32.0, lr=0.008, ce_weight=1.0),
        ('noperm', 0): dict(lam_is=128.0, lam_ic=16.0, lr=0.05, ce_weight=0.0),
    }
    return values[(mode, ce_weight)].copy()


def evaluate_stage(mode, fold, stage, ce_weight=0.0):
    if stage == 'teacher':
        model = build_teacher().to(DEVICE)
        checkpoint = os.path.join(teacher_dir(mode, fold), 'best.pt')
    elif stage.startswith('vrm_hint'):
        regressor, _, _ = build_regressor(CFG['guided_idx'])
        model = HintStudent(build_student(), regressor, CFG['guided_idx']).to(DEVICE)
        checkpoint = os.path.join(stage_dir(mode, stage, fold), 'best.pt')
    else:
        model = build_student().to(DEVICE)
        checkpoint = os.path.join(stage_dir(mode, stage, fold), 'best.pt')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state'])
    _, loader = get_loaders(mode, fold, CFG['vrm']['batch_size'])
    metrics = validate_full(model, loader, DEVICE)
    metrics.update(dict(top1_agreement=None, teach_stu_jsd=None))
    if stage != 'teacher' and os.path.isfile(os.path.join(teacher_dir(mode, fold), 'best.pt')):
        teacher = load_teacher_for_fold(mode, fold)
        metrics.update(compute_fidelity(model, teacher, loader, DEVICE))
    return dict(regime='transformed' if mode == 'perm' else 'canonical', fold=fold,
                method=stage, alpha=ce_weight, beta=0.25 if 'b025' in stage else 0.0, metrics=metrics)


def run(data_root, output_root, regimes, folds, device, stages, search='run'):
    configure(data_root, output_root, device)
    records = []
    for regime in regimes:
        mode = 'perm' if regime == 'transformed' else 'noperm'
        ce_weights = (0, 1) if mode == 'perm' else (0,)
        teacher_folds = sorted(set(folds) | ({SWEEP_FOLD} if 'search' in stages else set()))
        if 'teacher' in stages or 'search' in stages:
            for fold in teacher_folds:
                run_teacher(mode, fold)
        selected = {}
        for ce_weight in ce_weights:
            path = best_hp_path(mode) if ce_weight else os.path.join(mode_root(mode), 'vrm_sweep_ce0', 'best_hparams_ce0.json')
            if search == 'paper':
                selected[ce_weight] = paper_hparams(mode, ce_weight)
            elif 'search' in stages and search == 'run':
                selected[ce_weight] = vrm_sweep(mode) if ce_weight else vrm_sweep_ce0(mode)
            elif os.path.isfile(path):
                with open(path) as stream:
                    selected[ce_weight] = json.load(stream)
            elif 'distill' in stages:
                raise FileNotFoundError(path)
        for fold in folds:
            if 'baseline' in stages:
                run_baseline(mode, fold)
            if 'hint' in stages:
                run_hint(mode, fold)
            if 'distill' in stages:
                for ce_weight in ce_weights:
                    hp = selected[ce_weight]
                    suffix = '_ce0' if ce_weight == 0 else ''
                    run_vrm_stage(mode, fold, 'vrm' + suffix, hp, ce_weight, 0.0, False)
                    if 'hint' in stages or os.path.isfile(os.path.join(stage_dir(mode, 'hint', fold), 'best.pt')):
                        run_vrm_stage(mode, fold, 'vrm_hint' + suffix, hp, ce_weight, 0.0, True)
                        run_vrm_stage(mode, fold, 'vrm_hint_b025' + suffix, hp, ce_weight, 0.25, True)
            if 'evaluate' in stages:
                for stage in ('teacher', 'baseline'):
                    path = os.path.join(teacher_dir(mode, fold) if stage == 'teacher' else stage_dir(mode, stage, fold), 'best.pt')
                    if os.path.isfile(path):
                        records.append(evaluate_stage(mode, fold, stage))
                for ce_weight in ce_weights:
                    suffix = '_ce0' if ce_weight == 0 else ''
                    for name in ('vrm', 'vrm_hint', 'vrm_hint_b025'):
                        stage = name + suffix
                        if os.path.isfile(os.path.join(stage_dir(mode, stage, fold), 'best.pt')):
                            records.append(evaluate_stage(mode, fold, stage, ce_weight))
                save_results(records, LOG_ROOT)
            _FOLD_CACHE.pop((mode, fold), None)
    return records
