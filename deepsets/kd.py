import csv
import json
from functools import partial
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .jetid_datamodule import JetIDDataModule
from .lightning_modules import DeepSetsModule, FitNetsModule, MLPModule
from .metrics import compute_fidelity, save_results, validate_full
from .models import DeepSetsInvariant, MLPBasic


PROFILES = {
    'canonical': {
        'teacher': dict(epochs=200, lr=0.0032, batch_size=128, patience=20, lr_patience=20),
        'baseline': dict(epochs=200, lr=0.01, batch_size=1024, patience=20, lr_patience=20),
        'hint': dict(epochs=200, lr=0.001, batch_size=1024, patience=20, lr_patience=20),
        'kd': dict(epochs=200, lr=0.01, batch_size=1024, patience=20, lr_patience=20),
    },
    'transformed': {
        'teacher': dict(epochs=200, lr=0.0032, batch_size=128, patience=20, lr_patience=20),
        'baseline': dict(epochs=200, lr=0.01, batch_size=1024, patience=20, lr_patience=20),
        'hint': dict(epochs=200, lr=0.001, batch_size=1024, patience=20, lr_patience=20),
        'kd': dict(epochs=200, lr=0.01, batch_size=1024, patience=20, lr_patience=20),
    },
}
SEARCH_BASELINE_LRS = (0.01, 0.05, 0.1)
SEARCH_TEMPERATURES = (4, 8, 16)
SEARCH_ALPHAS = (0.5, 0.75, 1.0, 2.0, 3.0)
SEARCH_BETAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
SEARCH_PROFILES = {
    'teacher': dict(epochs=300, lr=0.0032, batch_size=128, patience=20, lr_patience=15),
    'hint': dict(epochs=200, lr=0.001, batch_size=1024, patience=30, lr_patience=50),
    'baseline': dict(epochs=200, lr=0.01, batch_size=1024, patience=30, lr_patience=20),
    'kd': dict(epochs=300, lr=0.01, batch_size=1024, patience=60, lr_patience=50),
    'beta': dict(epochs=300, lr=0.001, batch_size=1024, patience=30, lr_patience=50),
}


def build_teacher():
    return DeepSetsInvariant(input_dim=3, phi_layers=[32, 32, 32], rho_layers=[32],
                             activ='relu', aggreg='mean', dropout=0.0, output_dim=5)


def build_student():
    return MLPBasic(input_dim=96, layers=[64, 32, 32, 32], output_dim=5, activ='relu')


def data_module(data_root, regime, fold, batch_size, n_folds=5, device=None):
    return JetIDDataModule(data_dir=str(data_root), nconst=32, feats='ptetaphi', norm='robust_fast',
                           batch_size=batch_size, n_folds=n_folds, fold_idx=fold, fold_seed=0,
                           permute_constituents=regime == 'transformed', permute_seed=0, device=device)


def checkpoint(directory, best=True):
    directory = Path(directory) / 'checkpoints'
    if best:
        paths = sorted(directory.glob('epoch_*.ckpt'))
        return paths[0] if paths else directory / 'best.ckpt'
    return directory / 'last.ckpt'


def fit_stage(stage, directory, data_root, regime, fold, device, config, teacher_ckpt=None,
              hint_ckpt=None, temperature=1, alpha=0.0, beta=0.0, seed=325, n_folds=5,
              guided_idx=None, monitor=None):
    directory = Path(directory)
    last = checkpoint(directory, best=False)
    done = directory / 'checkpoints' / 'done.txt'
    if done.exists() and checkpoint(directory).is_file():
        return checkpoint(directory)
    seed_everything(seed, workers=True, verbose=False)
    optimizer = partial(torch.optim.Adam, lr=config['lr'], weight_decay=0.0)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1,
                        patience=config['lr_patience'])
    if stage == 'teacher':
        module = DeepSetsModule(build_teacher(), optimizer, scheduler, compile=False)
    elif stage == 'baseline':
        module = MLPModule(build_student(), optimizer, scheduler, compile=False, l1_lambda=0.0)
    else:
        guided_idx = 3 if guided_idx is None else guided_idx
        module = FitNetsModule(build_teacher(), build_student(), optimizer, scheduler,
                               teacher_ckpt=str(teacher_ckpt), stage='hint' if stage == 'hint' else 'kd',
                               student_ckpt=str(hint_ckpt) if hint_ckpt is not None else None,
                               temperature=temperature, alpha=alpha, beta=beta, guided_idx=guided_idx,
                               hint_depth=3, freeze_guided=False, compile=False)
    monitor = monitor or ('val/loss' if stage == 'hint' else 'val/acc')
    mode = 'min' if monitor == 'val/loss' else 'max'
    model_checkpoint = ModelCheckpoint(dirpath=directory / 'checkpoints', filename='epoch_{epoch:03d}',
                                       auto_insert_metric_name=False, monitor=monitor, mode=mode,
                                       save_last=False, save_top_k=1, enable_version_counter=False)
    last_checkpoint = ModelCheckpoint(dirpath=directory / 'checkpoints', filename='last',
                                      monitor=None, save_top_k=-1, save_last=False,
                                      enable_version_counter=False, every_n_epochs=1, save_on_train_epoch_end=True)
    early_stopping = EarlyStopping(monitor=monitor, mode=mode, patience=config['patience'])
    device = torch.device(device)
    accelerator = 'gpu' if device.type == 'cuda' else device.type
    devices = [device.index or 0] if device.type == 'cuda' else 1
    trainer = Trainer(max_epochs=config['epochs'], accelerator=accelerator, devices=devices,
                      callbacks=[model_checkpoint, last_checkpoint, early_stopping], default_root_dir=str(directory),
                      logger=CSVLogger(str(directory), name='csv', version=0),
                      enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=50)
    dm = data_module(data_root, regime, fold, config['batch_size'], n_folds)
    trainer.fit(module, datamodule=dm, ckpt_path=str(last) if last.exists() else None, weights_only=False)
    done.write_text('completed', encoding='utf-8')
    return Path(model_checkpoint.best_model_path)


def load_model(path, stage):
    model = build_teacher() if stage == 'teacher' else build_student()
    prefix = 'net.' if stage in ('teacher', 'baseline') else 'student.'
    state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict({name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)})
    return model


def evaluate_stage(directory, teacher_directory, stage, data_root, regime, fold, device,
                   temperature=None, alpha=0.0, beta=0.0, n_folds=5):
    model = load_model(checkpoint(directory), stage)
    dm = data_module(data_root, regime, fold, 1024, n_folds, device)
    dm.prepare_data()
    dm.setup('test')
    loader = dm.val_dataloader()
    metrics = validate_full(model, loader, device)
    metrics.update(top1_agreement=None, teach_stu_jsd=None)
    if stage != 'teacher' and checkpoint(teacher_directory).is_file():
        teacher = load_model(checkpoint(teacher_directory), 'teacher')
        metrics.update(compute_fidelity(model, teacher, loader, device))
    return dict(regime=regime, fold=fold, method=stage, temperature=temperature,
                alpha=alpha, beta=beta, metrics=metrics)


def best_val_accuracy(directory):
    values = []
    path = Path(directory) / 'csv' / 'version_0' / 'metrics.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row.get('val/acc'):
                values.append(float(row['val/acc']))
    return max(values)


def run_search(data_root, output_root, regime, device):
    root = Path(output_root) / regime / 'search'
    teacher_dir, hint_dir = root / 'teacher', root / 'hint'
    teacher_ckpt = fit_stage('teacher', teacher_dir, data_root, regime, 0, device,
                            SEARCH_PROFILES['teacher'], seed=42, n_folds=1)
    hint_ckpt = fit_stage('hint', hint_dir, data_root, regime, 0, device,
                         SEARCH_PROFILES['hint'], teacher_ckpt=teacher_ckpt, seed=42,
                         n_folds=1, guided_idx=3)
    results = []
    for lr in SEARCH_BASELINE_LRS:
        config = dict(SEARCH_PROFILES['baseline'], lr=lr)
        directory = root / 'baseline' / f'lr_{lr:g}'
        fit_stage('baseline', directory, data_root, regime, 0, device, config, seed=42, n_folds=1)
        item = evaluate_stage(directory, teacher_dir, 'baseline', data_root, regime, 0, device, n_folds=1)
        item.update(lr=lr, val_acc_best=best_val_accuracy(directory))
        results.append(item)
    for temperature in SEARCH_TEMPERATURES:
        for alpha in SEARCH_ALPHAS:
            directory = root / 'kd' / f'T_{temperature:g}_alpha_{alpha:g}'
            fit_stage('kd', directory, data_root, regime, 0, device, SEARCH_PROFILES['kd'],
                      teacher_ckpt=teacher_ckpt, temperature=temperature, alpha=alpha, seed=42,
                      n_folds=1, monitor='val/loss')
            item = evaluate_stage(directory, teacher_dir, 'kd', data_root, regime, 0, device,
                                  temperature=temperature, alpha=alpha, n_folds=1)
            item.update(lr=SEARCH_PROFILES['kd']['lr'], val_acc_best=best_val_accuracy(directory))
            results.append(item)
    for beta in SEARCH_BETAS:
        directory = root / 'beta' / f'beta_{beta:g}'
        fit_stage('kd_hint_beta', directory, data_root, regime, 0, device, SEARCH_PROFILES['beta'],
                  teacher_ckpt=teacher_ckpt, hint_ckpt=hint_ckpt, temperature=32, beta=beta,
                  guided_idx=3, seed=42, n_folds=1)
        item = evaluate_stage(directory, teacher_dir, 'kd_hint_beta', data_root, regime, 0, device,
                              temperature=32, beta=beta, n_folds=1)
        item.update(lr=SEARCH_PROFILES['beta']['lr'], val_acc_best=best_val_accuracy(directory))
        results.append(item)
    (root / 'search_results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    best = {method: max((row for row in results if row['method'] == method), key=lambda row: row['val_acc_best'])
            for method in ('baseline', 'kd', 'kd_hint_beta')}
    (root / 'best_hparams.json').write_text(json.dumps(best, indent=2, allow_nan=False), encoding='utf-8')
    return results


def run(data_root, output_root, regimes, folds, device, stages, temperatures=(1, 2, 4, 8, 16)):
    records = []
    for regime in regimes:
        if 'search' in stages:
            run_search(data_root, output_root, regime, device)
        profile = PROFILES[regime]
        alphas = (0.0,) if regime == 'canonical' else (0.0, 1.0)
        for fold in folds:
            root = Path(output_root) / regime / f'fold_{fold}'
            teacher_dir, baseline_dir, hint_dir = root / 'teacher', root / 'baseline', root / 'hint'
            if 'teacher' in stages:
                fit_stage('teacher', teacher_dir, data_root, regime, fold, device, profile['teacher'])
            teacher_ckpt = checkpoint(teacher_dir)
            if 'baseline' in stages:
                fit_stage('baseline', baseline_dir, data_root, regime, fold, device, profile['baseline'])
            if 'hint' in stages:
                fit_stage('hint', hint_dir, data_root, regime, fold, device, profile['hint'], teacher_ckpt=teacher_ckpt)
            hint_ckpt = checkpoint(hint_dir)
            candidates = [('teacher', teacher_dir, None, 0.0, 0.0), ('baseline', baseline_dir, None, 0.0, 0.0)]
            for temperature in temperatures:
                for alpha in alphas:
                    for method, warm_start, beta in (('kd', False, 0.0), ('kd_hint', True, 0.0), ('kd_hint_beta', True, 0.25)):
                        directory = root / method / f'T_{temperature:g}_alpha_{alpha:g}'
                        if 'distill' in stages:
                            fit_stage(method, directory, data_root, regime, fold, device, profile['kd'],
                                      teacher_ckpt=teacher_ckpt, hint_ckpt=hint_ckpt if warm_start else None,
                                      temperature=temperature, alpha=alpha, beta=beta)
                        candidates.append((method, directory, temperature, alpha, beta))
            if 'evaluate' in stages:
                for method, directory, temperature, alpha, beta in candidates:
                    if checkpoint(directory).exists():
                        records.append(evaluate_stage(directory, teacher_dir, method, data_root, regime, fold,
                                                      device, temperature, alpha, beta))
                save_results(records, output_root)
    return records
