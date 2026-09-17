import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .aligned_data import aligned_paths, verify_aligned_dataset
from .data import cv_indices
from .lorentz_shuffle import verify_shuffled_dataset
from .models import HintModel, LorentzNet, MLPStudent
from .training import TensorBatchLoader, _cached_tensor, _scheduler, atomic_json, kd_loss

def strict_fp32() -> None:
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def _dataset_manifest(data_dir: Path) -> dict:
    if (data_dir / 'manifest.json').is_file() and (data_dir / 'features_pt_deta_dphi.npy').is_file():
        return verify_shuffled_dataset(data_dir)
    return verify_aligned_dataset(data_dir)

def _paths(data_dir: Path) -> dict[str, Path | None]:
    if (data_dir / 'features_pt_deta_dphi.npy').is_file():
        return {'features': data_dir / 'features_pt_deta_dphi.npy', 'fourmom': data_dir / 'fourmom.npy', 'jet_fourmom': data_dir / 'jet_fourmom.npy', 'matrices': data_dir / 'lorentz_matrices.npy', 'labels': data_dir / 'labels.npy', 'sample_ids': data_dir / 'sample_ids.npy'}
    paths = aligned_paths(data_dir)
    return {'features': paths['features'], 'fourmom': paths['fourmom'], 'jet_fourmom': paths['jet_fourmom'], 'matrices': None, 'labels': paths['labels'], 'sample_ids': None}

def _indices(samples: int, task: dict) -> tuple[np.ndarray, np.ndarray]:
    train, validation = cv_indices(samples, int(task.get('fold', 0)), n_folds=int(task.get('n_folds', 5)), fold_seed=int(task.get('fold_seed', 0)))
    train_limit = task.get('train_samples')
    val_limit = task.get('val_samples')
    if train_limit:
        train = train[:min(len(train), int(train_limit))]
    if val_limit:
        validation = validation[:min(len(validation), int(val_limit))]
    return (train, validation)

def _task_complete(output_dir: Path, task: dict) -> dict | None:
    result_path = output_dir / 'result.json'
    config_path = output_dir / 'config.json'
    if not result_path.is_file() or not config_path.is_file() or (not (output_dir / 'best.pt').is_file()):
        return None
    saved_task = json.loads(config_path.read_text(encoding='utf-8'))
    if saved_task != task:
        return None
    result = json.loads(result_path.read_text(encoding='utf-8'))
    return result if result.get('status') == 'complete' else None

def _save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def _l1(parameters, coefficient: float, reference: torch.Tensor) -> torch.Tensor:
    if coefficient <= 0:
        return reference.new_zeros(())
    return coefficient * sum((parameter.abs().sum() for parameter in parameters))

def _forward_supervised(model: nn.Module, kind: str, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if kind == 'teacher':
        fourmom, matrices, _ = batch
        return model(fourmom, lorentz_matrix=matrices)
    features, _ = batch
    return model(features)

@torch.no_grad()
def _validate_supervised(model: nn.Module, kind: str, arrays: tuple[torch.Tensor, ...], indices: np.ndarray, batch_size: int) -> dict:
    model.eval()
    loader = TensorBatchLoader(arrays, indices, batch_size, False, 0)
    count = correct = 0
    nll_sum = 0.0
    for batch in loader:
        labels = batch[-1]
        logits = _forward_supervised(model, kind, batch)
        nll_sum += float(F.cross_entropy(logits, labels, reduction='sum'))
        correct += int(logits.argmax(dim=-1).eq(labels).sum())
        count += labels.numel()
    return {'val_accuracy': correct / max(1, count), 'val_nll': nll_sum / max(1, count)}

@torch.no_grad()
def _calibrate_teacher(model: LorentzNet, arrays: tuple[torch.Tensor, ...], indices: np.ndarray, batch_size: int, max_samples: int) -> None:
    modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]
    if not modules or max_samples <= 0:
        return
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    calibration = indices[:min(len(indices), max_samples)]
    for fourmom, matrices, _ in TensorBatchLoader(arrays, calibration, batch_size, False, 0):
        model(fourmom, lorentz_matrix=matrices)
    model.eval()

def train_supervised(task: dict, data_dir: Path, output_dir: Path, device: torch.device) -> dict:
    prior = _task_complete(output_dir, task)
    if prior is not None:
        return prior
    strict_fp32()
    manifest = _dataset_manifest(data_dir)
    samples = int(manifest['samples'])
    train_indices, val_indices = _indices(samples, task)
    paths = _paths(data_dir)
    kind = task['kind']
    torch.manual_seed(int(task.get('seed', 0)))
    labels = _cached_tensor(paths['labels'], torch.long, device)
    if kind == 'teacher':
        arrays = (_cached_tensor(paths['fourmom'], torch.float32, device), _cached_tensor(paths['matrices'], torch.float32, device), labels)
        model = LorentzNet(hidden_dim=int(task['hidden_dim']), n_layers=int(task['layers']), c_weight=float(task['c_weight']), dropout=float(task['dropout']), add_beams=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(task['lr']), weight_decay=float(task['weight_decay']))
        scheduler = _scheduler(optimizer, int(task['warmup_epochs']), int(task['max_epochs']))
    elif kind == 'baseline':
        arrays = (_cached_tensor(paths['features'], torch.float32, device), labels)
        model = MLPStudent(task['student_hidden']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(task['lr']), weight_decay=float(task['weight_decay']))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=str(task.get('lr_scheduler_mode', 'max')), factor=float(task.get('lr_factor', 0.1)), patience=int(task.get('lr_patience', 15)))
    else:
        raise ValueError(f'Unknown supervised kind: {kind}')
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'config.json', task)
    start_epoch = 0
    best_accuracy = -1.0
    best_nll = float('inf')
    best_epoch = -1
    stale = 0
    last_path = output_dir / 'last.pt'
    if last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get('task') == task:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            start_epoch = int(saved['epoch'])
            best_accuracy = float(saved['best_accuracy'])
            best_nll = float(saved['best_nll'])
            best_epoch = int(saved['best_epoch'])
            stale = int(saved['stale'])
    metrics_path = output_dir / 'metrics.jsonl'
    mode = 'a' if start_epoch > 0 and metrics_path.is_file() else 'w'
    epochs_completed = start_epoch
    with metrics_path.open(mode, encoding='utf-8') as metrics_file:
        for epoch in range(start_epoch, int(task['max_epochs'])):
            model.train()
            loader = TensorBatchLoader(arrays, train_indices, int(task['batch_size']), True, int(task.get('seed', 0)) + epoch)
            train_loss_sum = 0.0
            train_count = 0
            started = time.time()
            for batch in loader:
                labels_batch = batch[-1]
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_supervised(model, kind, batch)
                loss = F.cross_entropy(logits, labels_batch, label_smoothing=float(task.get('label_smoothing', 0.0)))
                loss = loss + _l1(model.parameters(), float(task.get('l1', 0.0)), loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(task.get('grad_clip', 1.0)))
                optimizer.step()
                train_loss_sum += float(loss.detach()) * labels_batch.numel()
                train_count += labels_batch.numel()
            if kind == 'teacher':
                _calibrate_teacher(model, arrays, train_indices, int(task['batch_size']), int(task.get('bn_calibration_samples', 8192)))
            validation = _validate_supervised(model, kind, arrays, val_indices, int(task.get('eval_batch_size', task['batch_size'])))
            improved = validation['val_accuracy'] > best_accuracy or (validation['val_accuracy'] == best_accuracy and validation['val_nll'] < best_nll)
            if improved:
                best_accuracy = validation['val_accuracy']
                best_nll = validation['val_nll']
                best_epoch = epoch + 1
                stale = 0
                _save_checkpoint(output_dir / 'best.pt', {'model': model.state_dict(), 'task': task, 'epoch': best_epoch})
            else:
                stale += 1
            if kind == 'baseline':
                monitor = str(task.get('lr_scheduler_monitor', 'val_accuracy'))
                scheduler.step(validation[monitor])
            else:
                scheduler.step()
            epochs_completed = epoch + 1
            record = {'epoch': epochs_completed, 'train_loss': train_loss_sum / max(1, train_count), **validation, 'best_val_accuracy': best_accuracy, 'lr': optimizer.param_groups[0]['lr'], 'stale': stale, 'seconds': time.time() - started}
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            print(f"{task['stage']}/{task['name']} fold={task.get('fold', 0)} epoch={epochs_completed} loss={record['train_loss']:.5f} val_acc={validation['val_accuracy']:.5f} val_nll={validation['val_nll']:.5f}", flush=True)
            _save_checkpoint(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'task': task, 'epoch': epochs_completed, 'best_accuracy': best_accuracy, 'best_nll': best_nll, 'best_epoch': best_epoch, 'stale': stale})
            if stale >= int(task['patience']):
                break
    result = {**task, 'status': 'complete', 'train_samples_used': len(train_indices), 'val_samples_used': len(val_indices), 'epochs_completed': epochs_completed, 'best_epoch': best_epoch, 'best_val_accuracy': best_accuracy, 'best_val_nll': best_nll, 'output_dir': str(output_dir)}
    atomic_json(output_dir / 'result.json', result)
    return result

def teacher_checkpoint_config(checkpoint: Path) -> dict:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    task = saved.get('task', saved.get('config'))
    if not isinstance(task, dict):
        raise ValueError(f'Teacher checkpoint has no task/config record: {checkpoint}')
    return task

def load_teacher_checkpoint(checkpoint: Path, device: torch.device) -> LorentzNet:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    task = saved.get('task', saved.get('config'))
    if not isinstance(task, dict):
        raise ValueError(f'Teacher checkpoint has no task/config record: {checkpoint}')
    model = LorentzNet(hidden_dim=int(task['hidden_dim']), n_layers=int(task.get('layers', task.get('n_layers'))), c_weight=float(task['c_weight']), dropout=float(task['dropout']), add_beams=not bool(task.get('no_beams', False)), normalization=str(task.get('normalization', 'batch')))
    model.load_state_dict(saved['model'])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

@torch.no_grad()
def cache_aligned_teacher(data_dir: Path, checkpoint: Path, output_dir: Path, device: torch.device, batch_size: int) -> dict:
    manifest = _dataset_manifest(data_dir)
    samples = int(manifest['samples'])
    paths = _paths(data_dir)
    expected = {'data_dir': str(data_dir.resolve()), 'data_modified': [path.stat().st_mtime_ns for path in paths.values() if path is not None], 'schema_version': 1, 'samples': samples, 'teacher_checkpoint': str(checkpoint.resolve()), 'checkpoint_mtime': checkpoint.stat().st_mtime_ns, 'precision': 'float32_no_tf32', 'row_contract': 'logits[i] and hints[i] are produced from the same sample_ids[i] row'}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'
    logits_path = output_dir / 'logits.npy'
    hints_path = output_dir / 'hints.npy'
    ids_path = output_dir / 'sample_ids.npy'
    if manifest_path.is_file() and logits_path.is_file() and hints_path.is_file() and ids_path.is_file():
        current = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((current.get(key) == value for key, value in expected.items())):
            return current
    strict_fp32()
    paths = _paths(data_dir)
    fourmom = _cached_tensor(paths['fourmom'], torch.float32, device)
    matrices = _cached_tensor(paths['matrices'], torch.float32, device) if paths['matrices'] is not None else None
    model = load_teacher_checkpoint(checkpoint, device)
    hint_dim = model.hidden_dim
    logits_out = np.lib.format.open_memmap(logits_path, mode='w+', dtype=np.float32, shape=(samples, 5))
    hints_out = np.lib.format.open_memmap(hints_path, mode='w+', dtype=np.float32, shape=(samples, hint_dim))
    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        matrix_batch = matrices[start:end] if matrices is not None else None
        logits, hints = model.forward_with_hint(fourmom[start:end], lorentz_matrix=matrix_batch)
        logits_out[start:end] = logits.float().cpu().numpy()
        hints_out[start:end] = hints.float().cpu().numpy()
    logits_out.flush()
    hints_out.flush()
    if paths['sample_ids'] is not None:
        shutil.copyfile(paths['sample_ids'], ids_path)
    else:
        np.save(ids_path, np.arange(samples, dtype=np.int64))
    completed = {**expected, 'hint_dim': hint_dim}
    atomic_json(manifest_path, completed)
    return completed

def verify_teacher_cache(data_dir, cache_dir):
    return json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))

def relation_losses(student_logits: torch.Tensor, teacher_logits: torch.Tensor, criterion: str, max_pairs: int=256) -> tuple[torch.Tensor, torch.Tensor]:
    if 0 < max_pairs < len(student_logits):
        selected = torch.randperm(len(student_logits), device=student_logits.device)[:max_pairs]
        student_logits = student_logits[selected]
        teacher_logits = teacher_logits[selected]

    def discrepancy(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        ds = student[:, None, :] - student[None, :, :]
        dt = teacher[:, None, :] - teacher[None, :, :]
        nodes = ds.shape[0]
        if nodes < 2:
            return student.sum() * 0.0
        mask = ~torch.eye(nodes, dtype=torch.bool, device=ds.device)
        ds = ds[mask]
        dt = dt[mask]
        if criterion == 'mse':
            scale = dt.square().mean().sqrt().detach().clamp_min(1e-08)
            return F.mse_loss(ds / scale, dt / scale)
        if criterion == 'smooth_l1':
            scale = dt.abs().mean().detach().clamp_min(1e-08)
            return F.smooth_l1_loss(ds / scale, dt / scale)
        if criterion == 'cosine':
            return (1.0 - F.cosine_similarity(ds, dt, dim=-1, eps=1e-08)).mean()
        if criterion == 'distance_mse':
            student_distance = ds.norm(dim=-1)
            teacher_distance = dt.norm(dim=-1)
            scale = teacher_distance.mean().detach().clamp_min(1e-08)
            return F.mse_loss(student_distance / scale, teacher_distance / scale)
        raise ValueError(f'Unknown relation criterion: {criterion}')
    inter_sample = discrepancy(student_logits, teacher_logits)
    inter_class = discrepancy(student_logits.transpose(0, 1), teacher_logits.transpose(0, 1))
    return (inter_sample, inter_class)

@torch.no_grad()
def _validate_student(model: MLPStudent, arrays: tuple[torch.Tensor, ...], indices: np.ndarray, batch_size: int) -> dict:
    model.eval()
    count = correct = agreement = 0
    nll_sum = jsd_sum = 0.0
    for features, teacher_logits, _, labels in TensorBatchLoader(arrays, indices, batch_size, False, 0):
        logits = model(features)
        probability = F.softmax(logits, dim=-1)
        teacher_probability = F.softmax(teacher_logits, dim=-1)
        middle = 0.5 * (probability + teacher_probability)
        tiny = torch.finfo(probability.dtype).tiny
        jsd = 0.5 * (torch.xlogy(probability, probability / middle.clamp_min(tiny)).sum(dim=-1) + torch.xlogy(teacher_probability, teacher_probability / middle.clamp_min(tiny)).sum(dim=-1)) / math.log(2.0)
        nll_sum += float(F.cross_entropy(logits, labels, reduction='sum'))
        jsd_sum += float(jsd.sum())
        prediction = logits.argmax(dim=-1)
        correct += int(prediction.eq(labels).sum())
        agreement += int(prediction.eq(teacher_logits.argmax(dim=-1)).sum())
        count += labels.numel()
    return {'val_accuracy': correct / max(1, count), 'val_nll': nll_sum / max(1, count), 'val_teacher_top1_agreement': agreement / max(1, count), 'val_teacher_jsd': jsd_sum / max(1, count)}

def train_hint_pretrain(task: dict, data_dir: Path, cache_dir: Path, output_dir: Path, device: torch.device) -> dict:
    prior = _task_complete(output_dir, task)
    if prior is not None:
        return prior
    cache = verify_teacher_cache(data_dir, cache_dir)
    samples = int(cache['samples'])
    train_indices, val_indices = _indices(samples, task)
    paths = _paths(data_dir)
    arrays = (_cached_tensor(paths['features'], torch.float32, device), _cached_tensor(cache_dir / 'hints.npy', torch.float32, device), _cached_tensor(paths['labels'], torch.long, device))
    torch.manual_seed(int(task.get('seed', 0)))
    model = HintModel(MLPStudent(task['student_hidden']), int(task['guided_hidden_index']), int(cache['hint_dim'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(task['lr']), weight_decay=float(task['weight_decay']))
    if task.get('scheduler') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=str(task.get('lr_scheduler_mode', 'min')), factor=float(task.get('lr_factor', 0.1)), patience=int(task.get('lr_patience', 30)))
    else:
        scheduler = _scheduler(optimizer, int(task['warmup_epochs']), int(task['max_epochs']))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'config.json', task)
    best = float('inf')
    best_epoch = -1
    stale = 0
    start_epoch = 0
    last_path = output_dir / 'last.pt'
    if last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get('task') == task:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            start_epoch = int(saved['epoch'])
            best = float(saved['best'])
            best_epoch = int(saved['best_epoch'])
            stale = int(saved['stale'])
    metrics_path = output_dir / 'metrics.jsonl'
    mode = 'a' if start_epoch > 0 and metrics_path.is_file() else 'w'
    epochs_completed = start_epoch
    with metrics_path.open(mode, encoding='utf-8') as metrics_file:
        for epoch in range(start_epoch, int(task['max_epochs'])):
            model.train()
            total = count = 0
            for features, hints, _ in TensorBatchLoader(arrays, train_indices, int(task['batch_size']), True, int(task.get('seed', 0)) + epoch):
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model.guided(features), hints)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * hints.numel()
                count += hints.numel()
            model.eval()
            val_total = val_count = 0
            with torch.no_grad():
                for features, hints, _ in TensorBatchLoader(arrays, val_indices, int(task['batch_size']), False, 0):
                    val_total += float(F.mse_loss(model.guided(features), hints, reduction='sum'))
                    val_count += hints.numel()
            val_mse = val_total / max(1, val_count)
            if val_mse < best:
                best = val_mse
                best_epoch = epoch + 1
                stale = 0
                _save_checkpoint(output_dir / 'best.pt', {'model': model.state_dict(), 'student': model.student.state_dict(), 'task': task, 'epoch': best_epoch})
            else:
                stale += 1
            if task.get('scheduler') == 'plateau':
                scheduler.step(val_mse)
            else:
                scheduler.step()
            record = {'epoch': epoch + 1, 'train_hint_mse': total / max(1, count), 'val_hint_mse': val_mse, 'best_val_hint_mse': best, 'lr': optimizer.param_groups[0]['lr'], 'stale': stale}
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            print(f"{task['stage']}/{task['name']} fold={task.get('fold', 0)} epoch={epoch + 1} loss={record['train_hint_mse']:.5f} val_mse={val_mse:.5f}", flush=True)
            epochs_completed = epoch + 1
            _save_checkpoint(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'task': task, 'epoch': epochs_completed, 'best': best, 'best_epoch': best_epoch, 'stale': stale})
            if stale >= int(task['patience']):
                break
    result = {**task, 'status': 'complete', 'best_epoch': best_epoch, 'best_val_hint_mse': best, 'epochs_completed': epochs_completed, 'output_dir': str(output_dir)}
    atomic_json(output_dir / 'result.json', result)
    return result

def train_distillation(task: dict, data_dir: Path, cache_dir: Path, output_dir: Path, device: torch.device, hint_checkpoint: Path | None=None) -> dict:
    prior = _task_complete(output_dir, task)
    if prior is not None:
        return prior
    cache = verify_teacher_cache(data_dir, cache_dir)
    samples = int(cache['samples'])
    train_indices, val_indices = _indices(samples, task)
    paths = _paths(data_dir)
    arrays = (_cached_tensor(paths['features'], torch.float32, device), _cached_tensor(cache_dir / 'logits.npy', torch.float32, device), _cached_tensor(cache_dir / 'hints.npy', torch.float32, device), _cached_tensor(paths['labels'], torch.long, device))
    torch.manual_seed(int(task.get('seed', 0)))
    student = MLPStudent(task['student_hidden']).to(device)
    hint_model = None
    if task['method'] in ('hint', 'hint_beta'):
        if hint_checkpoint is None or not hint_checkpoint.is_file():
            raise FileNotFoundError(f"Hint checkpoint required for {task['method']}: {hint_checkpoint}")
        saved_hint = torch.load(hint_checkpoint, map_location='cpu', weights_only=False)
        if task['method'] == 'hint_beta':
            hint_model = HintModel(student, int(task['guided_hidden_index']), int(cache['hint_dim'])).to(device)
            hint_model.load_state_dict(saved_hint['model'])
            student = hint_model.student
        else:
            student.load_state_dict(saved_hint['student'])
    parameters = list(student.parameters())
    if hint_model is not None:
        parameters += list(hint_model.projector.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(task['lr']), weight_decay=float(task.get('weight_decay', 0.0)))
    if task.get('scheduler') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=str(task.get('lr_scheduler_mode', 'min')), factor=float(task.get('lr_factor', 0.1)), patience=int(task.get('lr_patience', 30)))
    else:
        scheduler = _scheduler(optimizer, int(task['warmup_epochs']), int(task['max_epochs']))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'config.json', task)
    best_accuracy = -1.0
    best_nll = float('inf')
    best_epoch = -1
    stale = 0
    start_epoch = 0
    last_path = output_dir / 'last.pt'
    if last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get('task') == task:
            student.load_state_dict(saved['model'])
            if hint_model is not None and 'hint_model' in saved:
                hint_model.load_state_dict(saved['hint_model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            start_epoch = int(saved['epoch'])
            best_accuracy = float(saved['best_accuracy'])
            best_nll = float(saved['best_nll'])
            best_epoch = int(saved['best_epoch'])
            stale = int(saved['stale'])
    metrics_path = output_dir / 'metrics.jsonl'
    mode = 'a' if start_epoch > 0 and metrics_path.is_file() else 'w'
    epochs_completed = start_epoch
    with metrics_path.open(mode, encoding='utf-8') as metrics_file:
        for epoch in range(start_epoch, int(task['max_epochs'])):
            student.train()
            if hint_model is not None:
                hint_model.train()
            totals = {'loss': 0.0, 'kd': 0.0, 'ce': 0.0, 'hint': 0.0, 'is': 0.0, 'ic': 0.0}
            count = 0
            loader = TensorBatchLoader(arrays, train_indices, int(task['batch_size']), True, int(task.get('seed', 0)) + epoch)
            for features, teacher_logits, teacher_hints, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = student(features)
                zero = logits.new_zeros(())
                kd_value = ce_value = hint_value = is_value = ic_value = zero
                if task['method'] == 'vrm':
                    is_value, ic_value = relation_losses(logits, teacher_logits, task['criterion'], int(task.get('max_pairs', 256)))
                    main = float(task['lambda_is']) * is_value + float(task['lambda_ic']) * ic_value
                else:
                    kd_value = kd_loss(logits, teacher_logits, float(task['temperature']))
                    main = kd_value
                ce_weight = float(task.get('ce_weight', 0.0))
                if ce_weight > 0:
                    ce_value = F.cross_entropy(logits, labels)
                    main = main + ce_weight * ce_value
                loss = main
                beta = float(task.get('beta', 0.0))
                if hint_model is not None and beta > 0:
                    hint_value = F.mse_loss(hint_model.guided(features), teacher_hints)
                    scale = main.detach() / hint_value.detach().clamp_min(1e-12)
                    loss = (1.0 - beta) * main + beta * scale * hint_value
                loss = loss + _l1(parameters, float(task.get('l1', 0.0)), loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(task.get('grad_clip', 1.0)))
                optimizer.step()
                batch = labels.numel()
                totals['loss'] += float(loss.detach()) * batch
                totals['kd'] += float(kd_value.detach()) * batch
                totals['ce'] += float(ce_value.detach()) * batch
                totals['hint'] += float(hint_value.detach()) * batch
                totals['is'] += float(is_value.detach()) * batch
                totals['ic'] += float(ic_value.detach()) * batch
                count += batch
            validation = _validate_student(student, arrays, val_indices, int(task.get('eval_batch_size', task['batch_size'])))
            improved = validation['val_accuracy'] > best_accuracy or (validation['val_accuracy'] == best_accuracy and validation['val_nll'] < best_nll)
            if improved:
                best_accuracy = validation['val_accuracy']
                best_nll = validation['val_nll']
                best_epoch = epoch + 1
                stale = 0
                checkpoint = {'model': student.state_dict(), 'task': task, 'epoch': best_epoch}
                if hint_model is not None:
                    checkpoint['hint_model'] = hint_model.state_dict()
                _save_checkpoint(output_dir / 'best.pt', checkpoint)
            else:
                stale += 1
            if task.get('scheduler') == 'plateau':
                monitor = str(task.get('lr_scheduler_monitor', 'val_nll'))
                scheduler.step(validation[monitor])
            else:
                scheduler.step()
            record = {'epoch': epoch + 1, **{f'train_{key}': value / max(1, count) for key, value in totals.items()}, **validation, 'best_val_accuracy': best_accuracy, 'lr': optimizer.param_groups[0]['lr'], 'stale': stale}
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            print(f"{task['stage']}/{task['name']} fold={task.get('fold', 0)} epoch={epoch + 1} loss={record['train_loss']:.5f} val_acc={validation['val_accuracy']:.5f} val_nll={validation['val_nll']:.5f}", flush=True)
            epochs_completed = epoch + 1
            checkpoint = {'model': student.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'task': task, 'epoch': epochs_completed, 'best_accuracy': best_accuracy, 'best_nll': best_nll, 'best_epoch': best_epoch, 'stale': stale}
            if hint_model is not None:
                checkpoint['hint_model'] = hint_model.state_dict()
            _save_checkpoint(last_path, checkpoint)
            if stale >= int(task['patience']):
                break
    result = {**task, 'status': 'complete', 'train_samples_used': len(train_indices), 'val_samples_used': len(val_indices), 'best_epoch': best_epoch, 'epochs_completed': epochs_completed, 'best_val_accuracy': best_accuracy, 'best_val_nll': best_nll, 'output_dir': str(output_dir)}
    final_validation = _validate_student(student, arrays, val_indices, int(task.get('eval_batch_size', task['batch_size'])))
    result.update({f'last_{key}': value for key, value in final_validation.items()})
    atomic_json(output_dir / 'result.json', result)
    return result

def select_best(results: list[dict], output: Path, selection_name: str) -> dict:
    valid = [result for result in results if result.get('status') == 'complete']
    if not valid:
        raise RuntimeError(f'No completed trials for {selection_name}')
    best = sorted(valid, key=lambda value: (-float(value.get('best_val_accuracy', -1.0)), float(value.get('best_val_nll', float('inf'))), str(value.get('name', value.get('output_dir', '')))))[0]
    selection = {'selection': selection_name, 'metric': 'best validation accuracy; validation NLL tie-break', 'best': best, 'trials': valid}
    atomic_json(output, selection)
    return selection

def write_results_csv(results: list[dict], path: Path) -> None:
    rows = []
    keys = ('fold', 'stage', 'method', 'ce_mode', 'criterion', 'name', 'temperature', 'ce_weight', 'beta', 'lambda_is', 'lambda_ic', 'lr', 'batch_size', 'weight_decay', 'label_smoothing', 'training_profile', 'best_val_accuracy', 'best_val_nll', 'best_epoch', 'epochs_completed', 'train_samples_used', 'val_samples_used', 'output_dir')
    for result in results:
        rows.append({key: result.get(key) for key in keys})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(keys))
        writer.writeheader()
        writer.writerows(rows)
