from __future__ import annotations
from types import SimpleNamespace
import json
import math
import random
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from .data import JetIDNpyDataset, cv_indices, stratified_limit
from .model import LorentzNet

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_datasets(args: SimpleNamespace) -> tuple[JetIDNpyDataset, JetIDNpyDataset]:
    train_x = args.data_dir / 'x_train.npy'
    train_y = args.data_dir / 'y_train.npy'
    labels = np.load(train_y, mmap_mode='r')
    if args.split == 'cv5':
        train_indices, val_indices = cv_indices(len(labels), args.fold, n_folds=5, fold_seed=args.fold_seed)
        train_indices = stratified_limit(train_indices, labels, args.max_train_samples, args.seed)
        val_indices = stratified_limit(val_indices, labels, args.max_val_samples, args.seed + 1)
        return (JetIDNpyDataset(train_x, train_y, train_indices), JetIDNpyDataset(train_x, train_y, val_indices))
    validation_x = args.data_dir / 'x_val.npy'
    validation_y = args.data_dir / 'y_val.npy'
    train_indices = stratified_limit(np.arange(len(labels)), labels, args.max_train_samples, args.seed)
    validation_labels = np.load(validation_y, mmap_mode='r')
    val_indices = stratified_limit(np.arange(len(validation_labels)), validation_labels, args.max_val_samples, args.seed + 1)
    return (JetIDNpyDataset(train_x, train_y, train_indices), JetIDNpyDataset(validation_x, validation_y, val_indices))

def make_loader(dataset: JetIDNpyDataset, batch_size: int, shuffle: bool, workers: int, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=device.type == 'cuda', persistent_workers=workers > 0, generator=generator, drop_last=shuffle and len(dataset) >= batch_size)

def make_bn_calibration_loader(dataset: JetIDNpyDataset, sample_count: int, batch_size: int, workers: int, seed: int, device: torch.device) -> tuple[JetIDNpyDataset | None, DataLoader | None]:
    if sample_count <= 0:
        return (None, None)
    calibration_indices = stratified_limit(dataset.indices, dataset.y, min(sample_count, len(dataset)), seed)
    calibration_dataset = JetIDNpyDataset(dataset.x_path, dataset.y_path, calibration_indices)
    loader = make_loader(calibration_dataset, batch_size=batch_size, shuffle=False, workers=workers, seed=seed, device=device)
    return (calibration_dataset, loader)

def recalibrate_batch_norm(model: nn.Module, loader: DataLoader | None, device: torch.device) -> int:
    if loader is None:
        return 0
    layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm1d)]
    if not layers:
        return 0
    model.eval()
    original_momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    seen = 0
    with torch.no_grad():
        for x, target in loader:
            del target
            x = x.to(device, non_blocking=True)
            model(x)
            seen += len(x)
    for layer, momentum in zip(layers, original_momenta):
        layer.momentum = momentum
    model.eval()
    return seen

def batch_norm_summary(model: nn.Module) -> dict[str, float]:
    layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm1d)]
    if not layers:
        return {}
    means = torch.cat([layer.running_mean.detach().abs().cpu() for layer in layers])
    variances = torch.cat([layer.running_var.detach().cpu() for layer in layers])
    return {'bn_running_mean_abs_max': float(means.max()), 'bn_running_var_min': float(variances.min()), 'bn_running_var_max': float(variances.max())}

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: AdamW | None, grad_clip: float, label_smoothing: float) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, target)
        if training:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        batch = len(target)
        total_loss += float(loss.detach()) * batch
        correct += int((logits.argmax(dim=1) == target).sum())
        seen += batch
    if seen == 0:
        raise RuntimeError('Data loader produced no batches')
    return (total_loss / seen, correct / seen)

def scheduler_lambda(epoch: int, warmup: int, total: int, min_lr_ratio: float) -> float:
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

def train_teacher(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_dataset, val_dataset = build_datasets(args)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, args.seed + 1, device)
    calibration_dataset, calibration_loader = make_bn_calibration_loader(train_dataset, args.bn_calibration_samples, args.batch_size, args.num_workers, args.seed + 2, device)
    model = LorentzNet(hidden_dim=args.hidden_dim, n_layers=args.layers, c_weight=args.c_weight, dropout=args.dropout, add_beams=not args.no_beams, normalization=args.normalization).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler_lambda(epoch, args.warmup_epochs, args.epochs, args.min_lr_ratio))
    config = vars(args).copy()
    config.update({'data_dir': str(args.data_dir.resolve()), 'output_dir': str(args.output_dir.resolve()), 'device': str(device), 'compute_precision': 'float32_no_tf32', 'train_samples': len(train_dataset), 'val_samples': len(val_dataset), 'parameters': sum((p.numel() for p in model.parameters()))})
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    best_accuracy = -1.0
    best_epoch = -1
    best_rolling_val_accuracy = -math.inf
    best_raw_accuracy = -1.0
    best_raw_epoch = -1
    stale_epochs = 0
    target_epochs = 0
    validation_history: list[float] = []
    status = 'max_epochs'
    metrics_path = args.output_dir / 'metrics.jsonl'
    start_time = time.time()
    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer, args.grad_clip, args.label_smoothing)
            calibrated_samples = recalibrate_batch_norm(model, calibration_loader, device)
            val_loss, val_accuracy = run_epoch(model, val_loader, device, optimizer=None, grad_clip=0.0, label_smoothing=0.0)
            validation_history.append(val_accuracy)
            selection_values = validation_history[-args.selection_window:]
            selection_ready = len(selection_values) == args.selection_window
            rolling_val_accuracy = statistics.fmean(selection_values)
            record = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'selection_ready': selection_ready, 'rolling_val_accuracy': rolling_val_accuracy, 'bn_calibration_samples': calibrated_samples, 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - epoch_start, **batch_norm_summary(model)}
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            print(f'{args.output_dir.name} fold={args.fold} epoch={epoch + 1} loss={train_loss:.5f} val_acc={val_accuracy:.5f} val_nll={val_loss:.5f}', flush=True)
            if val_accuracy > best_raw_accuracy:
                best_raw_accuracy = val_accuracy
                best_raw_epoch = epoch + 1
            improved = selection_ready and rolling_val_accuracy > best_rolling_val_accuracy
            if improved:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                best_rolling_val_accuracy = rolling_val_accuracy
                stale_epochs = 0
                torch.save({'model': model.state_dict(), 'config': config, 'epoch': best_epoch, 'val_accuracy': best_accuracy, 'rolling_val_accuracy': best_rolling_val_accuracy, 'selection_window': args.selection_window}, args.output_dir / 'best.pt')
            elif selection_ready:
                stale_epochs += 1
            target_epochs = target_epochs + 1 if val_accuracy >= args.target_accuracy else 0
            scheduler.step()
            if epoch + 1 >= args.min_epochs and target_epochs >= args.target_hold_epochs:
                status = 'target_reached'
                break
            if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
                status = 'early_stopped'
                break
    result = {'status': status, 'best_val_accuracy': best_accuracy, 'best_epoch': best_epoch, 'best_rolling_val_accuracy': best_rolling_val_accuracy, 'best_raw_val_accuracy': best_raw_accuracy, 'best_raw_epoch': best_raw_epoch, 'target_accuracy': args.target_accuracy, 'target_reached': status == 'target_reached', 'best_accuracy_at_or_above_target': best_accuracy >= args.target_accuracy, 'epochs_completed': epoch + 1, 'elapsed_seconds': time.time() - start_time, 'output_dir': str(args.output_dir.resolve()), 'fold': args.fold, 'seed': args.seed}
    torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch + 1}, args.output_dir / 'last.pt')
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if calibration_dataset is not None:
        calibration_dataset.close()
    train_dataset.close()
    val_dataset.close()
    return result
