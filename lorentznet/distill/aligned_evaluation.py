import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .advanced_training import load_teacher_checkpoint, strict_fp32, teacher_checkpoint_config, verify_teacher_cache
from .aligned_data import aligned_paths, verify_aligned_dataset
from .data import cv_indices
from .lorentz_shuffle import REFERENCE_LORENTZ_SOURCE, REFERENCE_LORENTZ_TRANSFORMS, atomic_json, seeded_reference_lorentz_matrices
from .models import MLPStudent
from .training import TensorBatchLoader, _cached_tensor

def _jsd(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (probability + reference)
    safe = middle.clamp_min(torch.finfo(probability.dtype).tiny)
    value = 0.5 * (torch.xlogy(probability, probability / safe).sum(dim=-1) + torch.xlogy(reference, reference / safe).sum(dim=-1)) / math.log(2.0)
    return value.clamp(0.0, 1.0)

def _relative_features(constituents: torch.Tensor, jets: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    valid = constituents[..., 0] != 0.0
    px, py, pz = (constituents[..., 1], constituents[..., 2], constituents[..., 3])
    pt = torch.sqrt(px.square() + py.square())
    eta = torch.asinh(pz / pt.clamp_min(torch.finfo(pt.dtype).eps))
    phi = torch.atan2(py, px)
    jet_pt = torch.sqrt(jets[:, 1].square() + jets[:, 2].square())
    jet_eta = torch.asinh(jets[:, 3] / jet_pt.clamp_min(torch.finfo(jet_pt.dtype).eps))
    jet_phi = torch.atan2(jets[:, 2], jets[:, 1])
    delta_phi = torch.remainder(phi - jet_phi[:, None] + math.pi, 2.0 * math.pi) - math.pi
    features = torch.stack((pt, eta - jet_eta[:, None], delta_phi), dim=-1)
    return features * valid.unsqueeze(-1) / scale.view(1, 1, 3)

def _load_student(checkpoint: Path, device: torch.device) -> tuple[MLPStudent, dict]:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    task = saved['task']
    model = MLPStudent(task['student_hidden'])
    model.load_state_dict(saved['model'])
    model.to(device).eval()
    return (model, task)

def _ece_update(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor, confidence_sums: torch.Tensor, correct_sums: torch.Tensor) -> None:
    probability = F.softmax(logits, dim=-1)
    confidence, prediction = probability.max(dim=-1)
    bins = torch.clamp((confidence * counts.numel()).long(), max=counts.numel() - 1)
    counts.scatter_add_(0, bins, torch.ones_like(confidence, dtype=torch.float64))
    confidence_sums.scatter_add_(0, bins, confidence.to(torch.float64))
    correct_sums.scatter_add_(0, bins, prediction.eq(labels).to(torch.float64))

@torch.no_grad()
def evaluate_best_student(checkpoint: Path, data_dir: Path, cache_dir: Path, output_path: Path, device: torch.device, evaluation: dict, selection: dict) -> dict:
    strict_fp32()
    manifest = verify_aligned_dataset(data_dir)
    cache = verify_teacher_cache(data_dir, cache_dir)
    model, task = _load_student(checkpoint, device)
    transform_count = int(evaluation['lorentz_transforms'])
    identity = {'schema_version': 1, 'checkpoint': str(checkpoint), 'selection': selection, 'evaluation': evaluation}
    samples = int(manifest['samples'])
    _, validation = cv_indices(samples, int(task['fold']), n_folds=int(task['n_folds']), fold_seed=int(task['fold_seed']))
    paths = aligned_paths(data_dir)
    arrays = (_cached_tensor(paths['features'], torch.float32, device), _cached_tensor(paths['fourmom'], torch.float32, device), _cached_tensor(paths['jet_fourmom'], torch.float32, device), _cached_tensor(paths['labels'], torch.long, device), _cached_tensor(cache_dir / 'logits.npy', torch.float32, device))
    transformations = torch.from_numpy(seeded_reference_lorentz_matrices(int(evaluation['transform_sequence']), transform_count)).to(device=device, dtype=torch.float32)
    scale = torch.tensor(manifest['student_scale'], device=device, dtype=torch.float32)
    ece_bins = int(evaluation['ece_bins'])
    bin_counts = torch.zeros(ece_bins, device=device, dtype=torch.float64)
    bin_confidence = torch.zeros_like(bin_counts)
    bin_correct = torch.zeros_like(bin_counts)
    invariance_jsd = torch.zeros(transform_count, device=device, dtype=torch.float64)
    invariance_agreement = torch.zeros_like(invariance_jsd)
    invariance_difference_sq = torch.zeros_like(invariance_jsd)
    invariance_reference_sq = torch.zeros_like(invariance_jsd)
    correct = count = fidelity_agreement = 0
    nll_sum = fidelity_jsd = 0.0
    loader = TensorBatchLoader(arrays, validation, int(evaluation['batch_size']), False, 0)
    for features, fourmom, jets, labels, teacher_logits in loader:
        logits = model(features)
        probability = F.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1)
        teacher_probability = F.softmax(teacher_logits, dim=-1)
        batch = labels.numel()
        count += batch
        correct += int(prediction.eq(labels).sum())
        nll_sum += float(F.cross_entropy(logits, labels, reduction='sum'))
        fidelity_agreement += int(prediction.eq(teacher_logits.argmax(dim=-1)).sum())
        fidelity_jsd += float(_jsd(probability, teacher_probability).sum())
        _ece_update(logits, labels, bin_counts, bin_confidence, bin_correct)
        transformed_fourmom = torch.einsum('tij,bnj->tbni', transformations, fourmom)
        transformed_jets = torch.einsum('tij,bj->tbi', transformations, jets)
        transformed_features = _relative_features(transformed_fourmom.flatten(0, 1), transformed_jets.flatten(0, 1), scale)
        transformed_logits = model(transformed_features).view(transform_count, batch, -1)
        transformed_probability = F.softmax(transformed_logits, dim=-1)
        reference_probability = probability.unsqueeze(0).expand_as(transformed_probability)
        invariance_jsd += _jsd(reference_probability, transformed_probability).sum(dim=1).double()
        invariance_agreement += transformed_logits.argmax(dim=-1).eq(prediction.unsqueeze(0)).sum(dim=1).double()
        invariance_difference_sq += (transformed_logits - logits.unsqueeze(0)).square().sum(dim=(1, 2)).double()
        invariance_reference_sq += logits.square().sum().double()
    nonempty = bin_counts > 0
    ece = (bin_correct[nonempty] / bin_counts[nonempty] - bin_confidence[nonempty] / bin_counts[nonempty]).abs().mul(bin_counts[nonempty] / max(1, count)).sum()
    relative_error = torch.sqrt(invariance_difference_sq / invariance_reference_sq.clamp_min(torch.finfo(torch.float64).tiny))
    metrics = {'accuracy': correct / max(1, count), 'nll': nll_sum / max(1, count), 'ece': float(ece), 'lorentz_invariance_1_jsd': 1.0 - float((invariance_jsd / max(1, count)).mean()), 'lorentz_invariance_top1_agreement': float((invariance_agreement / max(1, count)).mean()), 'lorentz_relative_logit_error': float(relative_error.mean()), 'fid_top1_agreement': fidelity_agreement / max(1, count), 'fid_1_jsd': 1.0 - fidelity_jsd / max(1, count)}
    result = {**identity, 'fold': int(task['fold']), 'method': selection.get('method'), 'ce_mode': selection.get('ce_mode'), 'status': 'complete', 'validation_samples': count, 'metrics': metrics, 'invariance': {'transform_source': REFERENCE_LORENTZ_SOURCE, 'transform_count': transform_count, 'coordinate_adaptation': 'reference (px,py,pz,E) conjugated to stored (E,px,py,pz)', 'decomposition': 'P4 @ B_z(rapidity) @ Q4.T; P=U@U; Q=V@V', 'rapidity_distribution': 'torch.rand in [0,1)', 'scope': 'the same matrix is applied to constituent and jet four-vectors; MLP features are rebuilt'}, 'fid': {'definition': 'student-teacher fidelity on the same held-out event rows', 'top1_agreement': metrics['fid_top1_agreement'], 'one_minus_jsd': metrics['fid_1_jsd']}}
    atomic_json(output_path, result)
    return result

@torch.no_grad()
def evaluate_teacher(checkpoint: Path, data_dir: Path, output_path: Path, device: torch.device, evaluation: dict, fold: int) -> dict:
    strict_fp32()
    manifest = verify_aligned_dataset(data_dir)
    task = teacher_checkpoint_config(checkpoint)
    if int(task.get('fold', -1)) != int(fold):
        raise ValueError(f'Teacher checkpoint {checkpoint} is not fold {fold}')
    model = load_teacher_checkpoint(checkpoint, device)
    transform_count = int(evaluation['lorentz_transforms'])
    identity = {'schema_version': 1, 'checkpoint': str(checkpoint), 'model_kind': 'teacher', 'fold': int(fold), 'evaluation': evaluation}
    samples = int(manifest['samples'])
    _, validation = cv_indices(samples, int(fold), n_folds=int(task.get('n_folds', 5)), fold_seed=int(task.get('fold_seed', 0)))
    paths = aligned_paths(data_dir)
    arrays = (_cached_tensor(paths['fourmom'], torch.float32, device), _cached_tensor(paths['labels'], torch.long, device))
    transformations = torch.from_numpy(seeded_reference_lorentz_matrices(int(evaluation['transform_sequence']), transform_count)).to(device=device, dtype=torch.float32)
    ece_bins = int(evaluation['ece_bins'])
    bin_counts = torch.zeros(ece_bins, device=device, dtype=torch.float64)
    bin_confidence = torch.zeros_like(bin_counts)
    bin_correct = torch.zeros_like(bin_counts)
    invariance_jsd = torch.zeros(transform_count, device=device, dtype=torch.float64)
    invariance_agreement = torch.zeros_like(invariance_jsd)
    invariance_difference_sq = torch.zeros_like(invariance_jsd)
    invariance_reference_sq = torch.zeros_like(invariance_jsd)
    correct = count = 0
    nll_sum = 0.0
    loader = TensorBatchLoader(arrays, validation, int(evaluation['teacher_batch_size']), False, 0)
    for fourmom, labels in loader:
        logits = model(fourmom)
        probability = F.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1)
        count += labels.numel()
        correct += int(prediction.eq(labels).sum())
        nll_sum += float(F.cross_entropy(logits, labels, reduction='sum'))
        _ece_update(logits, labels, bin_counts, bin_confidence, bin_correct)
        for index, matrix in enumerate(transformations):
            transformed_fourmom = fourmom @ matrix.transpose(0, 1)
            transformed_logits = model(transformed_fourmom, lorentz_matrix=matrix)
            transformed_probability = F.softmax(transformed_logits, dim=-1)
            invariance_jsd[index] += _jsd(probability, transformed_probability).sum().double()
            invariance_agreement[index] += transformed_logits.argmax(dim=-1).eq(prediction).sum().double()
            invariance_difference_sq[index] += (transformed_logits - logits).square().sum().double()
            invariance_reference_sq[index] += logits.square().sum().double()
    nonempty = bin_counts > 0
    ece = (bin_correct[nonempty] / bin_counts[nonempty] - bin_confidence[nonempty] / bin_counts[nonempty]).abs().mul(bin_counts[nonempty] / max(1, count)).sum()
    relative_error = torch.sqrt(invariance_difference_sq / invariance_reference_sq.clamp_min(torch.finfo(torch.float64).tiny))
    metrics = {'accuracy': correct / max(1, count), 'nll': nll_sum / max(1, count), 'ece': float(ece), 'lorentz_invariance_1_jsd': 1.0 - float((invariance_jsd / max(1, count)).mean()), 'lorentz_invariance_top1_agreement': float((invariance_agreement / max(1, count)).mean()), 'lorentz_relative_logit_error': float(relative_error.mean()), 'fid_top1_agreement': 1.0, 'fid_1_jsd': 1.0}
    result = {**identity, 'method': 'teacher', 'ce_mode': 'not_applicable', 'status': 'complete', 'validation_samples': count, 'metrics': metrics, 'invariance': {'transform_source': REFERENCE_LORENTZ_SOURCE, 'transform_count': transform_count, 'coordinate_adaptation': 'reference (px,py,pz,E) conjugated to stored (E,px,py,pz)', 'decomposition': 'P4 @ B_z(rapidity) @ Q4.T; P=U@U; Q=V@V', 'rapidity_distribution': 'torch.rand in [0,1)', 'scope': 'the same matrix is applied to constituents and the LorentzNet beam vectors'}, 'fid': {'definition': 'teacher self-reference; reported only for schema compatibility', 'top1_agreement': 1.0, 'one_minus_jsd': 1.0}}
    atomic_json(output_path, result)
    return result
