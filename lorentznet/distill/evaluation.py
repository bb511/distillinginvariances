import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .advanced_training import load_teacher_checkpoint, strict_fp32, verify_teacher_cache
from .data import cv_indices
from .lorentz_shuffle import REFERENCE_LORENTZ_SOURCE, REFERENCE_LORENTZ_TRANSFORMS, atomic_json, seeded_reference_lorentz_matrices, verify_shuffled_dataset
from .models import MLPStudent
from .training import TensorBatchLoader, _cached_tensor

def _jsd(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (probability + reference)
    safe = middle.clamp_min(torch.finfo(probability.dtype).tiny)
    divergence = 0.5 * (torch.xlogy(probability, probability / safe).sum(dim=-1) + torch.xlogy(reference, reference / safe).sum(dim=-1)) / math.log(2.0)
    return divergence.clamp(0.0, 1.0)

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
    ones = torch.ones_like(confidence, dtype=torch.float64)
    counts.scatter_add_(0, bins, ones)
    confidence_sums.scatter_add_(0, bins, confidence.to(torch.float64))
    correct_sums.scatter_add_(0, bins, prediction.eq(labels).to(torch.float64))

@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, model_kind: str, data_dir: Path, cache_dir: Path, output_path: Path, device: torch.device, evaluation: dict, model_metadata: dict) -> dict:
    strict_fp32()
    manifest = verify_shuffled_dataset(data_dir)
    transform_count = int(evaluation['lorentz_transforms'])
    cache = None
    if model_kind == 'student':
        cache = verify_teacher_cache(data_dir, cache_dir)
        model, task = _load_student(checkpoint, device)
    elif model_kind == 'teacher':
        model = load_teacher_checkpoint(checkpoint, device)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        task = saved['task']
    else:
        raise ValueError(f'Unknown model kind: {model_kind}')
    identity = {'schema_version': 2, 'checkpoint': str(checkpoint), 'model_kind': model_kind, 'evaluation': evaluation, 'model': model_metadata}
    samples = int(manifest['samples'])
    _, validation = cv_indices(samples, int(task.get('fold', 0)), n_folds=int(task.get('n_folds', 5)), fold_seed=int(task.get('fold_seed', 0)))
    arrays = (_cached_tensor(data_dir / 'features_pt_deta_dphi.npy', torch.float32, device), _cached_tensor(data_dir / 'fourmom.npy', torch.float32, device), _cached_tensor(data_dir / 'jet_fourmom.npy', torch.float32, device), _cached_tensor(data_dir / 'lorentz_matrices.npy', torch.float32, device), _cached_tensor(data_dir / 'labels.npy', torch.long, device))
    teacher_logits = None
    if cache is not None:
        teacher_logits = _cached_tensor(cache_dir / 'logits.npy', torch.float32, device)
        arrays = (*arrays, teacher_logits)
    transformations = torch.from_numpy(seeded_reference_lorentz_matrices(int(evaluation['seed']), transform_count)).to(device=device, dtype=torch.float32)
    scale = torch.tensor(manifest['student_scale'], device=device, dtype=torch.float32)
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
    fidelity_agreement = 0
    fidelity_jsd = 0.0
    loader = TensorBatchLoader(arrays, validation, int(evaluation['batch_size']), False, 0)
    for batch in loader:
        if model_kind == 'student':
            features, fourmom, jets, base_matrices, labels, cached_teacher_logits = batch
            base_logits = model(features)
        else:
            features, fourmom, jets, base_matrices, labels = batch
            base_logits = model(fourmom, lorentz_matrix=base_matrices)
            cached_teacher_logits = None
        base_probability = F.softmax(base_logits, dim=-1)
        base_prediction = base_logits.argmax(dim=-1)
        batch_count = labels.numel()
        correct += int(base_prediction.eq(labels).sum())
        count += batch_count
        nll_sum += float(F.cross_entropy(base_logits, labels, reduction='sum'))
        _ece_update(base_logits, labels, bin_counts, bin_confidence, bin_correct)
        if cached_teacher_logits is not None:
            teacher_probability = F.softmax(cached_teacher_logits, dim=-1)
            fidelity_agreement += int(base_prediction.eq(cached_teacher_logits.argmax(dim=-1)).sum())
            fidelity_jsd += float(_jsd(base_probability, teacher_probability).sum())
        if model_kind == 'student':
            transformed_fourmom = torch.einsum('tij,bnj->tbni', transformations, fourmom)
            transformed_jets = torch.einsum('tij,bj->tbi', transformations, jets)
            flat_fourmom = transformed_fourmom.flatten(0, 1)
            flat_jets = transformed_jets.flatten(0, 1)
            transformed_features = _relative_features(flat_fourmom, flat_jets, scale)
            transformed_logits = model(transformed_features).view(transform_count, batch_count, -1)
            transformed_probability = F.softmax(transformed_logits, dim=-1)
            base_probability_views = base_probability.unsqueeze(0).expand_as(transformed_probability)
            invariance_jsd += _jsd(base_probability_views, transformed_probability).sum(dim=1).double()
            invariance_agreement += transformed_logits.argmax(dim=-1).eq(base_prediction.unsqueeze(0)).sum(dim=1).double()
            invariance_difference_sq += (transformed_logits - base_logits.unsqueeze(0)).square().sum(dim=(1, 2)).double()
            invariance_reference_sq += base_logits.square().sum().double()
        else:
            for index, matrix in enumerate(transformations):
                transformed_fourmom = fourmom @ matrix.transpose(0, 1)
                combined_matrices = matrix.unsqueeze(0) @ base_matrices
                transformed_logits = model(transformed_fourmom, lorentz_matrix=combined_matrices)
                transformed_probability = F.softmax(transformed_logits, dim=-1)
                invariance_jsd[index] += _jsd(base_probability, transformed_probability).sum().double()
                invariance_agreement[index] += transformed_logits.argmax(dim=-1).eq(base_prediction).sum().double()
                invariance_difference_sq[index] += (transformed_logits - base_logits).square().sum().double()
                invariance_reference_sq[index] += base_logits.square().sum().double()
    nonempty = bin_counts > 0
    bin_accuracy = bin_correct[nonempty] / bin_counts[nonempty]
    mean_confidence = bin_confidence[nonempty] / bin_counts[nonempty]
    ece = ((bin_accuracy - mean_confidence).abs() * bin_counts[nonempty] / max(1, count)).sum()
    mean_jsd = float((invariance_jsd / max(1, count)).mean())
    mean_agreement = float((invariance_agreement / max(1, count)).mean())
    relative_error = torch.sqrt(invariance_difference_sq / invariance_reference_sq.clamp_min(torch.finfo(torch.float64).tiny))
    metrics = {'accu': correct / max(1, count), 'nlll': nll_sum / max(1, count), 'ecel': float(ece), 'li_jsd': 1.0 - mean_jsd, 'li_agree': mean_agreement, 'lorentz_relative_logit_error': float(relative_error.mean()), 'top1_agreement': fidelity_agreement / max(1, count) if teacher_logits is not None else None, 'teach_stu_jsd': 1.0 - fidelity_jsd / max(1, count) if teacher_logits is not None else None}
    result = {**identity, 'status': 'complete', 'validation_samples': count, 'metrics': metrics, 'invariance': {'transform_source': REFERENCE_LORENTZ_SOURCE, 'transform_count': transform_count, 'coordinate_adaptation': 'reference (px,py,pz,E) conjugated to stored (E,px,py,pz)', 'decomposition': 'P4 @ B_z(rapidity) @ Q4.T; P=U@U; Q=V@V', 'rapidity_distribution': 'torch.rand in [0,1)', 'li_jsd_definition': '1 - mean normalized Jensen-Shannon divergence', 'li_agree_definition': 'mean top-1 agreement with the untransformed view', 'relative_error_definition': 'mean ||logits(Lx)-logits(x)||_2 / ||logits(x)||_2'}, 'fid': {'top1_agreement': metrics['top1_agreement'], 'teach_stu_jsd': metrics['teach_stu_jsd']}}
    atomic_json(output_path, result)
    return result
