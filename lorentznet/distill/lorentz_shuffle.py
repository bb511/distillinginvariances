import json
import os
from pathlib import Path
import numpy as np
import torch
REQUIRED_SOURCE_FILES = ('x_train.npy', 'y_train.npy', 'jet_train.npy', 'student_features.json')
REFERENCE_LORENTZ_SOURCE = 'jet_tagging_gdl-main/util.py:create_Lorentz_mat,check_invariance_Lorentz'
REFERENCE_LORENTZ_TRANSFORMS = 16

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def reference_lorentz_matrices(rotation_generator: np.random.RandomState, rapidity_generator: torch.Generator, count: int=REFERENCE_LORENTZ_TRANSFORMS) -> np.ndarray:
    if count < 1:
        raise ValueError(f'count must be positive, got {count}')
    random_matrices = rotation_generator.rand(count, 3, 3)
    u, _, v = np.linalg.svd(random_matrices)
    p = u @ u
    q = v @ v
    left = np.zeros((count, 4, 4), dtype=np.float64)
    right = np.zeros((count, 4, 4), dtype=np.float64)
    left[:, :3, :3] = p
    right[:, :3, :3] = np.swapaxes(q, 1, 2)
    left[:, 3, 3] = 1.0
    right[:, 3, 3] = 1.0
    rapidity = torch.rand(count, generator=rapidity_generator, dtype=torch.float32).numpy().astype(np.float64)
    boost = np.zeros((count, 4, 4), dtype=np.float64)
    boost[:, 0, 0] = 1.0
    boost[:, 1, 1] = 1.0
    boost[:, 2, 2] = np.cosh(rapidity)
    boost[:, 2, 3] = np.sinh(rapidity)
    boost[:, 3, 2] = np.sinh(rapidity)
    boost[:, 3, 3] = np.cosh(rapidity)
    matrix_px_py_pz_e = left @ boost @ right
    e_first = np.array([3, 0, 1, 2])
    return matrix_px_py_pz_e[:, e_first][:, :, e_first].astype(np.float32)

def seeded_reference_lorentz_matrices(seed: int, count: int=REFERENCE_LORENTZ_TRANSFORMS) -> np.ndarray:
    rotation_generator = np.random.RandomState(int(seed))
    rapidity_generator = torch.Generator(device='cpu')
    rapidity_generator.manual_seed(int(seed))
    return reference_lorentz_matrices(rotation_generator, rapidity_generator, count)

def minkowski_norm(x: np.ndarray) -> np.ndarray:
    value = x.astype(np.float64, copy=False)
    return value[..., 0] ** 2 - np.sum(value[..., 1:] ** 2, axis=-1)

def fourmom_to_relative_features(constituents: np.ndarray, jets: np.ndarray, scale: np.ndarray) -> np.ndarray:
    valid = constituents[..., 0] != 0.0
    px, py, pz = (constituents[..., 1], constituents[..., 2], constituents[..., 3])
    pt = np.sqrt(px * px + py * py)
    eta = np.arcsinh(pz / np.maximum(pt, np.finfo(np.float32).eps))
    phi = np.arctan2(py, px)
    jet_pt = np.sqrt(jets[:, 1] ** 2 + jets[:, 2] ** 2)
    jet_eta = np.arcsinh(jets[:, 3] / np.maximum(jet_pt, np.finfo(np.float32).eps))
    jet_phi = np.arctan2(jets[:, 2], jets[:, 1])
    delta_phi = (phi - jet_phi[:, None] + np.pi) % (2.0 * np.pi) - np.pi
    features = np.stack([pt, eta - jet_eta[:, None], delta_phi], axis=-1)
    features *= valid[..., None]
    return (features / scale[None, None, :]).astype(np.float32)

def _validate_source(source_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for name in REQUIRED_SOURCE_FILES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    x = np.load(source_dir / 'x_train.npy', mmap_mode='r')
    y = np.load(source_dir / 'y_train.npy', mmap_mode='r')
    jets = np.load(source_dir / 'jet_train.npy', mmap_mode='r')
    metadata = json.loads((source_dir / 'student_features.json').read_text(encoding='utf-8'))
    scale = np.asarray(metadata.get('iqr'), dtype=np.float32)
    if x.shape[1:] != (32, 4) or y.shape != (len(x),) or jets.shape != (len(x), 4):
        raise ValueError(f'Misaligned source arrays: x={x.shape}, y={y.shape}, jets={jets.shape}')
    if metadata.get('representation') != ['pt', 'delta_eta', 'delta_phi']:
        raise ValueError('Source student_features.json is not the HLS4ML pt/deta/dphi representation')
    if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f'Invalid robust-fast feature scale: {scale}')
    return (x, y, jets, scale)

def build_shuffled_dataset(source_dir: Path, output_dir: Path, seed: int, chunk_size: int=4096, limit: int | None=None) -> dict:
    x, y, jets, scale = _validate_source(source_dir)
    samples = min(len(x), limit or len(x))
    expected_base = {'schema_version': 3, 'samples': samples, 'constituents': 32, 'classes': 5, 'seed': int(seed), 'generation_chunk_size': int(chunk_size), 'four_vector_order': ['E', 'px', 'py', 'pz'], 'student_representation': ['pt', 'delta_eta', 'delta_phi'], 'student_scale': scale.tolist(), 'transform_policy': 'one frozen independent reference-package Lorentz transform per sample', 'reference_lorentz_source': REFERENCE_LORENTZ_SOURCE, 'reference_coordinate_order': ['px', 'py', 'pz', 'E'], 'reference_decomposition': 'P4 @ B_z(rapidity) @ Q4.T; P=U@U; Q=V@V', 'reference_rapidity_distribution': 'torch.rand in [0,1)', 'stored_matrix_convention': 'column matrix in (E,px,py,pz); apply row vectors with p @ matrix.T', 'beam_policy': "LorentzNet beam nodes receive the row's same matrix at model forward", 'source_samples': len(x)}
    manifest_path = output_dir / 'manifest.json'
    outputs = {'fourmom': output_dir / 'fourmom.npy', 'jet_fourmom': output_dir / 'jet_fourmom.npy', 'features': output_dir / 'features_pt_deta_dphi.npy', 'matrices': output_dir / 'lorentz_matrices.npy', 'labels': output_dir / 'labels.npy', 'sample_ids': output_dir / 'sample_ids.npy'}
    if manifest_path.is_file() and all((path.is_file() for path in outputs.values())):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((manifest.get(key) == value for key, value in expected_base.items())):
            return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    fourmom_out = np.lib.format.open_memmap(outputs['fourmom'], mode='w+', dtype=np.float32, shape=(samples, 32, 4))
    jet_out = np.lib.format.open_memmap(outputs['jet_fourmom'], mode='w+', dtype=np.float32, shape=(samples, 4))
    features_out = np.lib.format.open_memmap(outputs['features'], mode='w+', dtype=np.float32, shape=(samples, 32, 3))
    matrices_out = np.lib.format.open_memmap(outputs['matrices'], mode='w+', dtype=np.float32, shape=(samples, 4, 4))
    labels_out = np.lib.format.open_memmap(outputs['labels'], mode='w+', dtype=np.int64, shape=(samples,))
    ids_out = np.lib.format.open_memmap(outputs['sample_ids'], mode='w+', dtype=np.int64, shape=(samples,))
    rotation_generator = np.random.RandomState(seed)
    rapidity_generator = torch.Generator(device='cpu')
    rapidity_generator.manual_seed(int(seed))
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        matrix = reference_lorentz_matrices(rotation_generator, rapidity_generator, end - start)
        source_fourmom = np.asarray(x[start:end], dtype=np.float32)
        source_jets = np.asarray(jets[start:end], dtype=np.float32)
        transformed = np.einsum('bij,bnj->bni', matrix, source_fourmom, optimize=True)
        transformed_jets = np.einsum('bij,bj->bi', matrix, source_jets, optimize=True)
        valid = source_fourmom[..., 0] != 0.0
        fourmom_out[start:end] = transformed
        jet_out[start:end] = transformed_jets
        features_out[start:end] = fourmom_to_relative_features(transformed, transformed_jets, scale)
        matrices_out[start:end] = matrix
        labels_out[start:end] = np.asarray(y[start:end], dtype=np.int64)
        ids_out[start:end] = np.arange(start, end, dtype=np.int64)
    for array in (fourmom_out, jet_out, features_out, matrices_out, labels_out, ids_out):
        array.flush()
    manifest = {**expected_base, 'files': {name: path.name for name, path in outputs.items()}}
    atomic_json(manifest_path, manifest)
    return manifest

def verify_shuffled_dataset(data_dir):
    return json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
