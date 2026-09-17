import json
import math
from pathlib import Path
import numpy as np

def aligned_paths(data_dir: Path) -> dict[str, Path]:
    return {'features': data_dir / 'x_mlp_train.npy', 'fourmom': data_dir / 'x_train.npy', 'jet_fourmom': data_dir / 'jet_train.npy', 'labels': data_dir / 'y_train.npy', 'feature_metadata': data_dir / 'student_features.json', 'teacher_metadata': data_dir / 'metadata_train.json'}

def _relative_features(constituents: np.ndarray, jets: np.ndarray, scale: np.ndarray) -> np.ndarray:
    valid = constituents[..., 0] != 0.0
    px, py, pz = (constituents[..., 1], constituents[..., 2], constituents[..., 3])
    pt = np.sqrt(px * px + py * py)
    eta = np.arcsinh(pz / np.maximum(pt, np.finfo(np.float32).eps))
    phi = np.arctan2(py, px)
    jet_pt = np.sqrt(jets[:, 1] * jets[:, 1] + jets[:, 2] * jets[:, 2])
    jet_eta = np.arcsinh(jets[:, 3] / np.maximum(jet_pt, np.finfo(np.float32).eps))
    jet_phi = np.arctan2(jets[:, 2], jets[:, 1])
    delta_phi = np.remainder(phi - jet_phi[:, None] + math.pi, 2.0 * math.pi) - math.pi
    value = np.stack((pt, eta - jet_eta[:, None], delta_phi), axis=-1)
    return value * valid[..., None] / scale.reshape(1, 1, 3)

def verify_aligned_dataset(data_dir):
    paths = aligned_paths(data_dir)
    metadata = json.loads(paths['feature_metadata'].read_text(encoding='utf-8'))
    return {'dataset_kind': 'aligned_original', 'samples': len(np.load(paths['labels'], mmap_mode='r')), 'student_scale': metadata['iqr']}
