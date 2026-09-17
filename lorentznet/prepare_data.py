import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from .distill.training import atomic_json
from .distill.lorentz_shuffle import fourmom_to_relative_features


def selected_h5_rows(path, chunk_size=4096):
    with h5py.File(path, "r") as handle:
        for start in range(0, len(handle["jetConstituentList"]), chunk_size):
            raw = np.asarray(handle["jetConstituentList"][start:start + chunk_size])
            jets = np.asarray(handle["jets"][start:start + chunk_size])
            mask = raw[..., 5] > 2.0
            keep = mask.any(axis=1)
            raw, jets, mask = raw[keep], jets[keep], mask[keep]
            output = np.zeros((len(raw), 32, raw.shape[-1]), dtype=np.float32)
            ranks = np.cumsum(mask, axis=1) - 1
            selected = mask & (ranks < 32)
            rows, constituents = np.nonzero(selected)
            output[rows, ranks[selected]] = raw[rows, constituents]
            first = np.argmax(mask, axis=1)
            rows = np.arange(len(raw))
            phi = (raw[rows, first, 10] - raw[rows, first, 11] + np.pi) % (2 * np.pi) - np.pi
            pt, eta, mass = jets[:, 1], jets[:, 2], jets[:, 3]
            px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
            energy = np.sqrt(np.maximum(0.0, mass * mass + px * px + py * py + pz * pz))
            jet_fourmom = np.stack([energy, px, py, pz], axis=-1)
            labels = np.argmax(jets[:, -6:-1], axis=1).astype(np.int64)
            yield output, jet_fourmom, labels


def prepare_data(data_root, output_dir, limit=None, chunk_size=4096, require_features=True):
    data_root, output_dir = Path(data_root), Path(output_dir)
    names = ("x_train.npy", "y_train.npy", "jet_train.npy", "x_mlp_train.npy", "student_features.json")
    for directory in (data_root, data_root / "aligned", data_root / "processed"):
        if all((directory / name).is_file() for name in names):
            return directory
        if all((directory / name).is_file() for name in names if name != "x_mlp_train.npy"):
            if not require_features:
                return directory
            output_dir.mkdir(parents=True, exist_ok=True)
            source_files = [directory / name for name in names if name != "x_mlp_train.npy"]
            prepared_from = {"directory": str(directory.resolve()), "modified": [path.stat().st_mtime_ns for path in source_files]}
            if all((output_dir / name).is_file() for name in names):
                previous = json.loads((output_dir / "student_features.json").read_text(encoding="utf-8"))
                if previous.get("prepared_from") == prepared_from:
                    return output_dir
            for name in names:
                if name != "x_mlp_train.npy" and directory.resolve() != output_dir.resolve():
                    shutil.copyfile(directory / name, output_dir / name)
            fourmom = np.load(directory / "x_train.npy", mmap_mode="r")
            jets = np.load(directory / "jet_train.npy", mmap_mode="r")
            metadata = json.loads((directory / "student_features.json").read_text(encoding="utf-8"))
            scale = np.asarray(metadata["iqr"], dtype=np.float32)
            features = np.lib.format.open_memmap(output_dir / "x_mlp_train.npy", mode="w+", dtype=np.float32, shape=(len(fourmom), 32, 3))
            for start in range(0, len(fourmom), chunk_size):
                end = start + chunk_size
                features[start:end] = fourmom_to_relative_features(fourmom[start:end], jets[start:end], scale)
            features.flush()
            atomic_json(output_dir / "student_features.json", {**metadata, "prepared_from": prepared_from})
            return output_dir
    directories = (data_root / "raw" / "train", data_root / "train", data_root)
    files = next((sorted(directory.glob("*.h5")) for directory in directories if any(directory.glob("*.h5"))), [])
    if not files:
        raise FileNotFoundError(f"No aligned arrays or HLS4ML training HDF5 shards in {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if all((output_dir / name).is_file() for name in names):
        metadata = json.loads((output_dir / "student_features.json").read_text(encoding="utf-8"))
        if metadata.get("source") == str(data_root.resolve()) and metadata.get("limit") == limit:
            return output_dir
    total = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            for start in range(0, len(handle["jetConstituentList"]), chunk_size):
                pt = np.asarray(handle["jetConstituentList"][start:start + chunk_size, :, 5])
                total += int((pt > 2.0).any(axis=1).sum())
                if limit and total >= limit:
                    break
        if limit and total >= limit:
            break
    total = min(total, limit) if limit else total
    fourmom = np.lib.format.open_memmap(output_dir / "x_train.npy", mode="w+", dtype=np.float32, shape=(total, 32, 4))
    jets = np.lib.format.open_memmap(output_dir / "jet_train.npy", mode="w+", dtype=np.float32, shape=(total, 4))
    features = np.lib.format.open_memmap(output_dir / "x_mlp_train.npy", mode="w+", dtype=np.float32, shape=(total, 32, 3))
    labels = np.lib.format.open_memmap(output_dir / "y_train.npy", mode="w+", dtype=np.int64, shape=(total,))
    cursor = 0
    for path in files:
        for raw, jet, y in selected_h5_rows(path, chunk_size):
            take = min(len(raw), total - cursor)
            fourmom[cursor:cursor + take] = raw[:take, :, [3, 0, 1, 2]]
            features[cursor:cursor + take] = raw[:take, :, [5, 8, 11]]
            jets[cursor:cursor + take] = jet[:take]
            labels[cursor:cursor + take] = y[:take]
            cursor += take
            if cursor == total:
                break
        if cursor == total:
            break
    scale = np.array([np.percentile(features[..., axis], 95) - np.percentile(features[..., axis], 5) for axis in range(3)], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    for start in range(0, total, chunk_size):
        features[start:start + chunk_size] /= scale
    for array in (fourmom, jets, features, labels):
        array.flush()
    atomic_json(output_dir / "student_features.json", {"samples": total, "representation": ["pt", "delta_eta", "delta_phi"], "normalization": "robust_fast P95-P5", "iqr": scale.tolist(), "source": str(data_root.resolve()), "limit": limit})
    atomic_json(output_dir / "metadata_train.json", {"samples": total, "constituents": 32, "four_vector_order": ["E", "px", "py", "pz"]})
    return output_dir
