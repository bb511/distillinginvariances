import os
from pathlib import Path
import tarfile
import h5py
import numpy as np
import torch
import urllib.request
from . import standardization

class HLS4MLData150:

    def __init__(self, root: str, nconst: int, feats: str, norm: str, train: bool):
        self.root = Path(root)
        self.nconst = nconst
        self.norm = norm
        self.feats = feats
        self.train = train
        self.type = 'train' if self.train else 'val'
        self.min_pt = 2
        self.train_url = 'https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz'
        self.test_url = 'https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz'
        self.preproc_output_name = f'{self.type}_{self.nconst}const.npy'
        self.proc_output_name = f'{self.type}_{self.norm}_{self.nconst}const_{self.feats}.npy'
        self.processed_dir = self.root / 'processed'
        if (self.root / f'x_{self.proc_output_name}').is_file() or (self.root / f'x_train_{self.norm}_{self.nconst}const_{self.feats}.npy').is_file():
            self.processed_dir = self.root
        self.data_file_dir = self.root / 'raw' / self.type
        self.x_pro = None
        self.y_pro = None
        self._get_processed_data()

    def _get_raw_data(self):
        if not self._check_raw_data_exists():
            self._download_data()
        return self.root / 'raw' / self.type

    def _check_raw_data_exists(self):
        if self.root.is_dir():
            raw_dir = self.root / 'raw'
            if raw_dir.is_dir():
                data_dir = raw_dir / self.type
                if data_dir.is_dir():
                    if any(data_dir.iterdir()):
                        return 1
        return 0

    def _check_processed_data_exists(self):
        if self.root.is_dir():
            proc_folder = self.processed_dir
            x_proc_file = proc_folder / f'x_{self.proc_output_name}'
            y_proc_file = proc_folder / f'y_{self.proc_output_name}'
            if x_proc_file.is_file() and y_proc_file.is_file():
                self.x_pro = np.load(x_proc_file)
                self.y_pro = np.load(y_proc_file)
                return 1
        return 0

    def _check_preprocessed_data_exists(self):
        if self.root.is_dir():
            proc_folder = self.processed_dir
            x_preproc_file = proc_folder / f'x_preproc_{self.preproc_output_name}'
            y_preproc_file = proc_folder / f'y_preproc_{self.preproc_output_name}'
            if x_preproc_file.is_file() and y_preproc_file.is_file():
                self.x_pre = np.load(x_preproc_file)
                self.y_pre = np.load(y_preproc_file)
                return 1
        return 0

    def _download_data(self):
        raw_dir = self.root / 'raw'
        if not raw_dir.is_dir():
            os.makedirs(raw_dir)
        if self.train:
            data_file_path = raw_dir / 'hls4ml_train.tar.gz'
            urllib.request.urlretrieve(self.train_url, data_file_path)
        else:
            data_file_path = raw_dir / 'hls4ml_val.tar.gz'
            urllib.request.urlretrieve(self.test_url, data_file_path)
        data_tar = tarfile.open(data_file_path, 'r:gz')
        data_tar.extractall(str(raw_dir), filter='data')
        data_tar.close()
        os.remove(data_file_path)

    def _import_raw_data(self):
        xs, ys = ([], [])
        files = sorted(list(self.data_file_dir.glob('*.h5')) + list(self.data_file_dir.glob('*.hdf5')))
        for path in files:
            with h5py.File(path, 'r') as data:
                xs.append(np.asarray(data['jetConstituentList']))
                ys.append(np.asarray(data['jets'][:, -6:-1]))
        return (np.concatenate(xs), np.concatenate(ys))

    def _preproc_raw_data(self, x_data: np.ndarray, y_data: np.ndarray):
        x_data, y_data = self._cut_transverse_momentum(x_data, y_data)
        x_data = self._restrict_nb_constituents(x_data)
        proc_dir = self.processed_dir
        if not proc_dir.is_dir():
            os.makedirs(proc_dir)
        np.save(proc_dir / f'x_preproc_{self.preproc_output_name}', x_data)
        np.save(proc_dir / f'y_preproc_{self.preproc_output_name}', y_data)
        return (x_data, y_data)

    def _get_processed_data(self):
        if not self._check_processed_data_exists():
            if not self._check_preprocessed_data_exists():
                self.data_file_dir = self._get_raw_data()
                self.x_raw, self.y_raw = self._import_raw_data()
                self.x_pre, self.y_pre = self._preproc_raw_data(self.x_raw, self.y_raw)
                del self.x_raw
                del self.y_raw
            self.x_pro, self.y_pro = self._process_data(self.x_pre, self.y_pre)
            del self.x_pre
            del self.y_pre

    def _load_preproc_train_data(self):
        preproc_file_name = f'x_preproc_train_{self.nconst}const.npy'
        try:
            x_data_train = np.load(self.processed_dir / preproc_file_name)
        except OSError:
            raise RuntimeError(f"Training preprocessed data not found at {self.root / 'processed' / preproc_file_name}. Process training data before validation data.")
        return x_data_train

    def _process_data(self, x_data: np.ndarray, y_data: np.ndarray):
        x_data = self._get_features(x_data, self.feats)
        if not self.train:
            x_data_train = self._load_preproc_train_data()
            x_data_train = self._get_features(x_data_train, self.feats)
            norm_params = standardization.fit_standardisation(self.norm, x_data_train)
            del x_data_train
        else:
            norm_params = standardization.fit_standardisation(self.norm, x_data)
        x_data = standardization.apply_standardisation(self.norm, x_data, norm_params)
        proc_folder = self.processed_dir
        np.save(proc_folder / f'x_{self.proc_output_name}', x_data)
        np.save(proc_folder / f'y_{self.proc_output_name}', y_data)
        del x_data
        del y_data
        return (np.load(proc_folder / f'x_{self.proc_output_name}'), np.load(proc_folder / f'y_{self.proc_output_name}'))

    def _get_features(self, data: np.ndarray, feat_selection: str):
        switcher = {'ptetaphi': lambda: self._select_features_ptetaphi(data), 'allfeats': lambda: self._select_features_all(data)}
        data = switcher.get(feat_selection, lambda: None)()
        if data is None:
            raise TypeError(f"Feature selection '{feat_selection}' not valid!")
        return data

    def _select_features_ptetaphi(self, data: np.ndarray):
        return data[:, :, [5, 8, 11]]

    def _select_features_all(self, data: np.ndarray):
        return data[:, :, :]

    def _cut_transverse_momentum(self, x_data: np.ndarray, y_data: np.ndarray):
        boolean_mask = x_data[:, :, 5] > self.min_pt
        structure_memory = boolean_mask.sum(axis=1)
        x_data = np.split(x_data[boolean_mask, :], np.cumsum(structure_memory)[:-1])
        x_data = [jet_const for jet_const in x_data if jet_const.size > 0]
        y_data = y_data[structure_memory > 0]
        return (x_data, y_data)

    def _restrict_nb_constituents(self, x_data: np.ndarray):
        for jet in range(len(x_data)):
            if x_data[jet].shape[0] >= self.nconst:
                x_data[jet] = x_data[jet][:self.nconst, :]
            else:
                padding_length = self.nconst - x_data[jet].shape[0]
                x_data[jet] = np.pad(x_data[jet], ((0, padding_length), (0, 0)))
        return np.array(x_data)

    def get_torch_tensors(self):
        x = torch.from_numpy(self.x_pro.astype(np.float32))
        y = torch.from_numpy(self.y_pro.astype(np.float32))
        return (x, y)

    def get_torch_dataset(self):
        x, y = self.get_torch_tensors()
        return torch.utils.data.TensorDataset(x, y)
