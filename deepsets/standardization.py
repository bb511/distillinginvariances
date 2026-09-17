import numpy as np

def apply_standardisation(choice: str, x_data: np.ndarray, norm_params: dict):
    if choice == 'nonorm':
        return x_data
    switcher = {'minmax': lambda: minmax_apply(x_data, **norm_params), 'robust': lambda: robust_apply(x_data, **norm_params), 'robust_fast': lambda: robust_fast_apply(x_data, **norm_params), 'standard': lambda: standard_apply(x_data, **norm_params)}
    x_data = switcher.get(choice, lambda: None)()
    if x_data is None:
        raise NameError(f'Type of normalisation does not exist! Please choose from the following list: {list(switcher.keys())}')
    return x_data

def fit_standardisation(choice: str, x_data: np.ndarray):
    if choice == 'nonorm':
        return {}
    switcher = {'minmax': lambda: minmax_fit(x_data), 'robust': lambda: robust_fit(x_data), 'robust_fast': lambda: robust_fit(x_data), 'standard': lambda: standard_fit(x_data)}
    norm_params = switcher.get(choice, lambda: None)()
    if norm_params is None:
        raise NameError(f'Type of normalisation does not exist! Please choose from the following list: {list(switcher.keys())}')
    return norm_params

def minmax_fit(x: np.ndarray) -> dict:
    min_feats = x.min(axis=0).min(axis=0)
    max_feats = x.max(axis=0).max(axis=0)
    return {'min_feats': min_feats, 'max_feats': max_feats}

def minmax_apply(x: np.ndarray, min_feats: np.ndarray, max_feats: np.ndarray, feature_range: tuple=(0, 1)) -> np.ndarray:
    x_norm = (x - min_feats) / (max_feats - min_feats)
    x_norm = x_norm * (feature_range[1] - feature_range[0]) + feature_range[0]
    return x_norm

def robust_fit(x: np.ndarray, percentiles: list=[95, 5]) -> dict:
    x_median = []
    interquantile_range = []
    for feature_idx in range(x.shape[-1]):
        x_feature = x[:, :, feature_idx].flatten()
        x_median.append(np.nanmedian(x_feature, axis=0))
        quantile_high, quantile_low = np.nanpercentile(x_feature, percentiles)
        interquantile_range.append(quantile_high - quantile_low)
    return {'x_median': x_median, 'interquantile_range': interquantile_range}

def robust_apply(x: np.ndarray, x_median: np.ndarray, interquantile_range: np.ndarray):
    return (x - x_median) / interquantile_range

def robust_fast_apply(x: np.ndarray, x_median: np.ndarray, interquantile_range: np.ndarray):
    return x / interquantile_range

def standard_fit(x: np.ndarray) -> dict:
    x_mean = []
    x_std = []
    for feature_idx in range(x.shape[-1]):
        x_feature = x[:, :, feature_idx].flatten()
        x_mean.append(x_feature.mean(axis=0))
        x_std.append(x_feature.std(axis=0))
    return {'x_mean': x_mean, 'x_std': x_std}

def standard_apply(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (x - x_mean) / x_std
