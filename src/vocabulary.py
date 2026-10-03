# The names every producer and consumer of results share. Imports nothing, so the analysis
# scripts read it without Hydra or Lightning.

DATASETS = ("mnist", "jets")
TEACHERS = ("cnn_circular", "cnn_zeros", "deepsets", "lorentznet")
REGIMES = ("canonical", "transformed", "augmented")
# The regime whose teacher run and cache each regime distils from: an invariant
# teacher's outputs on a canonical row stand for any transform of it.
TEACHER_REGIME = {
    "canonical": "canonical",
    "transformed": "transformed",
    "augmented": "canonical",
}
# translation, permutation and lorentz_iso are the groups of the study; the rest are kept so
# augmentation.py stays a superset of the sibling repository's.
GROUPS = ("translation", "permutation", "so2", "so3", "lorentz", "lorentz_iso")
STAGES = (
    "teacher",
    "baseline",
    "cache",
    "hint",
    "kd",
    "kd_hint",
    "hint_beta",
    "relational",
)
METHODS = ("teacher", "baseline", "kd", "hint", "hint_beta", "relational", "constant")
OBJECTIVES = ("ce", "hint", "kd", "relational")
DISCREPANCIES = ("mse", "cosine", "smooth_l1", "distance_mse")
EVAL_SPLITS = ("fold", "test")
# The fold the manifest, the cache meta and the result rows carry for the main run (fold
# null); the folds themselves are the integers 1..n_folds.
MAIN_FOLD = "main"

# The exported method of each trained stage. The hint stage itself is no classifier (its
# layers above the guided one stay at initialisation), so the paper's hint row is the KD
# student warm-started from it.
METHOD_OF_STAGE = {
    "teacher": "teacher",
    "baseline": "baseline",
    "kd": "kd",
    "kd_hint": "hint",
    "hint_beta": "hint_beta",
    "relational": "relational",
}

# The columns of results.csv, in this order: the figure code selects rows by them.
# fmt: off
RESULT_COLUMNS = (
    "dataset", "teacher", "regime", "method",
    "temperature", "alpha", "beta", "discrepancy", "lambda_is", "lambda_ic",
    "lr", "batch_size", "epochs_run", "seed", "fold", "n_folds", "eval_split",
    "accuracy", "nll", "ece", "ece_bins",
    "fid_top1", "fid_1mjsd", "fid_top1_g", "fid_1mjsd_g",
    "inv_top1", "inv_1mjsd", "inv_M",
    "mean_confidence", "mean_entropy",
    "nll_ts", "ece_ts", "temperature_ts",
)
PROVENANCE_COLUMNS = (
    "stage", "variant", "optimizer", "scheduler", "guided_index", "group",
    "checkpoint_path", "run_dir", "config_hash",
)
# fmt: on
