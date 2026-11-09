# The names every producer and consumer of results share. Imports nothing, so the analysis
# scripts read it without Hydra or Lightning.

DATASETS = ("mnist", "jets", "top_tagging")
# The cells' teachers, then the control teachers imported from the sibling repository.
TEACHERS = (
    "cnn_circular",
    "cnn_zeros",
    "deepsets",
    "lorentznet",
    "lloca_z",
    "lloca_so3",
    "lloca_lorentz",
    "mlp",
    "jetformer",
    "jetformer_aug_so2",
    "jetformer_symm_so2",
)
# The teachers of the control and learned-invariance rows. Each distils at its cell
# teacher's selected point, so none is selected on, and none gives a firmware student.
CONTROL_TEACHERS = (
    "cnn_zeros",
    "mlp",
    "jetformer",
    "jetformer_aug_so2",
    "jetformer_symm_so2",
)
REGIMES = ("canonical", "transformed", "augmented")
# The regime whose teacher run and cache each regime distils from: an invariant
# teacher's outputs on a canonical row stand for any transform of it.
TEACHER_REGIME = {
    "canonical": "canonical",
    "transformed": "transformed",
    "augmented": "canonical",
}


def regime_tag(regime: str, group: str) -> str:
    """The regime as it names run directories and caches.

    Canonical rows are the same in every cell, so canonical runs and caches are shared
    across the cells; transformed and augmented rows depend on the cell's group.
    """
    return regime if regime == "canonical" else f"{regime}_{group}"


# translation, permutation, z, so3 and lorentz_iso are the groups of the study's cells; so2
# and lorentz are kept so augmentation.py stays a superset of the sibling repository's. z is
# a rotation about the beam composed with a boost along it.
GROUPS = ("translation", "permutation", "so2", "so3", "lorentz", "lorentz_iso", "z")
STAGES = (
    "teacher",
    "baseline",
    "cache",
    "hint",
    "kd",
    "kd_hint",
    "hint_beta",
    "relational",
    "logit_matching",
    "label_smoothing",
)
METHODS = (
    "teacher",
    "baseline",
    "kd",
    "hint",
    "hint_beta",
    "relational",
    "logit_matching",
    "label_smoothing",
    "constant",
)
OBJECTIVES = ("ce", "hint", "kd", "logit", "relational")
DISCREPANCIES = ("mse", "cosine", "smooth_l1", "distance_mse")
EVAL_SPLITS = ("fold", "test")
# The rows an evaluation reads (eval_data), and the rows each training regime is
# evaluated on by default: the augmented regime validates and tests on the transformed
# regime's rows.
EVAL_DATA = ("canonical", "transformed")
DATA_OF_REGIME = {
    "canonical": "canonical",
    "transformed": "transformed",
    "augmented": "transformed",
}
# The groups every checkpoint is probed under after its cell's own, per dataset.
PROBE_SUBGROUPS = {
    "jets": ("so2", "lorentz", "so3", "lorentz_iso"),
    "top_tagging": ("so2", "lorentz", "so3", "lorentz_iso"),
    "mnist": (),
}
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
    "logit_matching": "logit_matching",
    "label_smoothing": "label_smoothing",
}

# The columns of results.csv, in this order: the figure code selects rows by them. One
# row per (checkpoint, eval_split, probe). A training knob, temperature to
# label_smoothing, is filled only on the stages whose objective reads it (STAGE_KNOBS in
# src/utils/evaluation.py); label_smoothing is the epsilon of the label_smoothing stage's
# targets. A probe knob is filled only on the groups it acts on, and a blank max_shift or
# n_swaps there means the whole group.
# Metric names: *_top1 are top-1 agreements (higher is better); *_sqrtjsd are mean
# sqrt(JSD) in bits after each model's own temperature (lower is better); _g columns are
# on the first probe draw g.x; inv_* compare a network with itself, tinv_* the teacher
# with itself; inv_top1 and inv_sqrtjsd average over inv_M draws. The *_1mjsd columns
# are the legacy 1 - JSD of the raw logits.
# accuracy_macro is the mean per-class recall, the training loop's val/acc; on the main
# run's validation split it is the val/acc_best that Block 3 of the registration selects on.
# fmt: off
RESULT_COLUMNS = (
    "dataset", "teacher", "regime", "method",
    "temperature", "alpha", "beta", "discrepancy", "lambda_is", "lambda_ic",
    "label_smoothing", "lr", "batch_size", "epochs_run", "seed", "fold", "n_folds",
    "eval_split", "eval_data", "probe_group", "probe_eta_max", "probe_max_shift",
    "probe_n_swaps",
    "accuracy", "accuracy_macro", "nll", "ece", "ece_bins",
    "fid_top1", "fid_sqrtjsd", "fid_top1_g", "fid_sqrtjsd_g",
    "inv_top1", "inv_sqrtjsd", "inv_M", "inv_top1_g", "inv_sqrtjsd_g",
    "tinv_top1_g", "tinv_sqrtjsd_g",
    "mean_confidence", "mean_entropy",
    "nll_ts", "ece_ts", "temperature_ts", "temperature_teacher_ts",
    "fid_1mjsd", "fid_1mjsd_g", "inv_1mjsd",
)
PROVENANCE_COLUMNS = (
    "stage", "variant", "optimizer", "scheduler", "guided_index", "group",
    "checkpoint_path", "run_dir", "config_hash",
)
# The columns of coverage.csv (src/coverage.py): one row per (cell, eval_data, probe,
# magnitude bin); a blank magnitude is the whole draw. features names the student's
# input columns, or "pixels".
COVERAGE_COLUMNS = (
    "dataset", "teacher", "group", "features", "eval_data",
    "probe_group", "probe_eta_max", "probe_max_shift", "probe_n_swaps",
    "magnitude", "bin_low", "bin_high", "n_pairs", "auc", "ci_low", "ci_high",
    "n_boot", "seed", "classifier_hidden", "classifier_epochs",
)
# The columns of profile.csv (src/orbit_profile.py): one row per (checkpoint, probe, layer).
# init is "trained", or "random" for the same architecture at initialisation; nv is
# filled for an exhaustive orbit alone.
PROFILE_COLUMNS = (
    *RESULT_COLUMNS[: RESULT_COLUMNS.index("accuracy")],
    "init", "layer", "layer_index", "width", "n_live", "n_samples", "n_draws",
    "exhaustive", "invariance", "ci_low", "ci_high", "nv",
    *PROVENANCE_COLUMNS,
)
# The columns of probes.csv (src/concept_probe.py): one row per (checkpoint, layer, kind,
# target, statistic). kind is frame or concept (value: held-out R^2) or erasure (value: the
# change under LEACE, beside the mean and 95 % range of equal-rank random removals).
PROBE_COLUMNS = (
    *RESULT_COLUMNS[: RESULT_COLUMNS.index("accuracy")],
    "init", "layer", "layer_index", "width", "kind", "target", "statistic", "value",
    "ci_low", "ci_high", "rank", "control_mean", "control_low", "control_high",
    "n_train", "n_test",
    *PROVENANCE_COLUMNS,
)
# fmt: on
