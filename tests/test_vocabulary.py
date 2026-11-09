from src import vocabulary

CONTRACT = (
    "dataset, teacher, regime, method, temperature, alpha, beta, discrepancy, lambda_is, "
    "lambda_ic, label_smoothing, lr, batch_size, epochs_run, seed, fold, n_folds, "
    "eval_split, eval_data, "
    "probe_group, probe_eta_max, probe_max_shift, probe_n_swaps, accuracy, "
    "accuracy_macro, nll, ece, ece_bins, fid_top1, fid_sqrtjsd, fid_top1_g, "
    "fid_sqrtjsd_g, inv_top1, inv_sqrtjsd, inv_M, inv_top1_g, inv_sqrtjsd_g, "
    "tinv_top1_g, tinv_sqrtjsd_g, mean_confidence, mean_entropy, nll_ts, ece_ts, "
    "temperature_ts, temperature_teacher_ts, fid_1mjsd, fid_1mjsd_g, inv_1mjsd"
)


def test_the_contract_and_the_exported_methods() -> None:
    assert list(vocabulary.RESULT_COLUMNS) == CONTRACT.split(", ")
    assert set(vocabulary.METHOD_OF_STAGE) == set(vocabulary.STAGES) - {"cache", "hint"}
    assert set(vocabulary.METHOD_OF_STAGE.values()) <= set(vocabulary.METHODS)


def test_every_regime_distils_a_teacher_of_a_regime_that_distils_itself() -> None:
    assert set(vocabulary.TEACHER_REGIME) == set(vocabulary.REGIMES)
    for teacher_regime in vocabulary.TEACHER_REGIME.values():
        assert vocabulary.TEACHER_REGIME[teacher_regime] == teacher_regime


def test_every_regime_is_evaluated_on_rows_of_a_known_regime() -> None:
    assert set(vocabulary.DATA_OF_REGIME) == set(vocabulary.REGIMES)
    assert set(vocabulary.DATA_OF_REGIME.values()) == set(vocabulary.EVAL_DATA)
    assert set(vocabulary.PROBE_SUBGROUPS) == set(vocabulary.DATASETS)
    subgroups = {g for groups in vocabulary.PROBE_SUBGROUPS.values() for g in groups}
    assert subgroups <= set(vocabulary.GROUPS)
