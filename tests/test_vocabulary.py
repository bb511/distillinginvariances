from src import vocabulary

CONTRACT = (
    "dataset, teacher, regime, method, temperature, alpha, beta, discrepancy, lambda_is, "
    "lambda_ic, lr, batch_size, epochs_run, seed, fold, n_folds, eval_split, accuracy, "
    "nll, ece, ece_bins, fid_top1, fid_1mjsd, fid_top1_g, fid_1mjsd_g, inv_top1, "
    "inv_1mjsd, inv_M, mean_confidence, mean_entropy, nll_ts, ece_ts, temperature_ts"
)


def test_the_contract_and_the_exported_methods() -> None:
    assert list(vocabulary.RESULT_COLUMNS) == CONTRACT.split(", ")
    assert set(vocabulary.METHOD_OF_STAGE) == set(vocabulary.STAGES) - {"cache", "hint"}
    assert set(vocabulary.METHOD_OF_STAGE.values()) <= set(vocabulary.METHODS)
