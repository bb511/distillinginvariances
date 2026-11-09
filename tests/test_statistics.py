import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from src.utils.statistics import corrected_t, equivalence_p, transfer_fraction

ROOT = Path(__file__).resolve().parents[1]


def test_the_corrected_test_reduces_to_the_paired_t_test_without_overlap() -> None:
    deltas = [0.012, 0.018, 0.009, 0.015, 0.021]

    plain, corrected = corrected_t(deltas, 0.0), corrected_t(deltas, 0.25)

    assert abs(plain["p_value"] - stats.ttest_1samp(deltas, 0.0).pvalue) < 1e-12
    assert corrected["p_value"] > plain["p_value"] and corrected["df"] == 4
    expected = math.sqrt((1 / 5 + 0.25) * np.var(deltas, ddof=1))
    assert abs(corrected["se"] - expected) < 1e-15


def test_one_fold_gives_a_mean_and_no_spread() -> None:
    single = corrected_t([0.3], 0.25)

    assert single["mean"] == 0.3 and math.isnan(single["ci_low"])


def test_tost_accepts_a_tight_null_and_rejects_a_shift() -> None:
    tight = [0.001, -0.002, 0.0, 0.002, -0.001]
    shifted = [0.05, 0.06, 0.04, 0.055, 0.045]

    assert equivalence_p(tight, 0.25, 0.02) < 0.01
    assert equivalence_p(shifted, 0.25, 0.02) > 0.5


def test_the_transfer_fraction_reads_agreements_and_distances() -> None:
    assert abs(transfer_fraction(0.6, 0.5, 0.9) - 0.25) < 1e-12
    assert abs(transfer_fraction(0.2, 0.3, 0.1) - 0.5) < 1e-12
    assert math.isnan(transfer_fraction(0.4, 0.5, 0.5))


BLANKS = ("temperature", "alpha", "beta", "discrepancy", "lambda_is", "lambda_ic", "lr")
PROBE = ("probe_eta_max", "probe_max_shift", "probe_n_swaps")


def _cell(teacher: str) -> dict[str, str]:
    cell = {"dataset": "jets", "teacher": teacher, "eval_data": "canonical"}
    cell |= {"probe_group": "permutation", "group": "permutation"}
    return cell | dict.fromkeys((*BLANKS, *PROBE), "")


def _run(script: str, rows: list[dict[str, object]], tmp_path: Path, *extra: str):
    table, out = tmp_path / "results.csv", tmp_path / f"{script}.csv"
    columns = sorted({key for row in rows for key in row})
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    path = ROOT / "scripts" / "analysis" / f"{script}.py"
    command = [sys.executable, str(path), "--results", str(table), "--out", str(out)]
    subprocess.run([*command, *extra], check=True)
    return list(csv.DictReader(out.open()))


def test_the_transfer_fraction_follows_block_3(tmp_path: Path) -> None:
    rows = []
    # deepsets: a ceiling gap of 0.4 on the main run, margin 0.04, fraction defined;
    # lorentznet: a gap of 0.005, below the 0.01 floor, fraction undefined.
    for teacher, gap in (("deepsets", 0.4), ("lorentznet", 0.005)):
        cell = _cell(teacher)
        for fold, split in [("main", "fold")] + [(f, "test") for f in range(1, 6)]:
            student = 0.6 + (0.001 * fold if split == "test" else 0.0)
            for regime, method, value in (
                ("canonical", "baseline", 0.5),
                ("augmented", "baseline", 0.5 + gap),
                ("canonical", "kd", student),
            ):
                keys = {"fold": fold, "eval_split": split, "regime": regime}
                rows.append(cell | keys | {"method": method, "inv_top1": value})

    result = {r["teacher"]: r for r in _run("transfer", rows, tmp_path)}

    assert result["deepsets"]["fraction_defined"] == "True"
    assert abs(float(result["deepsets"]["fraction_mean"]) - 0.2575) < 1e-9
    assert abs(float(result["deepsets"]["margin"]) - 0.04) < 1e-12
    assert result["lorentznet"]["fraction_defined"] == "False"
    assert result["lorentznet"]["fraction_mean"] == ""
    assert float(result["lorentznet"]["margin"]) == 0.01


def test_the_selection_ties_to_the_lower_temperature_and_flags_an_edge(
    tmp_path: Path,
) -> None:
    keys = {"fold": "main", "eval_split": "fold", "regime": "canonical", "alpha": "0"}
    rows = [
        _cell("deepsets")
        | keys
        | {"method": "kd", "stage": "kd", "temperature": t}
        | {"accuracy_macro": score, "accuracy": score}
        for t, score in ((1, 0.8005), (2, 0.8), (4, 0.7), (16, 0.6))
    ]
    rows.append(
        _cell("deepsets")
        | keys
        | {"method": "baseline", "stage": "baseline"}
        | {"accuracy_macro": 0.9, "alpha": ""}
    )
    firmware = tmp_path / "firmware.csv"

    selected = _run("select_hparams", rows, tmp_path, "--firmware-out", str(firmware))
    (kd,) = (r for r in selected if r["method"] == "kd")
    (pick,) = list(csv.DictReader(firmware.open()))

    assert (kd["temperature"], kd["extend_with"], kd["n_points"]) == ("1", "0.5", "4")
    assert pick["method"] == "kd"
