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


BLANKS = ("temperature", "alpha", "beta", "discrepancy", "lambda_is", "lambda_ic")
BLANKS += ("label_smoothing", "lr", "variant", "seed")
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
    # lorentznet: a gap of 0.005, below the 0.02 floor of Block 5, fraction undefined.
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
    assert float(result["lorentznet"]["margin"]) == 0.02


def test_the_margin_averages_the_seeds_and_ignores_variants(tmp_path: Path) -> None:
    cell, rows = _cell("deepsets"), []
    # Main run: baselines at 0.4 and 0.6 over two seeds (mean 0.5), the ceiling at 0.9 and
    # 1.1 (mean 1.0), so the gap is 0.5 and the margin 0.05; a wide baseline at 0.0 and a
    # wide student on every fold would shift both the margin and the gain if read.
    for seed, base, top in (("12345", 0.4, 0.9), ("12346", 0.6, 1.1)):
        for regime, value in (("canonical", base), ("augmented", top)):
            keys = {
                "fold": "main",
                "eval_split": "fold",
                "regime": regime,
                "seed": seed,
            }
            rows.append(cell | keys | {"method": "baseline", "inv_top1": value})
    for fold in range(1, 6):
        keys = {"fold": fold, "eval_split": "test", "seed": "12345"}
        for regime, method, variant, value in (
            ("canonical", "baseline", "", 0.5),
            ("canonical", "baseline", "wide", 0.0),
            ("augmented", "baseline", "", 1.0),
            ("canonical", "kd", "", 0.75),
        ):
            extra = {"regime": regime, "method": method, "variant": variant}
            rows.append(cell | keys | extra | {"inv_top1": value})

    (result,) = _run("transfer", rows, tmp_path)

    assert abs(float(result["margin"]) - 0.05) < 1e-12
    assert abs(float(result["gain_mean"]) - 0.25) < 1e-12
    assert abs(float(result["fraction_mean"]) - 0.5) < 1e-12


def test_a_student_pairs_with_its_control_teacher_student(tmp_path: Path) -> None:
    rows = []
    # deepsets kd at T 4 against the mlp control at T 4: a gain of 0.1 + 0.01 x fold; the mlp
    # control at T 8, a wide deepsets student and a deepsets student at T 2, which has no
    # control, must not enter the pairs.
    for fold in range(1, 6):
        keys = {
            "fold": fold,
            "eval_split": "test",
            "regime": "canonical",
            "method": "kd",
        }
        for teacher, temperature, variant, value in (
            ("deepsets", "4", "", 0.6 + 0.01 * fold),
            ("deepsets", "4", "wide", 0.0),
            ("deepsets", "2", "", 0.0),
            ("mlp", "4", "", 0.5),
            ("mlp", "8", "", 0.9),
        ):
            extra = {"temperature": temperature, "alpha": "0", "variant": variant}
            rows.append(_cell(teacher) | keys | extra | {"inv_top1": value})
    extra = ("--method", "kd", "--baseline", "kd", "--reference-teacher", "mlp")

    (result,) = _run(
        "paired_differences", rows, tmp_path, *extra, "--metrics", "inv_top1"
    )

    assert result["teacher"] == "deepsets" and result["reference"] == "mlp:kd"
    assert result["n_folds"] == "5"
    assert abs(float(result["mean_delta"]) - 0.13) < 1e-12
    assert result["reference_temperature"] == ""


def test_a_relational_student_pairs_with_kd_at_its_selected_temperature(
    tmp_path: Path,
) -> None:
    cell, rows = _cell("deepsets"), []
    # The main run selects T 4 at alpha 0 and T 2 at alpha 1; every T is on the folds, and
    # the unselected ones would shift the difference if read.
    for alpha, scores in (("0", {"2": 0.70, "4": 0.80, "8": 0.75}), ("1", {"2": 0.9})):
        for temperature, score in scores.items():
            keys = {"fold": "main", "eval_split": "fold", "regime": "canonical"}
            knobs = {"method": "kd", "temperature": temperature, "alpha": alpha}
            rows.append(cell | keys | knobs | {"accuracy_macro": score})
    fold_kd = {("0", "2"): 0.0, ("0", "4"): 0.5, ("0", "8"): 0.9, ("1", "2"): 0.4}
    for fold in range(1, 6):
        keys = {"fold": fold, "eval_split": "test", "regime": "canonical"}
        for (alpha, temperature), value in fold_kd.items():
            knobs = {"method": "kd", "temperature": temperature, "alpha": alpha}
            rows.append(cell | keys | knobs | {"inv_top1": value})
        for alpha in ("0", "1"):
            knobs = {"method": "relational", "alpha": alpha, "discrepancy": "cosine"}
            rows.append(cell | keys | knobs | {"inv_top1": 0.6 + 0.01 * fold})
    extra = ("--method", "relational", "--baseline", "kd", "--metrics", "inv_top1")

    result = {r["alpha"]: r for r in _run("paired_differences", rows, tmp_path, *extra)}

    assert [result[a]["reference_temperature"] for a in ("0", "1")] == ["4", "2"]
    assert all(result[a]["n_folds"] == "5" for a in ("0", "1"))
    assert abs(float(result["0"]["mean_delta"]) - 0.13) < 1e-12
    assert abs(float(result["1"]["mean_delta"]) - 0.23) < 1e-12


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


def test_the_selection_reports_label_smoothing_whole_and_names_every_experiment(
    tmp_path: Path,
) -> None:
    keys = {"fold": "main", "eval_split": "fold", "regime": "canonical"}
    rows = []
    for dataset, teacher in (
        ("jets", "lloca_z"),
        ("top_tagging", "lorentznet"),
        ("mnist", "cnn_circular"),
        ("mnist", "cnn_zeros"),
    ):
        cell = _cell(teacher) | keys | {"dataset": dataset}
        logit = {"method": "logit_matching", "stage": "logit_matching", "alpha": "0"}
        rows.append(cell | logit | {"accuracy_macro": 0.8})
        smoothed = {"method": "label_smoothing", "stage": "label_smoothing"}
        rows += [
            cell | smoothed | {"label_smoothing": eps, "accuracy_macro": 0.9}
            for eps in ("0.05", "0.1", "0.2")
        ]
    firmware = tmp_path / "firmware.csv"

    selected = _run("select_hparams", rows, tmp_path, "--firmware-out", str(firmware))
    picks = list(csv.DictReader(firmware.open()))

    assert {r["method"] for r in selected} == {"logit_matching"}
    assert {r["method"] for r in picks} == {"logit_matching"} and len(picks) == 3
    assert sorted(r["overrides"].split()[0] for r in selected) == [
        "experiment=jets_lloca_z",
        "experiment=mnist_cnn",
        "experiment=top_tagging_lorentznet",
    ]


def test_the_selection_ranks_the_cell_grid_alone(tmp_path: Path) -> None:
    keys = {"fold": "main", "eval_split": "fold", "regime": "canonical", "alpha": "0"}
    kd = {"method": "kd", "stage": "kd", "temperature": "4", "variant": ""}
    rows = [
        _cell("lloca_z") | keys | kd | {"seed": "12345", "accuracy_macro": 0.80},
        _cell("lloca_z")
        | keys
        | kd
        | {"seed": "12345", "temperature": "8"}
        | {"accuracy_macro": 0.70},
        _cell("lloca_z")
        | keys
        | kd
        | {"seed": "12346", "temperature": "8"}
        | {"accuracy_macro": 0.95},
        _cell("lloca_z")
        | keys
        | kd
        | {"seed": "12345", "temperature": "8"}
        | {"variant": "wide", "accuracy_macro": 0.99},
        _cell("jetformer") | keys | kd | {"seed": "12345", "accuracy_macro": 0.99},
    ]
    firmware = tmp_path / "firmware.csv"

    selected = _run("select_hparams", rows, tmp_path, "--firmware-out", str(firmware))
    picks = list(csv.DictReader(firmware.open()))

    assert [(r["teacher"], r["temperature"]) for r in selected] == [("lloca_z", "4")]
    assert [r["teacher"] for r in picks] == ["lloca_z"]
    assert selected[0]["n_points"] == "2"
