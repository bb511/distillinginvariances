import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {"dataset": "jets", "group": "permutation", "regime": "canonical", "alpha": "0"}
BASE |= {"beta": "0.0", "discrepancy": "", "eval_data": "canonical", "variant": ""}
BASE |= {"eval_split": "test", "layer": "logits"}


def _profile(teacher, method, temperature, init, fold, value) -> dict[str, object]:
    cells = {"teacher": teacher, "method": method, "temperature": temperature}
    cells |= {"init": init, "fold": fold, "probe_group": "permutation"}
    return BASE | cells | {"invariance": value}


def _erasure(statistic: str, fold: int, value: float) -> dict[str, object]:
    cells = {
        "teacher": "deepsets",
        "method": "kd",
        "temperature": "4",
        "init": "trained",
    }
    cells |= {
        "fold": fold,
        "kind": "erasure",
        "target": "frame",
        "statistic": statistic,
    }
    return BASE | cells | {"value": value, "control_low": -0.1, "control_high": 0.1}


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _table(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def test_the_fold_summaries_pair_controls_init_and_hint(tmp_path: Path) -> None:
    # Per fold f: kd 0.6 + 0.01 f, hint 0.8, the mlp control 0.5; random init 0.1, no fold.
    profile = [_profile("deepsets", "kd", "4", "random", "main", 0.1)]
    for fold in range(1, 6):
        profile += [
            _profile("deepsets", "kd", "4", "trained", fold, 0.6 + 0.01 * fold),
            _profile("deepsets", "hint", "1", "trained", fold, 0.8),
            _profile("mlp", "kd", "4", "trained", fold, 0.5),
        ]
    # d_invariance: 0.3 on every fold, outside the range, frame used; d_accuracy_g:
    # outside on every fold but with alternating sign, not used.
    probes = [_erasure("d_invariance", f, 0.3) for f in range(1, 6)]
    probes += [_erasure("d_accuracy_g", f, 0.2 * (-1) ** f) for f in range(1, 6)]
    _write(tmp_path / "profile.csv", profile)
    _write(tmp_path / "probes.csv", probes)
    script = ROOT / "scripts" / "analysis" / "profile_probes.py"
    command = [sys.executable, str(script), "--out-dir", str(tmp_path / "out")]
    command += ["--profile", str(tmp_path / "profile.csv")]
    subprocess.run([*command, "--probes", str(tmp_path / "probes.csv")], check=True)

    summary = _table(tmp_path / "out" / "profile_folds.csv")
    diffs = _table(tmp_path / "out" / "profile_differences.csv")
    rule = {r["statistic"]: r for r in _table(tmp_path / "out" / "erasure_rule.csv")}

    kd = next(
        r
        for r in summary
        if r["method"] == "kd" and r["teacher"] == "deepsets" and r["init"] == "trained"
    )
    assert abs(float(kd["mean"]) - 0.63) < 1e-12 and kd["n_folds"] == "5"
    by_comparison = {(r["comparison"], r["method"]): r for r in diffs}
    assert abs(float(by_comparison[("control", "kd")]["mean"]) - 0.13) < 1e-12
    assert by_comparison[("control", "kd")]["reference"] == "mlp"
    assert abs(float(by_comparison[("random_init", "kd")]["mean"]) - 0.53) < 1e-12
    assert abs(float(by_comparison[("hint_vs_kd", "hint")]["mean"]) - 0.17) < 1e-12
    assert rule["d_invariance"]["used"] == "True"
    assert rule["d_accuracy_g"]["used"] == "False"
    assert rule["d_accuracy_g"]["n_outside"] == "5"
