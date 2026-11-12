import importlib.util

from tests.conftest import ROOT


def _select_hparams():
    path = ROOT / "scripts" / "analysis" / "select_hparams.py"
    spec = importlib.util.spec_from_file_location("select_hparams", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _point(method: str, temperature: str, beta: str, score: float) -> dict[str, str]:
    return {
        "method": method,
        "temperature": temperature,
        "beta": beta,
        "accuracy_macro": str(score),
    }


def test_a_hint_beta_arm_selected_at_t_1_is_not_flagged() -> None:
    betas = {"0.1": 0.80, "0.25": 0.82, "0.5": 0.79}
    points = [_point("hint_beta", "1.0", beta, score) for beta, score in betas.items()]
    best = _select_hparams()._select(points)

    assert best["beta"] == "0.25"
    assert best["extend_with"] == ""


def test_a_temperature_grid_arm_selected_at_t_1_is_flagged_for_t_0_5() -> None:
    grid = {"1.0": 0.82, "2.0": 0.80, "4.0": 0.79, "8.0": 0.78, "16.0": 0.77}
    points = [_point("kd", t, "0.0", score) for t, score in grid.items()]
    best = _select_hparams()._select(points)

    assert best["temperature"] == "1.0"
    assert best["extend_with"] == "0.5"
