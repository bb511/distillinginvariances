import csv
from pathlib import Path

import pytest

from src.utils.results import ResultsWriter
from src.vocabulary import PROVENANCE_COLUMNS, RESULT_COLUMNS


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as table:
        rows = list(csv.reader(table))
    return rows[0], [dict(zip(rows[0], row)) for row in rows[1:]]


def test_rows_land_under_the_contract_header(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    writer = ResultsWriter(path)

    writer.append({"dataset": "jets", "accuracy": float("nan")})
    writer.append({"eval_split": "test", "nll": 0.5})
    header, rows = _rows(path)

    assert header == list(RESULT_COLUMNS + PROVENANCE_COLUMNS)
    assert (rows[0]["accuracy"], rows[0]["teacher"], rows[1]["nll"]) == (
        "nan",
        "",
        "0.5",
    )
    with pytest.raises(KeyError):
        writer.append({"invariance": 1.0})
    with pytest.raises(ValueError):
        ResultsWriter(path, columns=("dataset",)).append({"dataset": "jets"})
