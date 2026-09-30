"""The long-format results table: one row per (checkpoint, evaluation split)."""

import csv
import fcntl
import math
import os
from pathlib import Path
from typing import Any, TextIO

from src.vocabulary import PROVENANCE_COLUMNS, RESULT_COLUMNS


class ResultsWriter:
    """Appends rows to one CSV, safely under the parallel evaluations of a sweep."""

    def __init__(
        self,
        path: str | Path,
        columns: tuple[str, ...] = RESULT_COLUMNS + PROVENANCE_COLUMNS,
    ) -> None:
        self.path = Path(path)
        self.columns = list(columns)

    def append(self, row: dict[str, Any]) -> None:
        """Append one row, writing the header first when the table is new.

        :param row: any subset of the columns; what is missing is written empty, so a
            stage that has no temperature leaves the column blank rather than inventing
            a value.
        :raises KeyError: on a key outside the columns, which would otherwise be
            dropped.
        :raises ValueError: when the table on disk carries another header.
        """
        payload = self._payload(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+", newline="") as table:
            # The evaluations of one multirun append to the same table from several
            # processes, and an unlocked append interleaves two half-written rows.
            fcntl.flock(table.fileno(), fcntl.LOCK_EX)
            try:
                self._write(table, payload)
            finally:
                fcntl.flock(table.fileno(), fcntl.LOCK_UN)

    def _payload(self, row: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(row) - set(self.columns))
        if unknown:
            raise KeyError(
                f"{unknown} are no columns of {self.path}. Add them to "
                "src/vocabulary.py, or correct the caller."
            )
        return {column: _cell(row.get(column, "")) for column in self.columns}

    def _write(self, table: TextIO, payload: dict[str, Any]) -> None:
        writer = csv.DictWriter(table, fieldnames=self.columns)
        if table.seek(0, os.SEEK_END) == 0:
            writer.writeheader()
        else:
            self._check_header()
            table.seek(0, os.SEEK_END)
        writer.writerow(payload)

    def _check_header(self) -> None:
        with open(self.path, newline="") as table:
            header = next(csv.reader(table), None)
        if header is None or header == self.columns:
            return
        raise ValueError(
            f"{self.path} carries the columns {header}, and this run writes "
            f"{self.columns}. Appending would misalign every value, so move the old "
            "table aside or point results.path elsewhere."
        )


def _cell(value: Any) -> Any:
    """NaN is a result (a metric no split could produce), never a reason to raise."""
    return "nan" if isinstance(value, float) and math.isnan(value) else value
