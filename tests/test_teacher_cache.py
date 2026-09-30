from pathlib import Path

import numpy as np
import pytest

from src.data.components.teacher_cache import TeacherCache, read_cache, write_cache


def test_the_cache_round_trips_and_refuses_another_run(tmp_path: Path) -> None:
    logits = np.arange(35, dtype=np.float32).reshape(7, 5)
    cache = TeacherCache(
        logits, np.ones((7, 4)), np.zeros((3, 5)), np.zeros((3, 4)), {"fold": 1}
    )
    path = tmp_path / "fold_1.npz"

    write_cache(path, cache)
    read = read_cache(path, {"fold": 1})

    assert np.array_equal(read.pool_logits, logits) and read.meta == {"fold": 1}
    with pytest.raises(ValueError, match="does not match"):
        read_cache(path, {"fold": 2})
