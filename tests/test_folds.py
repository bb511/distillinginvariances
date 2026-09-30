import torch

from src.data.components.folds import fold_indices


def test_the_folds_partition_the_pool_and_follow_the_seed() -> None:
    held_out = [fold_indices(100, 5, i, 42)[1] for i in range(5)]
    train, val = fold_indices(100, 5, 0, 42)

    assert sorted(torch.cat(held_out).tolist()) == list(range(100))
    assert sorted(train.tolist() + val.tolist()) == list(range(100))
    assert torch.equal(val, fold_indices(100, 5, 0, 42)[1])
    assert not torch.equal(val, fold_indices(100, 5, 0, 43)[1])
