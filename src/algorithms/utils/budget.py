"""The trigger budget of the students: multiply-accumulates per sample.

A dense layer of fan-in a and fan-out b costs a * b multiply-accumulates; biases and
activations are free. The searches cap the total at ``max_macs`` (30k, 03_student.md).
"""

from itertools import product

from torch import nn


def stack_macs(sizes: list[int]) -> int:
    """Multiply-accumulates of a dense stack whose widths, the input and the output
    included, are *sizes*."""
    return sum(fan_in * fan_out for fan_in, fan_out in zip(sizes, sizes[1:]))


def linear_macs(net: nn.Module) -> int:
    """Multiply-accumulates of every ``nn.Linear`` in *net*, which for an MLP is its
    whole cost."""
    linears = (m for m in net.modules() if isinstance(m, nn.Linear))
    return sum(m.in_features * m.out_features for m in linears)


def feasible_stacks(
    in_dim: int,
    out_dim: int,
    widths: list[int],
    depths: list[int],
    max_macs: int,
) -> list[tuple[int, ...]]:
    """Every hidden stack of a depth in *depths* over *widths* within *max_macs*, in
    order of depth, then lexicographically."""
    stacks = (h for d in depths for h in product(sorted(widths), repeat=d))
    return [h for h in stacks if stack_macs([in_dim, *h, out_dim]) <= max_macs]
