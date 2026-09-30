#!/usr/bin/env python3
"""Parameter and operation counts of every network the study trains.

Each experiment's teacher and student are instantiated from their own config and traced
once, so the numbers follow the configs rather than a second copy of the architectures.
The counts are per sample of the network's own input width, which is narrower than the
stored batch: a network reads the columns of its ``required_features`` alone.

python3 scripts/analysis/compute_costs.py python3 scripts/analysis/compute_costs.py
--experiments jets_deepsets --out /tmp/compute.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# configs/paths/default.yaml reads it, and a bare interpreter has neither .env nor the variable.
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from src.algorithms.utils.flops import count_flops_per_sample  # noqa: E402
from src.utils.omegaconf import register_resolvers  # noqa: E402

EXPERIMENTS = ("mnist_cnn", "jets_deepsets", "jets_lorentznet")
ROLES = ("teacher_net", "student_net")
COLUMNS = (
    "experiment",
    "role",
    "model",
    "input_shape",
    "params",
    "macs",
    "flops",
    "ebops",
)
# Bit operations at 8 bit x 8 bit, the convention this study quotes hardware cost in:
# one MAC costs b_a * b_w = 64 bit operations (Baskin et al. 2018, UNIQ).
BITS = 8
HEADER_NOTE = (
    f"# eBOPs = MACs x {BITS * BITS} ({BITS} bit activations x {BITS} bit weights, "
    "Baskin et al. 2018); MACs = FLOPs / 2, FLOPs from torch's FlopCounterMode, per sample."
)


def _config(experiment: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])


def _input_shape(cfg: DictConfig, model: torch.nn.Module) -> tuple[int, ...]:
    """The width the network itself reads: (constituents, columns) or the image
    shape."""
    required = getattr(model, "required_features", None)
    if required is None:
        return tuple(cfg.input_shape)
    return int(cfg.data.nconstituents), len(required)


def _row(experiment: str, role: str, cfg: DictConfig) -> dict[str, object]:
    model = instantiate(cfg[role])
    shape = _input_shape(cfg, model)
    flops = count_flops_per_sample(model, shape)
    return {
        "experiment": experiment,
        "role": role.removesuffix("_net"),
        "model": type(model).__name__,
        "input_shape": "x".join(str(dim) for dim in shape),
        "params": sum(p.numel() for p in model.parameters()),
        "macs": flops / 2.0,
        "flops": flops,
        "ebops": flops / 2.0 * BITS * BITS,
    }


def _rows(experiments: list[str]) -> list[dict[str, object]]:
    rows = []
    for experiment in experiments:
        cfg = _config(experiment)
        rows += [_row(experiment, role, cfg) for role in ROLES]
    return rows


def _write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(f"{HEADER_NOTE}\n")
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    register_resolvers()
    rows = _rows(args.experiments)
    _write(rows, args.out)
    for row in rows:
        print(
            f"{row['experiment']:>16} {row['role']:>7} {row['model']:>16} "
            f"params={row['params']:>9} MACs={row['macs']:>12.0f} eBOPs={row['ebops']:>14.0f}"
        )
    print(f"\n{len(rows)} network(s) -> {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "compute.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
