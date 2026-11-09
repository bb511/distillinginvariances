#!/usr/bin/env python3
"""Synthesise an exported quantised network with alkaid and append one CSV row.

jet_tagging_gdl's ``synthesise_vn.py``, made generic in the top module and the input
layout. The input is what ``scripts/synthesis/export_hgq.py`` writes for an ``hgq``
checkpoint: the directory ``model.keras``, ``inputs.npz``, ``export.json`` and
``reference.npz``, whose logits are the bit-exact reference. The top module is named by
``--project``, else by the ``port`` of ``export.json``.

1. Keras runs on JAX's CPU backend at the ``floatx`` of ``export.json`` (float64
   reproduces the fixed-point arithmetic exactly, float32 does not); another backend is
   refused, since XLA's float64 2**f is inexact on a GPU;
2. ``import hgq`` registers its layers and alkaid's plugin for them; the model loads;
3. Keras predicts the exported words, and a drift from the reference logits is logged;
4. alkaid traces the model into combinational logic, as ``alkaid convert`` does but
   without its pass ``canonicalize_msb_wraps`` (see ``_disable_msb_wrap_pass``) and
   with the constant products of its matrix rows kept (``_keep_constant_products``), its
   constant matrix-vector products solved by ``--method0`` at ``--hard-dc``; under
   ``--ternary aware`` its adder pairs are then fused latency-aware, so that the
   emulation of step 5 and the co-simulation of step 7 check the fused logic;
5. the Keras and the alkaid predictions are written to ``<out>/keras_outputs.npz`` and
   ``<out>/alkaid_outputs.npz`` in the layout of the reference file; alkaid's emulation
   is then compared with the reference logits: ``n_mismatch`` counts the logit entries
   that differ, and more than ``--max-mismatch`` stops the run, after the two files are
   written; the macro (mean per-class recall, torchmetrics' default) and micro
   accuracies follow;
6. the Verilog project ``<out>/prj/`` is written for ``--part`` at ``--clock`` ns, with
   alkaid's own ternary-adder fusion under ``--ternary fused`` alone, and
   ``<out>/check.json`` records steps 3 to 7;
7. unless ``--no-cosim``, Verilator co-simulates ``--cosim-jets`` jets, which must
   match alkaid's emulation exactly;
8. unless ``--no-vivado``, Vivado implements the project out of context through
   alkaid's ``build_vivado_prj.tcl`` (reports in ``prj/output_<project>/reports/``);
9. alkaid's project report (``prj/report.json``) and ``check.json`` make one row of
   ``--csv``; the Vivado columns hold NaN when Vivado has not run. An existing
   ``--csv`` whose header is not today's is refused before step 1.

Nothing is imported from ``src``, so the script runs wherever
``scripts/synthesis/requirements.txt`` is installed. A project written on one machine
and implemented on another takes two calls, the second (step 9 alone) once Vivado has
run in ``<dir>/prj``::

    python scripts/synthesis/synthesise.py --export-dir <export dir> --out <dir> \\
        --no-cosim --no-vivado
    python scripts/synthesis/synthesise.py --out <dir> --report-only

``--estimate-only`` needs ``--export-dir`` alone and runs neither Keras nor a tool: it
traces the model under every CMVM method of ``METHODS`` at every hard_dc of
``SCAN_HARD_DC``, keeps the setting of fewest pipeline stages at ``--latency-cutoff``
(of equal stages, the shortest delay) and writes ``<out>/estimate.json``: the setting,
its cycles and their nominal latency at ``--clock``, and the DSPs and LUTs Vivado would
use (``DSP_MIN_BITS``, ``LUTS_PER_COST``), so that a checkpoint is chosen on hardware
before any implementation.
"""

import argparse
import csv
import functools
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("synthesise")

# Beside the export, what export_hgq.py writes the reference logits and labels to.
REFERENCE_FILE = "reference.npz"
VIVADO_ARGS = (
    "-mode",
    "batch",
    "-nojournal",
    "-log",
    "vivado.log",
    "-source",
    "build_vivado_prj.tcl",
)
PREDICT_BATCH = 16384
DISTRIBUTIONS = (("alkaid", "alkaid"), ("hgq", "hgq2"), ("keras", "keras"))

# Command-line option -> keyword of alkaid's RTLModel.
RTL_OPTIONS = {
    "part": "part_name",
    "clock": "clock_period",
    "clock_uncertainty": "clock_uncertainty",
    "latency_cutoff": "latency_cutoff",
}

# CSV column -> key of alkaid's project report, which merges metadata.json (cost,
# registers, latency in cycles, nominal clock) with the post-route Vivado reports.
REPORT_FIELDS = {
    "latency_cycles": "latency",
    "alkaid_cost": "cost",
    "reg_bits": "reg_bits",
    "lut": "LUT",
    "ff": "FF",
    "dsp": "DSP",
    "bram_tile": "Block RAM Tile",
    "ramb18": "RAMB18",
    "uram": "URAM",
}
VIVADO_DONE = {"LUT", "WNS(ns)"}
# alkaid's CMVM methods; at hard_dc 0, wmc and mc are solved as wmc-dc and mc-dc.
METHODS = ("wmc", "mc", "wmc-pdc", "mc-pdc")
# The ternary-adder fusion: alkaid's (RTLModel's default), none, or fuse_aware's.
TERNARY = ("fused", "off", "aware")
# --estimate-only traces under every method of METHODS at each of these hard_dc.
SCAN_HARD_DC = (0, 1)
# Vivado's mapping at 5 ns, fitted on the 13 designs of jet_tagging_gdl implemented by
# 2026-10-14 (JEDI-linear and the vector neurons): a product of two variables whose
# narrower operand has more than DSP_MIN_BITS bits lands in a DSP (counts within 2 %),
# and the LUTs are LUTS_PER_COST of alkaid's cost without those products (0.89 to 0.95).
DSP_MIN_BITS = 5
LUTS_PER_COST = 0.92

CSV_FIELDS = (
    "model",
    "nconstituents",
    "beta",
    "ebops",
    "part",
    "clock_ns",
    "latency_cutoff",
    "ternary",
    "method0",
    "ii",
    "latency_cycles",
    "latency_ns",
    "lut",
    "ff",
    "dsp",
    "bram_tile",
    "ramb18",
    "uram",
    "wns_ns",
    "fmax_mhz",
    "alkaid_cost",
    "reg_bits",
    "acc_macro_keras",
    "acc_macro_alkaid",
    "acc_micro_alkaid",
    "n_test",
    "n_mismatch",
    "max_abs_diff",
    "n_cosim",
    "cosim_exact",
    "vivado_done",
    "export_dir",
    "prj_dir",
    "alkaid_version",
    "hgq_version",
    "keras_version",
    "timestamp",
)

OPTIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--export-dir", {"type": Path, "help": "the --out of export_hgq.py"}),
    (
        "--outputs-npz",
        {"type": Path, "help": f"the reference; None: <export-dir>/{REFERENCE_FILE}"},
    ),
    ("--out", {"type": Path, "required": True, "help": "for check.json and prj/"}),
    ("--project", {"help": "top module; None: export.json's port"}),
    ("--csv", {"type": Path, "help": "CSV appended to; None: <out>/synthesis.csv"}),
    ("--part", {"default": "xcvu13p-flga2577-2-e", "help": "FPGA part"}),
    ("--clock", {"type": float, "default": 5.0, "help": "clock period (ns)"}),
    (
        "--clock-uncertainty",
        {"type": float, "default": 0.1, "help": "fraction of the clock period"},
    ),
    (
        "--latency-cutoff",
        {"type": float, "default": 5.0, "help": "combinational delay per stage (ns)"},
    ),
    ("--hard-dc", {"type": int, "default": 5, "help": "alkaid's delay constraint"}),
    ("--method0", {"choices": METHODS, "default": "wmc", "help": "alkaid's CMVM"}),
    ("--ternary", {"choices": TERNARY, "default": "fused", "help": "adder fusion"}),
    ("--cosim-jets", {"type": int, "default": 1000, "help": "jets co-simulated"}),
    ("--no-cosim", {"action": "store_true", "help": "skip the co-simulation"}),
    ("--no-vivado", {"action": "store_true", "help": "skip the Vivado run"}),
    ("--vivado", {"default": "vivado", "help": "Vivado executable"}),
    ("--threads", {"type": int, "default": 16, "help": "emulation, Verilator"}),
    (
        "--max-mismatch",
        {"type": int, "default": 0, "help": "differing logit entries tolerated"},
    ),
    (
        "--report-only",
        {"action": "store_true", "help": "only <out>/prj and check.json to a row"},
    ),
    (
        "--estimate-only",
        {"action": "store_true", "help": "scan alkaid, write <out>/estimate.json"},
    ),
)


class _Formatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep the steps of the description as written and show every default."""


def _timed(step: str, fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)`` and log the time it took."""
    LOG.info("%s ...", step)
    start = time.perf_counter()
    result = fn(*args)
    LOG.info("%s: %.1f s", step, time.perf_counter() - start)
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one JSON object, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def missing_tools(args: argparse.Namespace) -> list[str]:
    """The external tools that the requested steps need and PATH lacks."""
    skipped = {args.vivado: args.no_vivado, "verilator": args.no_cosim}
    alone = args.report_only or args.estimate_only
    needed = [tool for tool, skip in skipped.items() if not (skip or alone)]
    return [tool for tool in needed if shutil.which(tool) is None]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line, refusing a run whose inputs or tools are missing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    for flag, options in OPTIONS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if args.estimate_only and args.export_dir is None:
        parser.error("--export-dir is needed under --estimate-only")
    if not (args.report_only or args.estimate_only or args.export_dir):
        parser.error("--export-dir is needed without --report-only")
    if args.export_dir and args.outputs_npz is None:
        args.outputs_npz = args.export_dir / REFERENCE_FILE
    if missing := missing_tools(args):
        parser.error(f"not on PATH: {', '.join(missing)}; see --no-cosim, --no-vivado")
    args.csv = args.csv or args.out / "synthesis.csv"
    return args


def _float64_kept(name: str) -> Callable[[Any], Any]:
    """Keras's elementwise *name* on JAX, computed in float64 for a float64 tensor.

    A copy of ``_float64_kept`` in src/algorithms/components/hgq_port.py, since this
    script imports nothing from src.
    """
    import jax
    import keras
    from keras.src import backend as keras_backend

    keras_fn, jax_fn = getattr(keras_backend.numpy, name), getattr(jax.numpy, name)

    def fn(x):
        x = keras.ops.convert_to_tensor(x)
        if keras.backend.standardize_dtype(x.dtype) == "float64":
            return jax_fn(x)
        return keras_fn(x)

    return fn


def _keep_float64_rounding() -> None:
    """Make Keras's floor and ceil keep a float64 tensor in float64.

    Keras 3.15 casts it to float32 there on JAX, so HGQ2's rounding floor(x 2^f + 1/2)
    misrounds a value within float32 resolution of a tie, and the Keras check would
    report logits that differ from alkaid's where the hardware agrees. The component
    src/algorithms/components/hgq_port.py applies the same patch at import.
    """
    from keras.src import backend as keras_backend

    for name in ("floor", "ceil"):
        setattr(keras_backend.numpy, name, _float64_kept(name))


def _start_jax(floatx: str) -> None:
    """Start JAX on the CPU, unless ``JAX_PLATFORMS`` names another platform, and in x64
    under float64.

    :raises SystemExit: when JAX runs elsewhere than on the CPU: on a GPU, XLA's float64
        2**f is off by an ulp for some f, so that Keras would round some jets otherwise
        than alkaid.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", floatx == "float64")
    if (backend := jax.default_backend()) != "cpu":
        raise SystemExit(
            f"JAX runs on {backend!r}, where XLA's float64 2**f is inexact, so Keras "
            "would not be bit-exact against alkaid; set JAX_PLATFORMS=cpu."
        )


def import_keras(floatx: str) -> Any:
    """Import Keras on JAX at ``floatx``, with HGQ2's layers registered for loading.

    Under float64 the x64 mode is switched on before any array exists, since Keras would
    otherwise round the products of quantised values to float32, and floor and ceil are
    made to keep float64.
    """
    os.environ["KERAS_BACKEND"] = "jax"
    _start_jax(floatx)
    import hgq  # noqa: F401
    import keras

    keras.config.set_floatx(floatx)
    _keep_float64_rounding()
    return keras


def load_model(export_dir: Path, floatx: str) -> Any:
    """Load the exported ``model.keras`` without its training configuration."""
    keras = import_keras(floatx)
    return keras.models.load_model(export_dir / "model.keras", compile=False)


def load_reference(
    export_dir: Path, outputs_npz: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the quantised words (rows, ..., F) in the export's row order, the
    reference logits (rows, classes) and the class labels (rows,)."""
    with np.load(export_dir / "inputs.npz") as inputs, np.load(outputs_npz) as outputs:
        words = inputs["inputs_clean"].astype(np.float64)
        logits = outputs["logits_clean"].astype(np.float64)
        labels = outputs["labels"].astype(np.int64)
    if not len(words) == len(logits) == len(labels):
        raise ValueError(f"{len(words)} exported rows against {len(logits)} logits")
    return words, logits, labels


def split_inputs(model: Any, words: np.ndarray) -> list[np.ndarray]:
    """Split the words (rows, ..., F) along the feature axis into the model's inputs.

    An input of the words' rank, (..., k), takes k features and one of lower rank takes
    one, in the order of ``model.inputs``, which is also the order in which alkaid
    flattens them: an MLP's flat (rows, F) words are its one (F,) input, and a jet
    network's (rows, N, F) words split into inputs of shape (N,) and (N, k).
    """
    parts, start = [], 0
    for shape in (tuple(tensor.shape[1:]) for tensor in model.inputs):
        width = shape[-1] if len(shape) == words.ndim - 1 else 1
        parts.append(words[..., start : start + width].reshape(len(words), *shape))
        start += width
    if start != words.shape[-1]:
        raise ValueError(f"model reads {start} features, export {words.shape[-1]}")
    return parts


def flatten(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate the inputs per jet, the layout of the RTL's single input port."""
    return np.concatenate([part.reshape(len(part), -1) for part in parts], axis=1)


def keras_predict(model: Any, parts: list[np.ndarray]) -> np.ndarray:
    """Return the Keras logits (rows, classes) in float64."""
    feed = parts[0] if len(parts) == 1 else parts
    logits = model.predict(feed, batch_size=PREDICT_BATCH, verbose=0)
    return np.asarray(logits, dtype=np.float64)


def _ternary_op(ops: list[Any], root_idx: int, child_idx: int) -> Any:
    """The opcode-11 ternary adder of the binary adder *root_idx* and its single-use
    binary operand *child_idx*, as alkaid's ``fuse_ternary_adders`` builds it."""
    from alkaid.trace.passes.ternary import _to_ternary_terms
    from alkaid.types import Op

    root, terms = ops[root_idx], _to_ternary_terms(ops, root_idx, child_idx)
    addr = tuple(idx for idx, _, _ in terms)
    data = tuple(x for _, sign, shift in terms for x in (int(sign > 0), shift))
    return Op(addr, 11, data, root.qint, root.latency, root.cost)


def fuse_aware(comb: Any) -> Any:
    """Fuse the adder pairs alkaid would fuse, keeping a ternary adder only when it is
    no later than the binary root it replaces, so that no output is delayed.

    From jet_tagging_gdl's latency study of 2026-10-09 on alkaid 0.8.1 and the fusion of
    every design of its FPGA comparison (``--ternary aware``), but not an alkaid option:
    alkaid scores a pair by the adders it saves alone, and fusing a root with one child
    of a balanced tree lengthens the path, the sibling staying binary.
    """
    from alkaid.trace.passes import add_surrogate, dead_code_elimin
    from alkaid.trace.passes.surrogate import cost_lat_ternary_add
    from alkaid.trace.passes.ternary import _fusion_leafs

    ops = list(comb.ops)
    for root_idx, child_idx in _fusion_leafs(comb).items():
        root, ops[root_idx] = ops[root_idx], _ternary_op(comb.ops, root_idx, child_idx)
        _, delay = cost_lat_ternary_add(root_idx, ops, comb.carry_size % 65535)
        ready = max(comb.ops[idx].latency for idx in ops[root_idx].addr)
        if ready + delay > root.latency:
            ops[root_idx] = root
    fused = comb._replace(ops=ops)
    return add_surrogate(dead_code_elimin(fused), _skip_op8_cost=True)


def _disable_msb_wrap_pass() -> None:
    """Replace alkaid's pass ``canonicalize_msb_wraps`` by the identity.

    In alkaid 0.8.1 the pass sets to zero the shift of a one-bit quantise op whose
    output step equals the MSB weight of its source, but the step is measured after the
    shift and the MSB before it, so a nonzero shift is dropped and the op reads another
    bit of its source. The emulation and the RTL carry the error alike and only the
    Keras check sees it; without the pass the affected JEDI-linear designs of
    jet_tagging_gdl were exact at the same cost (2026-10-12).
    """
    # TODO(Patrick): remove once alkaid fixes the pass upstream.
    import alkaid.trace.passes.canon as canon

    canon.canonicalize_msb_wraps = lambda comb: comb


def _keep_constant_products() -> None:
    """Make alkaid's product of a traced and a constant matrix accept a row whose
    product is constant.

    In alkaid 0.8.1 and 0.8.2 a traced matrix that is not constant throughout passes
    each row to ``cmvm``, which sums the solver's parts into an array whose hardware
    configuration it takes from a traced variable among them. Where every element of the
    row that meets a nonzero weight is constant, none is left and StopIteration is
    raised. Such a row arose where the EBOPs penalty had pruned a dense layer to two
    weights and coarsened its input quantiser until the elements they meet took single
    values (2026-10-15). Its product is computed directly here, as alkaid computes a
    constant matrix's.
    """
    # TODO(Patrick): remove once alkaid fixes cmvm upstream.
    import alkaid.trace.fixed_variable_array as fva

    if hasattr(fva.cmvm, "__wrapped__"):
        return
    cmvm = fva.cmvm

    @functools.wraps(cmvm)
    def constant_kept(cm: np.ndarray, v: Any, solver_options: Any) -> np.ndarray:
        used = np.any(cm != 0, axis=1)
        row = np.asarray(v)[used]
        if any(x.low != x.high for x in row):
            return cmvm(cm, v, solver_options)
        return np.array([x.low for x in row], dtype=np.float64) @ cm[used]

    fva.cmvm = constant_kept


def trace_comb(model: Any, options: dict[str, Any], ternary: str) -> Any:
    """Trace the model into alkaid's combinational logic with the solver *options*, as
    ``alkaid convert`` does but without ``canonicalize_msb_wraps`` and with the constant
    products of its matrix rows kept, and fuse its adders latency-aware under *ternary*
    ``aware``."""
    from alkaid.converter import trace_model
    from alkaid.trace import HWConfig, trace

    _disable_msb_wrap_pass()
    _keep_constant_products()
    inputs, outputs = trace_model(model, HWConfig(1, 1, -1), options)
    comb = trace(inputs, outputs)
    return fuse_aware(comb) if ternary == "aware" else comb


def macro_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean per-class recall of the argmax, torchmetrics' default
    ``MulticlassAccuracy``.

    As there, a class neither present nor predicted is left out of the mean, and a class
    predicted but absent enters it with recall 0.
    """
    classes = np.arange(logits.shape[1])[:, None]
    predicted, true = logits.argmax(axis=1) == classes, labels == classes
    hits, support = (predicted & true).sum(axis=1), true.sum(axis=1)
    recall = np.divide(hits, support, out=np.zeros(len(classes)), where=support > 0)
    return float(recall[(predicted | true).any(axis=1)].mean())


def emulation_stats(
    y_keras: np.ndarray, y_alkaid: np.ndarray, logits: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Compare both predictions (rows, classes) with the reference logits and score
    them."""
    diff = np.abs(y_alkaid - logits)
    return {
        "n_test": len(labels),
        "n_mismatch": int(np.count_nonzero(diff)),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "n_mismatch_keras": int(np.count_nonzero(y_keras != logits)),
        "acc_macro_keras": macro_accuracy(y_keras, labels),
        "acc_macro_alkaid": macro_accuracy(y_alkaid, labels),
        "acc_micro_alkaid": float(np.mean(y_alkaid.argmax(axis=1) == labels)),
    }


def run_record(args: argparse.Namespace, export: dict[str, Any]) -> dict[str, Any]:
    """The run's identity: the exported row, the target and the tool versions."""
    target = {"part": args.part, "clock_ns": args.clock}
    return {
        "model": export["row"],
        **{key: export[key] for key in ("nconstituents", "beta", "ebops")},
        "project": args.project,
        **target,
        "latency_cutoff": args.latency_cutoff,
        "ternary": args.ternary,
        "method0": args.method0,
        "export_dir": str(args.export_dir.resolve()),
        "prj_dir": str((args.out / "prj").resolve()),
        **{f"{name}_version": version(dist) for name, dist in DISTRIBUTIONS},
    }


def record_emulation(
    args: argparse.Namespace, export: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any]:
    """Write and return ``check.json``; stop if alkaid strays from the reference
    logits."""
    check = {**run_record(args, export), **stats}
    write_json(args.out / "check.json", check)
    level = logging.WARNING if stats["n_mismatch_keras"] else logging.INFO
    LOG.log(level, "Keras: %(n_mismatch_keras)d logits differ", stats)
    LOG.info("alkaid: %(n_mismatch)d logits differ, max %(max_abs_diff)g", stats)
    LOG.info("macro accuracy: Keras %(acc_macro_keras).6f", stats)
    LOG.info("macro accuracy: alkaid %(acc_macro_alkaid).6f", stats)
    if stats["n_mismatch"] > args.max_mismatch:
        raise SystemExit(f"{stats['n_mismatch']} mismatches exceed --max-mismatch")
    return check


def write_rtl(args: argparse.Namespace, comb: Any, meta: dict[str, Any]) -> Any:
    """Pipeline the logic and write the Verilog project ``<out>/prj/``."""
    from alkaid.codegen import RTLModel

    kwargs = {name: getattr(args, key) for key, name in RTL_OPTIONS.items()}
    kwargs["ternary_fuse"] = args.ternary == "fused"
    rtl = RTLModel(comb, args.out / "prj", args.project, flavor="verilog", **kwargs)
    rtl.write(meta)
    return rtl


def initiation_interval(rtl: Any) -> int | None:
    """The period of the input schedule in clock cycles, None without a schedule."""
    schedule = rtl.fsm.inp_signals[0].schedule
    return None if schedule is None else int(schedule.period)


def cosimulate(
    rtl: Any, meta: dict[str, Any], flat: np.ndarray, y_alkaid: np.ndarray, threads: int
) -> dict[str, Any]:
    """Build the Verilator emulator of the project and compare it with alkaid's."""
    # compile() writes the project again, so the metadata has to come along.
    rtl.compile(nproc=threads, metadata=meta)
    y_rtl = np.asarray(rtl.predict(flat, n_threads=threads), dtype=np.float64)
    return {"n_cosim": len(flat), "cosim_exact": bool(np.array_equal(y_rtl, y_alkaid))}


def generate(
    args: argparse.Namespace,
    comb: Any,
    flat: np.ndarray,
    y_alkaid: np.ndarray,
    check: dict[str, Any],
) -> None:
    """Steps 6 and 7: write the RTL project and, unless ``--no-cosim``, co-simulate."""
    meta = {key: check[key] for key in ("model", "nconstituents", "beta", "ebops")}
    rtl = _timed("RTL generation", write_rtl, args, comb, meta)
    check.update(ii=initiation_interval(rtl), n_cosim=0, cosim_exact=None)
    if not args.no_cosim:
        jets = slice(args.cosim_jets)
        cosim = (rtl, meta, flat[jets], y_alkaid[jets], args.threads)
        check.update(_timed("Verilator co-simulation", cosimulate, *cosim))
    write_json(args.out / "check.json", check)
    if check["cosim_exact"] is False:
        raise SystemExit("the Verilator co-simulation differs from alkaid's emulation")


def write_predictions(out: Path, labels: np.ndarray, **logits: np.ndarray) -> None:
    """Write ``<out>/<name>_outputs.npz`` for each keyword, in the layout of the
    reference file: ``logits_clean`` (rows, classes) in float64, since a port without
    output quantiser may carry more bits than float32's 24, and ``labels`` (rows,) in
    int8."""
    out.mkdir(parents=True, exist_ok=True)
    for name, values in logits.items():
        arrays = {"logits_clean": values.astype(np.float64)}
        np.savez(out / f"{name}_outputs.npz", **arrays, labels=labels.astype(np.int8))


def predict(
    args: argparse.Namespace, model: Any, parts: list[np.ndarray], labels: np.ndarray
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Steps 3 to 5 up to the comparison: the Keras prediction, alkaid's trace and its
    emulation, both predictions written beside ``check.json``.

    :return: the traced logic and the Keras and alkaid logits (rows, classes).
    """
    y_keras = _timed("Keras prediction", keras_predict, model, parts)
    options = {"hard_dc": args.hard_dc, "method0": args.method0}
    comb = _timed("alkaid trace", trace_comb, model, options, args.ternary)
    y_alkaid = _timed("alkaid emulation", comb.predict, parts, args.threads)
    write_predictions(args.out, labels, keras=y_keras, alkaid=y_alkaid)
    return comb, y_keras, y_alkaid


def project_name(args: argparse.Namespace, export: dict[str, Any]) -> str:
    """The top module: ``--project``, else the export's port."""
    return args.project or export["port"]


def build(args: argparse.Namespace) -> None:
    """Steps 1 to 7: load, predict, trace, emulate, write the project, co-simulate."""
    export = read_json(args.export_dir / "export.json")
    args.project = project_name(args, export)
    model = _timed("loading", load_model, args.export_dir, export["floatx"])
    words, logits, labels = load_reference(args.export_dir, args.outputs_npz)
    parts = split_inputs(model, words)
    comb, y_keras, y_alkaid = predict(args, model, parts, labels)
    stats = emulation_stats(y_keras, y_alkaid, logits, labels)
    check = record_emulation(args, export, stats)
    generate(args, comb, flatten(parts), y_alkaid, check)


def _bits(qint: Any) -> int:
    """The bits of a fixed-point interval (min, max, step), its sign bit included."""
    low, high, step = qint
    if step == 0 or low == high == 0:
        return 0
    return math.ceil(math.log2(max(-low, high + step) / step)) + int(low < 0)


def dsp_products(comb: Any) -> tuple[int, float]:
    """The products of two variables (opcode 7) that Vivado maps to DSPs, and their
    alkaid cost."""
    ops = comb.ops
    wide = [
        op
        for op in ops
        if op.opcode == 7 and min(_bits(ops[i].qint) for i in op.addr) > DSP_MIN_BITS
    ]
    return len(wide), sum(op.cost for op in wide)


def _pipelined(args: argparse.Namespace, model: Any, method0: str, hard_dc: int):
    """Trace under one setting and pipeline at --latency-cutoff, writing nothing."""
    import tempfile

    from alkaid.codegen import RTLModel

    comb = trace_comb(model, {"hard_dc": hard_dc, "method0": method0}, args.ternary)
    kwargs = {name: getattr(args, key) for key, name in RTL_OPTIONS.items()}
    with tempfile.TemporaryDirectory() as unused:
        rtl = RTLModel(
            comb, unused, args.project, ternary_fuse=args.ternary == "fused", **kwargs
        )
    return rtl.latency, rtl._comb


def scan(args: argparse.Namespace, model: Any) -> tuple[dict[str, Any], Any]:
    """The setting of fewest pipeline stages, of equal stages the shortest delay, as the
    implementations were chosen from 2026-10-12, and its pipelined logic."""
    results = []
    for method0, hard_dc in itertools.product(METHODS, SCAN_HARD_DC):
        cycles, comb = _pipelined(args, model, method0, hard_dc)
        setting = {"method0": method0, "hard_dc": hard_dc, "cycles": cycles}
        setting |= {"delay_ns": comb.latency[1], "cost": comb.cost}
        results.append((setting, comb))
    return min(results, key=lambda result: (result[0]["cycles"], result[0]["delay_ns"]))


def estimate(args: argparse.Namespace) -> None:
    """Write ``<out>/estimate.json``: the scan's setting, its cycles and nominal latency
    at --clock, and the LUTs and DSPs Vivado would use, without running Vivado."""
    export = read_json(args.export_dir / "export.json")
    args.project = project_name(args, export)
    model = _timed("loading", load_model, args.export_dir, export["floatx"])
    setting, comb = _timed("alkaid scan", scan, args, model)
    n_dsp, dsp_cost = dsp_products(comb)
    hardware = {"latency_ns": setting["cycles"] * args.clock, "dsp": n_dsp}
    hardware["lut"] = round(LUTS_PER_COST * (setting["cost"] - dsp_cost))
    write_json(
        args.out / "estimate.json", setting | hardware | {"ebops": export["ebops"]}
    )


def run_vivado(vivado: str, prj: Path) -> None:
    """Implement the project out of context; Vivado's own log is ``prj/vivado.log``."""
    subprocess.run(
        [vivado, *VIVADO_ARGS], cwd=prj, check=True, stdout=subprocess.DEVNULL
    )


def implement(args: argparse.Namespace) -> None:
    """Step 8, or the command that runs it elsewhere."""
    prj = args.out / "prj"
    if args.no_vivado:
        LOG.info(
            "no Vivado run; elsewhere: cd %s && vivado %s", prj, " ".join(VIVADO_ARGS)
        )
    else:
        _timed("Vivado implementation", run_vivado, args.vivado, prj)


def project_report(prj: Path) -> dict[str, Any]:
    """Read the project as ``alkaid report prj -o prj/report.json`` does, and write that
    file."""
    from alkaid._cli.report import load_project

    report = load_project(prj)
    if report is None:
        raise RuntimeError(f"alkaid cannot read the project {prj}")
    write_json(prj / "report.json", report)
    return report


def _number(report: dict[str, Any], key: str) -> float:
    """A numeric report entry, NaN when absent or unparsed."""
    value = report.get(key)
    return value if isinstance(value, int | float) else math.nan


def vivado_timing(report: dict[str, Any]) -> dict[str, float]:
    """The post-route WNS and the Fmax and latency it implies, derived as ``alkaid
    report`` derives them: the achieved period is the clock period minus the WNS."""
    wns_ns = _number(report, "WNS(ns)")
    period_ns = _number(report, "clock_period") - wns_ns
    latency_ns = _number(report, "latency") * period_ns
    return {"wns_ns": wns_ns, "fmax_mhz": 1000.0 / period_ns, "latency_ns": latency_ns}


def csv_row(check: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Merge ``check.json`` and alkaid's project report into one row of ``CSV_FIELDS``,
    NaN wherever a step has not run.

    The row holds no working point: those are read off ``keras_outputs.npz`` and
    ``alkaid_outputs.npz`` beside ``check.json``.
    """
    row = {column: _number(report, key) for column, key in REPORT_FIELDS.items()}
    row.update(vivado_timing(report), **check)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    row.update(vivado_done=report.keys() >= VIVADO_DONE, timestamp=now)
    return {key: math.nan if row.get(key) is None else row[key] for key in CSV_FIELDS}


def check_header(path: Path) -> None:
    """Refuse to append to a CSV whose header is not ``CSV_FIELDS``, which a row would
    misalign: one written before a column was added."""
    if not path.exists():
        return
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(CSV_FIELDS):
        raise SystemExit(f"{path} has another header than CSV_FIELDS; pass a new --csv")


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append the row, writing the header first into a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_row(args: argparse.Namespace) -> None:
    """Step 9: the project report and ``check.json`` to one CSV row."""
    report = _timed("alkaid report", project_report, args.out / "prj")
    row = csv_row(read_json(args.out / "check.json"), report)
    append_row(args.csv, row)
    LOG.info("row appended to %s: %s", args.csv, json.dumps(row))


def main(argv: list[str] | None = None) -> int:
    """Run the steps the command line asks for; return the exit status."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.estimate_only:
        estimate(args)
        return 0
    check_header(args.csv)
    if not args.report_only:
        build(args)
        implement(args)
    write_row(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
