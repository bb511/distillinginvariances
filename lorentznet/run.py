import json
from pathlib import Path
from types import SimpleNamespace

from . import distillation_pipeline, lorentz_shuffle_pipeline
from .distill.training import atomic_json
from .folds_1_to_4_pipeline import FollowupPipeline, load_fold0_parameters
from .prepare_data import prepare_data
from .teacher.train import train_teacher


def train_canonical_teachers(args, config):
    teacher_root = args.teacher_root or args.output_root / "teachers"
    teacher_root.mkdir(parents=True, exist_ok=True)
    shared = {
        "split": "cv5", "fold": 0, "fold_seed": 0, "seed": 325,
        "no_beams": False, "normalization": "batch", "bn_calibration_samples": 8192,
        "num_workers": 0, "target_accuracy": 1.0, "selection_window": config["selection_window"],
        "min_lr_ratio": 0.05, "grad_clip": 1.0, "device": args.device,
    }
    def run_trial(trial, output, epochs, fold, train_limit=None, val_limit=None):
        values = {**shared, **{key: value for key, value in trial.items() if key != "name"},
                  "data_dir": args.data_dir, "output_dir": output, "epochs": epochs,
                  "fold": fold, "min_epochs": epochs, "patience": epochs,
                  "selection_window": min(config["selection_window"], epochs),
                  "target_hold_epochs": epochs + 1, "max_train_samples": train_limit,
                  "max_val_samples": val_limit}
        if not (output / "result.json").is_file() or not (output / "best.pt").is_file():
            train_teacher(SimpleNamespace(**values))
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
        tail = records[-values["selection_window"]:]
        return {"trial": trial, "mean_val_accuracy": sum(record["val_accuracy"] for record in tail) / len(tail),
                "mean_val_loss": sum(record["val_loss"] for record in tail) / len(tail)}
    existing = all(distillation_pipeline.Pipeline.teacher_checkpoint_path(teacher_root, fold).is_file() for fold in args.folds)
    if existing:
        return teacher_root
    ranked = [run_trial(trial, teacher_root / "screen" / trial["name"], config["search_epochs"], 0,
                        config["search_train_samples"], config["search_val_samples"]) for trial in config["trials"]]
    ranked.sort(key=lambda item: (item["mean_val_accuracy"], -item["mean_val_loss"]), reverse=True)
    atomic_json(teacher_root / "parameter_ranking.json", ranked)
    selected = ranked[0]["trial"]
    atomic_json(teacher_root / "best_trial.json", selected)
    for fold in args.folds:
        output = distillation_pipeline.Pipeline.teacher_checkpoint_path(teacher_root, fold).parent
        run_trial(selected, output, config["final_epochs"], fold)
    return teacher_root


def main(data_root, output_root, regime="both", device="auto", folds=(0, 1, 2, 3, 4), config=None, teacher_config=None, teacher_root=None):
    folds = sorted(set(folds))
    if 0 not in folds or any(fold < 0 or fold > 4 for fold in folds):
        raise ValueError("folds must include selection fold 0 and use fold numbers 0 through 4")
    if regime == "both":
        for selected in ("canonical", "transformed"):
            main(data_root, Path(output_root) / selected, regime=selected, device=device,
                 folds=folds, config=config[selected] if config is not None else None,
                 teacher_config=teacher_config, teacher_root=teacher_root)
        return
    if regime not in ("canonical", "transformed"):
        raise ValueError("regime must be canonical, transformed, or both")
    args = SimpleNamespace(data_root=Path(data_root), output_root=Path(output_root).resolve(),
                           regime=regime, device=str(lorentz_shuffle_pipeline.choose_device(device)),
                           folds=folds, teacher_root=Path(teacher_root) if teacher_root is not None else None)
    args.output_root.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    if not isinstance(config, dict):
        config_path = Path(config) if config is not None else package / ("config_distillation.json" if regime == "canonical" else "config_lorentz_shuffle.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    args.data_dir = prepare_data(args.data_root, args.output_root / "data" / "canonical", config.get("data_limit"), require_features=args.regime == "canonical")
    if args.regime == "canonical":
        if not isinstance(teacher_config, dict):
            teacher_config_path = Path(teacher_config) if teacher_config is not None else package / "config_teacher.json"
            teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
        args.teacher_root = train_canonical_teachers(args, teacher_config)
        distillation_pipeline.Pipeline(args, config).run()
    else:
        args.source_data_dir = args.data_dir
        args.shuffled_data_dir = args.output_root / "data" / "transformed"
        args.output_root = args.output_root / "fold0"
        lorentz_shuffle_pipeline.Pipeline(args, config).run()
        if len(args.folds) > 1:
            args.fold0_output_root = args.output_root
            args.output_root = args.output_root.parent / "remaining_folds"
            promoted = load_fold0_parameters(args.fold0_output_root, config)
            FollowupPipeline(args, config, promoted).run()
