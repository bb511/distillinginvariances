# Validation script for the deepsets network, computing all the metrics of interest.

import os
import argparse
import copy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--models_dir", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch

torch.manual_seed(args.seed)

import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

import util


def main(args: dict):
    device = args.device
    model_dirs = [x[0] for x in os.walk(args.models_dir)][1:]

    # Load the configuration file of a trained model or distilled student model.
    config = util.load_config_file(os.path.join(args.models_dir, "config.yml"))
    # Load the validation data corresponding to its training data.
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    # Import the trained model from the config.
    model = util.get_model(config["model_type"], config["model_hyperparams"])
    all_metrics = {"accu": [], "nlll": [], "ecel": [], "perm": []}

    # Set up additional metrics and import the teacher model if computing the validation
    # metrics of  a distilled student model.
    if "teacher" in config.keys():
        all_metrics.update({"top1_agreement": [], "teach_stu_kldiv": []})
        config_teacher = util.load_config_file(
            os.path.join(config["teacher"], "config.yml")
        )
        print(util.tcols.OKGREEN + "Teacher network" + util.tcols.ENDC)
        teacher_model = copy.deepcopy(model)
        # If the teacher does not have the same architecture as the student, i.e.,
        # self-distillation, import the actual architecture.
        if not config["model_type"] == config_teacher["model_type"]:
            if not config["model_hyperparams"] == config_teacher["model_hyperparams"]:
                teacher_model = util.get_model(
                    config_teacher["model_type"], config_teacher["model_hyperparams"]
                )

        # Import a secondary, unrelated teacher to compute the fidelity metrics of
        # the distilled student with.
        if "target_teacher" in config.keys():
            all_metrics.update({"top1_agreement'": [], "teach_stu_kldiv'": []})
            config_target_teacher = util.load_config_file(
                os.path.join(config["target_teacher"], "config.yml")
            )
            target_teacher_model = util.get_model(
                config_target_teacher["model_type"],
                config_target_teacher["model_hyperparams"],
            )
            weights_file = os.path.join(
                config["target_teacher"],
                "seed" + str(config["teacher_seed"]),
                "model.pt",
            )
            target_teacher_model.load_state_dict(torch.load(weights_file))

        weights_file = os.path.join(
            config["teacher"], "seed" + str(config["teacher_seed"]), "model.pt"
        )
        teacher_model.load_state_dict(torch.load(weights_file))

    # Run inference on the validation data and compute metrics for each of the models
    # in the given model dirs folder.
    for model_dir in model_dirs:
        print(util.tcols.HEADER + f"Model at: {model_dir}" + util.tcols.ENDC)
        weights_file = os.path.join(model_dir, "model.pt")
        metrics = validate(model, weights_file, valid_data, device)
        if "teacher" in config.keys():
            metrics.update(compute_fidelity(model, teacher_model, valid_data, device))
        if "target_teacher" in config.keys():
            metrics.update(
                compute_fidelity(model, target_teacher_model, valid_data, device, "'")
            )
        for metric, value in metrics.items():
            all_metrics[metric].append(value)

    # Compute the average and standard deviation for the metrics computed for each
    # model in the given models dir, i.e., for each seed.
    print(util.tcols.OKGREEN + "Average model metrics: " + util.tcols.ENDC)
    metrics_file_path = os.path.join(args.models_dir, "metrics_avg.log")
    with open(metrics_file_path, "a") as metrics_file:
        for metric, value in all_metrics.items():
            metric_mean = np.mean(value)
            metric_std = np.std(value)
            print(f"{metric}: {metric_mean:.3e} ± {metric_std:.3e}")
            metrics_file.write(f"{metric_mean:.3e} ± {metric_std:.3e}")

    print(util.tcols.OKCYAN + "Wrote average metrics to file in: " + util.tcols.ENDC)
    print(f"{metrics_file_path}")


def validate(model: nn.Module, weights_file: str, valid_data: DataLoader, device: str):
    """Run the model on the test data and save all relevant metrics to file."""
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    nll = nn.NLLLoss().to(device)
    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=5)

    batch_accu_sum = 0
    batch_nlll_sum = 0
    batch_ecel_sum = 0
    batch_perm_invl_sum = 0
    totnum_batches = 0
    for x, y in valid_data:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        y_true = y

        # Compute top-1 accuracy, negative log likelihood, and exp calibration error.
        accu = torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
        log_probs = nn.LogSoftmax(dim=1)(y_pred)
        nll_loss = nll(log_probs, torch.argmax(y_true, dim=1))
        ece_loss = ece(y_pred, torch.argmax(y_true, dim=1))

        # Compute how invariant the model is to permutations.
        perm_inv_loss = test_perm_inv(model, x)

        batch_accu_sum += accu
        batch_nlll_sum += nll_loss
        batch_ecel_sum += ece_loss
        batch_perm_invl_sum += perm_inv_loss
        totnum_batches += 1

    metrics = {
        "accu": (batch_accu_sum / totnum_batches).cpu().item(),
        "nlll": (batch_nlll_sum / totnum_batches).cpu().item(),
        "ecel": (batch_ecel_sum / totnum_batches).cpu().item(),
        "perm": (batch_perm_invl_sum / totnum_batches).cpu().item(),
    }
    print_metrics(metrics)

    return metrics


def compute_fidelity(
    student: nn.Module, teacher: nn.Module, valid_data: DataLoader, device: str, flag=""
):
    """Compute the student-teacher average top-1 agreement and the KL divergence."""
    kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
    top1_agreement_running = []
    kldiv_loss_running = []

    teacher.to(device)
    student.to(device)
    for x, y in valid_data:
        x = x.to(device)
        y = y.to(device)

        y_teacher = teacher.predict(x)
        y_student = student.predict(x)
        top1_agreement = torch.sum(
            y_student.max(dim=1)[1] == y_teacher.max(dim=1)[1]
        ) / len(y_student)
        kldiv_loss = kldiv(
            nn.functional.log_softmax(y_teacher, dim=1),
            nn.functional.log_softmax(y_student, dim=1),
        )
        top1_agreement_running.append(top1_agreement.cpu().item())
        kldiv_loss_running.append(kldiv_loss.cpu().item())

    valid_top1_agreement = np.mean(top1_agreement_running)
    valid_kldiv_loss = np.mean(kldiv_loss_running)

    return {
        f"top1_agreement{flag}": valid_top1_agreement,
        f"teach_stu_kldiv{flag}": valid_kldiv_loss,
    }


def invariance_measure(y_normal: torch.Tensor, y_transf: torch.Tensor):
    """Compute the difference of a model output given normal and symmetry trans data."""
    y_normal = nn.functional.softmax(y_normal, dim=1)
    y_transf = nn.functional.softmax(y_transf, dim=1)

    return torch.sum(torch.norm(y_normal - y_transf, dim=1))


def test_perm_inv(model: nn.Module, data: DataLoader):
    """Computes how invariant a model is with respect to a permutation of the data."""
    y_normal = model.predict(data)

    permutation = torch.randperm(data.size(1))
    data = data[:, permutation]

    y_transf = model.predict(data)

    permutation_reversal = torch.sort(permutation).indices
    data = data[:, permutation_reversal]

    inv_measure = invariance_measure(y_normal, y_transf)
    return inv_measure


def print_metrics(metrics: dict):
    """Prints the training and validation metrics in a nice format."""
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")
    print("")


if __name__ == "__main__":
    main(args)
