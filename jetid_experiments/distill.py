# Run the distillation framework.
# Train a student network through the knowledge distillation framework.

import os
import time
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str, default="default_cofig.yml")
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

from distiller import Distiller
import util


def main(config: dict):
    device = args.device
    outdir = util.make_output_directory("distilled_deepsets", config["outdir"])
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    # Get the configuration of the trained teacher network.
    config_teacher = util.load_config_file(
        os.path.join(config["teacher"], "config.yml")
    )

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory("distilled_mlps", config["outdir"])

    # Import the distillation data set. Here, the same as the data set the teacher
    # is trained on.
    train_data = util.import_data(device, config["data_hyperparams"], train=True)
    util.print_data_deets(train_data, "Training")
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    # Import the trained teacher network and its weights.
    print(util.tcols.OKGREEN + "Teacher network" + util.tcols.ENDC)
    teacher_model = util.get_model(
        config_teacher["model_type"], config_teacher["model_hyperparams"]
    )
    weights_file = os.path.join(
        config["teacher"], f"seed{config['teacher_seed']}", "model.pt"
    )
    teacher_model.load_state_dict(torch.load(weights_file))

    # Import the student network, ready for training through distillation.
    print(util.tcols.OKGREEN + "Student network" + util.tcols.ENDC)
    student_model = util.get_model(config["model_type"], config["model_hyperparams"])

    # Perform the distillation procedure.
    distill_hyperparams = config["distill_hyperparams"]
    distill = Distiller(student_model, teacher_model, device, **distill_hyperparams)
    hist = distill.distill(train_data, valid_data)

    # Plot the evolution of the distillation process metrics through the epochs.
    util.loss_plot(hist["student_train_losses"], hist["student_valid_losses"], outdir)
    util.accu_plot(hist["student_train_accurs"], hist["student_valid_accurs"], outdir)
    util.loss_plot(
        hist["distill_train_losses"],
        hist["distill_valid_losses"],
        outdir,
        "distillation",
    )
    model_file = os.path.join(outdir, "model.pt")

    # Save the student model.
    torch.save(distill.get_student().state_dict(), model_file)


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
