# Hyperparameter optimisation using optuna.

import os
import argparse
import copy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str)
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
import optuna
import torch.nn as nn
import torch_geometric
from torch.utils.data import DataLoader

import util


def main(config):
    device = args.device
    outdir = util.make_output_directory(".", "optuna_hyperoptim")

    study = optuna.create_study(
        study_name=config["study_name"],
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
        storage=f"sqlite:///{outdir}/{config['storage']}.db",
        load_if_exists=True,
    )
    study.optimize(Objective(device, config), n_trials=250, gc_after_trial=True)
    print("Best hyperparameters:", study.best_trial.params)


class Objective:
    """Objective optuna function.

    Trains a model and then outputs the accuracy of the model, which is used as the
    objective value optuna tries to maximise through changing the hyperparameters
    of a given model. In this case, an MLP.

    Args:
        device: String specifying the device this optimisation process takes place on.
        config: Dictionary containing the hyperparameters that should be optimised and
            their ranges.
    """

    def __init__(self, device: str, config: dict):
        self.device = device
        self.config = config

    def __call__(self, trial):
        data_hyperparams = copy.deepcopy(self.config["data_hyperparams"])
        batch_size = trial.suggest_categorical(
            "batch", self.config["data_hyperparams"]["torch_dataloader"]["batch_size"]
        )
        data_hyperparams["torch_dataloader"].update({"batch_size": batch_size})

        train_data = util.import_data(self.device, data_hyperparams, train=True)
        util.print_data_deets(train_data, "Training")
        valid_data = util.import_data(self.device, data_hyperparams, train=False)
        util.print_data_deets(valid_data, "Validation")

        model_hyperparams = copy.deepcopy(self.config["model_hyperparams"])
        nlayers = copy.deepcopy(self.config["model_hyperparams"]["nlayers"])
        nlayers = trial.suggest_categorical("nlayers", nlayers)
        layers = [
            trial.suggest_categorical(
                f"layer_{idx}", self.config["model_hyperparams"]["nnodes"]
            )
            for idx in range(nlayers)
        ]
        activ = trial.suggest_categorical(
            "activ", self.config["model_hyperparams"]["activ"]
        )
        drate_lower = self.config["model_hyperparams"]["dropout_rate"][0]
        drate_upper = self.config["model_hyperparams"]["dropout_rate"][1]
        dropout_rate = trial.suggest_float("dropout_rate", drate_lower, drate_upper)
        del model_hyperparams["nlayers"]
        del model_hyperparams["nnodes"]

        model_hyperparams.update(
            {
                "layers": layers,
                "dropout_rate": dropout_rate,
                "activ": activ,
            }
        )

        model = util.get_model(self.config["model_type"], model_hyperparams)
        util.profile_model(model, train_data)

        training_hyperparams = copy.deepcopy(self.config["training_hyperparams"])
        lr_lower_limit = self.config["training_hyperparams"]["lr"][0]
        lr_upper_limit = self.config["training_hyperparams"]["lr"][1]
        wg_lower_limit = self.config["training_hyperparams"]["weight_decay"][0]
        wg_upper_limit = self.config["training_hyperparams"]["weight_decay"][1]
        training_hyperparams.update(
            {
                "lr": trial.suggest_float("lr", lr_lower_limit, lr_upper_limit),
                "weight_decay": trial.suggest_loguniform(
                    "weight_decay", wg_lower_limit, wg_upper_limit
                ),
            }
        )

        valid_accu = train(
            model, train_data, valid_data, self.device, training_hyperparams
        )

        return valid_accu


def train(
    model: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    device: str,
    training_hyperparams: dict,
):
    """Trains a given model on given data for a number of epochs. Same as in train.py."""
    model = model.to(device)
    epochs = training_hyperparams["epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_hyperparams["lr"],
        weight_decay=training_hyperparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(400, epochs, 400)), gamma=0.1
    )
    print(util.tcols.OKGREEN + "Optimizer summary: " + util.tcols.ENDC)
    print(optimizer)

    loss_function = torch.nn.CrossEntropyLoss().to(device)

    print(util.tcols.OKCYAN + "\n\nTraining model..." + util.tcols.ENDC)
    epochs_no_improve = 0
    epochs_es_limit = training_hyperparams["early_stopping"]
    best_accu = 0

    for epoch in range(epochs):
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        model.train()
        for data in train_data:
            data = data.to(device)
            y_pred = model(data.pos)
            y_true = data.y.flatten()

            loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_sum += loss
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)
            totnum_batches += 1

        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        scheduler.step()

        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        for data in valid_data:
            data = data.to(device)
            y_true = data.y.flatten()
            y_pred = model.predict(data.pos)

            loss = loss_function(y_pred, y_true)
            accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)
            batch_loss_sum += loss
            batch_accu_sum += accu
            totnum_batches += 1

        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches

        if valid_accu <= best_accu:
            epochs_no_improve += 1
        else:
            best_accu = valid_accu
            epochs_no_improve = 0

        if early_stopping(epochs_no_improve, epochs_es_limit):
            break

        print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu)

    return best_accu


def print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu):
    """Prints the training and validation metrics in a nice format."""
    print(
        util.tcols.OKGREEN
        + f"Epoch : {epoch + 1}/{epochs}\n"
        + util.tcols.ENDC
        + f"Train loss (average) = {train_loss.item():.8f}\n"
        f"Train accuracy  = {train_accu:.8f}\n"
        f"Valid loss = {valid_loss.item():.8f}\n"
        f"Valid accuracy = {valid_accu:.8f}\n"
    )


def early_stopping(epochs_no_improve: int, epochs_limit: int) -> bool:
    """Stops the training if there has been no improvement in the loss."""
    if epochs_no_improve >= epochs_limit:
        return 1
    return 0


if __name__ == "__main__":
    # Load config
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
