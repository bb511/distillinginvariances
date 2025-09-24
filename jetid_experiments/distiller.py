# Knowledge distillation class from a general teacher model to a student.
# This is configured to use pytorch geometric loader of the ModelNet40 data.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepspeed

import util


class Distiller(nn.Module):
    """Performs the knowledge distillation between a given teacher and a student.

    For more details on this process, see
        Hinton et. al. 2015 - Distilling Knowledge in a Neural Network
        Stanton et. al. 2021 - Does Knowledge Distillation Really Work?

    Args:
        student: Student pytorch neural network module.
        teacher: Teacher pytorch neural network module, pre-trained on the data.
        device: String that specifies the pytorch device on which to cast the data and
            the networks.
        lr: The learning rate to use in the distillation process.
        lr_patience: Number of epochs after which, if there is no improvement in the
            accuracy of the student, the lr decreases by a factor of 0.1.
        epochs: Total epochs to distill for.
        early_stopping: Number of epochs after which, if there is no improvement in
            the accuracy of the student, the distillation stops.
        temp: The temperature in the distillation process, inserted in the softmax.
        alpha: Factor that balances between the loss of the student on the teacher
            output and the hard labels. In our experiments, we always set this to 0,
            i.e., the distillation process is solely based on the teacher output.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str,
        lr: float = 0.001,
        lr_patience: int = 50,
        epochs: int = 100,
        early_stopping: int = 20,
        temp: float = 3.5,
        alpha: float = 0,
    ):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="max",
            patience=lr_patience,
            factor=0.1,
            threshold=1e-3,
            verbose=True,
        )

        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.device = device

        self.all_student_loss_train = []
        self.all_distill_loss_train = []
        self.all_total_loss_train = []
        self.all_student_accu_train = []

        self.all_student_loss_valid = []
        self.all_distill_loss_valid = []
        self.all_total_loss_valid = []
        self.all_student_accu_valid = []

        self.alpha = alpha
        self.temp = temp
        self.epochs = epochs
        self.early_stopping = early_stopping

    def distill(self, train_data: DataLoader, valid_data: DataLoader):
        """Distill a teacher into a student model."""
        best_accu = 0
        epochs_no_improve = 0
        self.student.to(self.device)
        self.teacher.to(self.device)

        print(util.tcols.OKGREEN + "\nDistilling..." + util.tcols.ENDC)
        for epoch in range(self.epochs):
            self.student.train()

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for x, y in train_data:
                x = x.to(self.device)
                y = y.to(self.device)

                y_true = y

                teacher_predictions = self.teacher.predict(x)
                student_predictions = self.student(x)

                # Compute but the cross-entropy between the student predictions and
                # the truth labels. If alpha = 0, this is not used in the training.
                student_loss = self.student_loss_fn(student_predictions, y_true)

                # Compute, by default, the kl divergence between the teacher and the
                # student outputs that are passed through softmax with temperature.
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(
                            teacher_predictions / self.temp, dim=1
                        ),
                        nn.functional.log_softmax(
                            student_predictions / self.temp, dim=1
                        ),
                    )
                    * self.temp**2
                )

                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )

                self.optimiser.zero_grad()
                total_loss.mean().backward()
                self.optimiser.step()
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y_true.max(dim=1)[1]
                ) / len(y_true)

                student_loss_running.append(student_loss.cpu().item())
                distill_loss_running.append(distillation_loss.cpu().item())
                student_accu_running.append(student_accu.cpu().item())

            self.all_student_loss_train.append(np.mean(student_loss_running))
            self.all_distill_loss_train.append(np.mean(distill_loss_running))
            self.all_student_accu_train.append(np.mean(student_accu_running))

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for x, y in valid_data:
                x = x.to(self.device)
                y = y.to(self.device)

                y_true = y

                teacher_predictions = self.teacher.predict(x)
                student_predictions = self.student.predict(x)
                student_loss = self.student_loss_fn(student_predictions, y_true)
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(
                            teacher_predictions / self.temp, dim=1
                        ),
                        nn.functional.log_softmax(
                            student_predictions / self.temp, dim=1
                        ),
                    )
                    * self.temp**2
                )
                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y_true.max(dim=1)[1]
                ) / len(y_true)

                student_loss_running.append(student_loss.cpu().item())
                distill_loss_running.append(distillation_loss.cpu().item())
                student_accu_running.append(student_accu.cpu().item())

            self.all_student_loss_valid.append(np.mean(student_loss_running))
            self.all_distill_loss_valid.append(np.mean(distill_loss_running))
            self.all_student_accu_valid.append(np.mean(student_accu_running))

            self.scheduler.step(best_accu)
            if self.all_student_accu_valid[-1] <= best_accu:
                epochs_no_improve += 1
            else:
                best_accu = self.all_student_accu_valid[-1]
                epochs_no_improve = 0

            if epochs_no_improve == self.early_stopping:
                break

            self.print_metrics(epoch)

        return {
            "student_train_losses": self.all_student_loss_train,
            "student_train_accurs": self.all_student_accu_train,
            "distill_train_losses": self.all_distill_loss_train,
            "student_valid_losses": self.all_student_loss_valid,
            "student_valid_accurs": self.all_student_accu_valid,
            "distill_valid_losses": self.all_distill_loss_valid,
        }

    def get_student(self):
        return self.student

    def print_metrics(self, epoch: int):
        """Prints the training and validation metrics in a nice format."""
        print(
            util.tcols.OKGREEN
            + f"Epoch : {epoch + 1}/{self.epochs}\n"
            + util.tcols.ENDC
            + f"Student train loss = {self.all_student_loss_train[epoch]:.8f}\n"
            + f"Distill train loss = {self.all_distill_loss_train[epoch]:.8f}\n"
            + f"Student train accu = {self.all_student_accu_train[epoch]:.8f}\n\n"
            + f"Student valid loss = {self.all_student_loss_valid[epoch]:.8f}\n"
            + f"Distill valid loss = {self.all_distill_loss_valid[epoch]:.8f}\n"
            + f"Student valid accu = {self.all_student_accu_valid[epoch]:.8f}\n"
            + "---"
        )
