import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        epochs: Total epochs to distill for.
        early_stopping: Number of epochs after which, if there is no improvement in
            the accuracy of the student, the distillation stops.
        temp: The temperature in the distillation process, inserted in the softmax.
        alpha: Factor that balances between the loss of the student on the teacher
            output and the hard labels. In our experiments, we always set this to 0,
            i.e., the distillation process is solely based on the teacher output.
        load_student_from_path: load a pretrained student from this path
        is_teacher_mlp: makes sure the inputs are in the correct format for the teacher
    """
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str,
        temp: float,
        lr: float = 0.001,
        epochs: int = 10,
        early_stopping: int = 5,
        alpha: float = 0,
        load_student_from_path: str = None,
        is_teacher_mlp: bool = False
    ):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.is_teacher_mlp = is_teacher_mlp

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)

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

        if load_student_from_path is not None:
            state_dict = torch.load(load_student_from_path)
            student.load_state_dict(state_dict=state_dict)


    def distill(self, train_data: DataLoader, valid_data: DataLoader, save_path_folder = None):
        """Distill a teacher into a student model."""
        best_accu = 0
        epochs_no_improve = 0
        self.student.to(self.device)
        self.teacher.to(self.device)

        print("\nDistilling...")
        for epoch in range(self.epochs):
            self.student.train() #may break stuff

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for i, (x,y) in enumerate(train_data):
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    self.teacher.eval()
                    try:
                        teacher_predictions = self.teacher(x)
                    except RuntimeError:
                        teacher_predictions = self.teacher(x.view(-1,784))

                student_predictions = self.student(x.view(-1,784))
                student_loss = self.student_loss_fn(student_predictions, y)
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(teacher_predictions/self.temp, dim=1),
                        nn.functional.log_softmax(student_predictions/self.temp, dim=1),
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
                    student_predictions.max(dim=1)[1] == y
                ) / len(y)

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

            for x,y in valid_data:
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    self.student.eval()
                    self.teacher.eval()
                    try:
                        teacher_predictions = self.teacher(x)
                    except RuntimeError:
                        teacher_predictions = self.teacher(x.view(-1,784))
                    student_predictions = self.student(x.view(-1,784))

                student_loss = self.student_loss_fn(student_predictions, y)
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(teacher_predictions/self.temp, dim=1),
                        nn.functional.log_softmax(student_predictions/self.temp, dim=1),
                    )
                    * self.temp**2
                )
                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y
                ) / len(y)

                student_loss_running.append(student_loss.cpu().item())
                distill_loss_running.append(distillation_loss.cpu().item())
                student_accu_running.append(student_accu.cpu().item())

            self.all_student_loss_valid.append(np.mean(student_loss_running))
            self.all_distill_loss_valid.append(np.mean(distill_loss_running))
            self.all_student_accu_valid.append(np.mean(student_accu_running))

            if self.all_student_accu_valid[-1] <= best_accu:
                epochs_no_improve += 1
            else:
                best_accu = self.all_student_accu_valid[-1]
                epochs_no_improve = 0

            self.print_metrics(epoch)
        
        if save_path_folder is not None:
            save_path = save_path_folder + "distiller"
            torch.save(self.get_student().state_dict(), save_path)
            print(f"Student model saved as {save_path}!")
    
        return {
            "student_train_losses": self.all_student_loss_train,
            "student_train_accurs": self.all_student_accu_train,
            "distill_train_losses": self.all_distill_loss_train,
            "student_valid_losses": self.all_student_loss_valid,
            "student_valid_accurs": self.all_student_accu_valid,
            "distill_valid_losses": self.all_distill_loss_valid
        }            

    def get_student(self):
        return self.student
    
    def eval_student(self, valid_data):
        accu = []
        loss = []
        for i, (x,y) in enumerate(valid_data):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                self.student.eval()
                student_predictions = self.student(x.view(-1,784))
            student_loss = self.student_loss_fn(student_predictions, y)
            student_accu = torch.sum(
                student_predictions.max(dim=1)[1] == y
            ) / len(y)
            loss.append(student_loss.item())
            accu.append(student_accu.item())

        print(f"Loss : {torch.mean(student_loss)}\n"
            + f"Accuracy: {torch.mean(student_accu)}\n")


    def print_metrics(self, epoch: int):
        """Prints the training and validation metrics in a nice format."""
        print(
            f"Epoch : {epoch + 1}/{self.epochs}\n"
            + f"Student train loss = {self.all_student_loss_train[epoch]:.8f}\n"
            + f"Distill train loss = {self.all_distill_loss_train[epoch]:.8f}\n"
            + f"Student train accu = {self.all_student_accu_train[epoch]:.8f}\n\n"
            + f"Student valid loss = {self.all_student_loss_valid[epoch]:.8f}\n"
            + f"Distill valid loss = {self.all_distill_loss_valid[epoch]:.8f}\n"
            + f"Student valid accu = {self.all_student_accu_valid[epoch]:.8f}\n"
            + "---"
        )

    def get_temperature(self):
        return self.temp
   
    def compute_fidelity(self, test_loader):
       """Compute the student-teacher average top-1 agreement and the KL divergence."""
       kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
       top1_agreement_running = []
       kldiv_loss_running = []
       teacher = self.teacher
       student = self.student
       for x,_ in test_loader:
          x = x.to(self.device)
          if self.is_teacher_mlp:
            y_teacher = teacher(x.view(-1, 784))
            y_student = student(x.view(-1, 784))
          else:    
            y_teacher = teacher(x)
            y_student = student(x.view(-1, 784))              
          top1_agreement = torch.sum(
                y_student.max(dim=1)[1] == y_teacher.max(dim=1)[1]
            ) / len(y_student)
          kldiv_loss = kldiv(
                nn.functional.log_softmax(y_teacher, dim=1),
                nn.functional.log_softmax(y_student, dim=1)
            )
          top1_agreement_running.append(top1_agreement.cpu().item())
          kldiv_loss_running.append(kldiv_loss.cpu().item())

       valid_top1_agreement = np.mean(top1_agreement_running)
       valid_kldiv_loss = np.mean(kldiv_loss_running)

       return {
         "T1agree": valid_top1_agreement,
         "KLDiv": valid_kldiv_loss
       }
    

    