from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class DeepSetsModule(LightningModule):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, compile: bool) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=5)
        self.val_acc = Accuracy(task='multiclass', num_classes=5)
        self.test_acc = Accuracy(task='multiclass', num_classes=5)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        return (loss, preds, targets)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == 'fit':
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss', 'interval': 'epoch', 'frequency': 1}}
        return {'optimizer': optimizer}
from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class MLPModule(LightningModule):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, compile: bool, l1_lambda: float=0.0) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.l1_lambda = l1_lambda
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=5)
        self.val_acc = Accuracy(task='multiclass', num_classes=5)
        self.test_acc = Accuracy(task='multiclass', num_classes=5)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        if self.l1_lambda > 0:
            l1 = sum((p.abs().sum() for p in self.net.parameters()))
            loss = loss + self.l1_lambda * l1
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        return (loss, preds, targets)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == 'fit':
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss', 'interval': 'epoch', 'frequency': 1}}
        return {'optimizer': optimizer}
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class FitNetsModule(LightningModule):

    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, teacher_ckpt: str, stage: str='hint', temperature: float=3.5, alpha: float=0.0, beta: float=0.0, guided_idx: int=3, hint_depth: int=3, freeze_guided: bool=False, student_ckpt: Optional[str]=None, compile: bool=False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['teacher', 'student'])
        self.teacher = teacher
        self.student = student
        ckpt = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        teacher_weights = {k[4:]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
        self.teacher.load_state_dict(teacher_weights)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        all_layers = [student.input_dim] + student.layers + [student.output_dim]
        guided_dim = all_layers[(guided_idx + 2) // 2]
        hint_dim = teacher.phi_layers[-1]
        if guided_dim != hint_dim:
            self.regressor = nn.Linear(guided_dim, hint_dim)
        else:
            self.regressor = nn.Identity()
        if stage == 'kd' and student_ckpt is not None:
            ckpt = torch.load(student_ckpt, map_location='cpu', weights_only=False)
            student_weights = {k[8:]: v for k, v in ckpt['state_dict'].items() if k.startswith('student.')}
            self.student.load_state_dict(student_weights)
            reg_weights = {k[10:]: v for k, v in ckpt['state_dict'].items() if k.startswith('regressor.')}
            if reg_weights:
                self.regressor.load_state_dict(reg_weights)
        if stage == 'kd' and freeze_guided:
            for i, layer in enumerate(self.student.mlp):
                if i <= guided_idx:
                    for p in layer.parameters():
                        p.requires_grad_(False)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        if stage == 'kd':
            self.train_acc = Accuracy(task='multiclass', num_classes=5)
            self.val_acc = Accuracy(task='multiclass', num_classes=5)
            self.val_acc_best = MaxMetric()
            self.val_pi_agree = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        if self.hparams.stage == 'kd':
            self.val_acc.reset()
            self.val_acc_best.reset()
            self.val_pi_agree.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, y = batch
        if self.hparams.stage == 'hint':
            student_logits, guided = self.student.forward_with_guided(x, self.hparams.guided_idx)
            guided = self.regressor(guided)
            with torch.no_grad():
                _, hint = self.teacher.forward_with_phi_hint(x, self.hparams.hint_depth)
            loss = self.mse(guided, hint)
            return (loss, None, None)
        T = self.hparams.temperature
        beta = self.hparams.beta
        student_logits, guided = self.student.forward_with_guided(x, self.hparams.guided_idx)
        guided = self.regressor(guided)
        with torch.no_grad():
            teacher_logits, hint = self.teacher.forward_with_phi_hint(x, self.hparams.hint_depth)
        kd_loss = T ** 2 * self.kl(F.log_softmax(student_logits / T, dim=1), F.log_softmax(teacher_logits / T, dim=1))
        ce_loss = self.ce(student_logits, y)
        hint_loss = self.mse(guided, hint)
        kd_total = kd_loss + self.hparams.alpha * ce_loss
        if beta > 0 and hint_loss > 0:
            scale = kd_total.detach() / hint_loss.detach().clamp_min(1e-12)
            loss = (1 - beta) * kd_total + beta * scale * hint_loss
        else:
            loss = kd_total
        preds = torch.argmax(student_logits, dim=1)
        targets = torch.argmax(y, dim=1)
        return (loss, preds, targets)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.stage == 'kd':
            self.train_acc(preds, targets)
            self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.stage == 'kd':
            self.val_acc(preds, targets)
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            x, _ = batch
            with torch.no_grad():
                orig_pred = torch.argmax(self.student(x), dim=1)
                agree_sum = 0.0
                for _ in range(3):
                    perm = torch.randperm(x.size(1), device=x.device)
                    perm_pred = torch.argmax(self.student(x[:, perm]), dim=1)
                    agree_sum += (perm_pred == orig_pred).float().mean()
                agree = agree_sum / 3
            self.val_pi_agree(agree)
            self.log('val/pi_agree', self.val_pi_agree, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.stage == 'kd':
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == 'fit':
            self.student = torch.compile(self.student)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.freeze_guided and self.hparams.stage == 'kd':
            params = [p for p in self.student.parameters() if p.requires_grad] + list(self.regressor.parameters())
        else:
            params = list(self.student.parameters()) + list(self.regressor.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss', 'interval': 'epoch', 'frequency': 1}}
        return {'optimizer': optimizer}
