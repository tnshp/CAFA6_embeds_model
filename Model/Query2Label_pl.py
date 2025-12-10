import torch
import torch.nn as nn
import pytorch_lightning as pl
from Utils.AsymetricLoss import AsymmetricLossOptimized, AsymmetricLoss
from Model.Query2Label import Query2Label
import numpy as np
import math
try:
    from torchmetrics.classification import MultilabelF1Score
except Exception:  # fallback import path for different torchmetrics versions
    from torchmetrics.classification.multilabel import MultilabelF1Score


class Query2Label_pl(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        nheads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Query2Label(
            num_classes=num_classes,
            in_dim=in_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        self.criterion = AsymmetricLoss()

        # Torchmetrics: per-class F1 (we will compute per-class and average in on_validation_epoch_end)
        self.val_f1_metric = MultilabelF1Score(num_labels=num_classes, average=None, threshold=0.5)
        
        # Store validation step outputs for on_validation_epoch_end
        self.validation_step_outputs = []
        # Test metrics/state
        self.test_f1_metric = MultilabelF1Score(num_labels=num_classes, average=None, threshold=0.5)
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Standard training step.

        Expects `batch` to be a dict with keys `'tokens'` and `'label'` where
        - `tokens` is a float Tensor of shape (B, L, C_in)
        - `label` is a binary Tensor of shape (B, num_classes)
        """
        x = batch['tokens']
        y = batch['label'].float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step computes loss and logs it.

        Stores loss in instance attribute for on_validation_epoch_end.
        """
        x = batch['tokens']
        y = batch['label'].float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Log per-step validation loss (will be reduced automatically)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Update torchmetrics multilabel F1 metric (stores per-class counts)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        targets = (y > 0.5).int()
        # update stateful metric
        try:
            self.val_f1_metric.update(preds, targets)
        except Exception:
            # some torchmetrics versions accept (preds, targets) ordering; catch and try swapped order
            try:
                self.val_f1_metric.update(targets, preds)
            except Exception:
                pass

        # Store for epoch end aggregation
        self.validation_step_outputs.append({'val_loss': loss.detach()})

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch. Aggregate metrics and reset."""
        # compute mean validation loss for the epoch
        if not self.validation_step_outputs:
            return
        losses = torch.stack([o['val_loss'] for o in self.validation_step_outputs])
        mean_loss = losses.mean()

        # compute per-class F1 using torchmetrics
        try:
            per_class_f1 = self.val_f1_metric.compute()  # Tensor shape (num_classes,)
        except Exception:
            # fallback: try calling as function
            try:
                per_class_f1 = self.val_f1_metric(preds=None, target=None)
            except Exception:
                per_class_f1 = None

        self.log('val_loss_epoch', mean_loss, prog_bar=True, logger=True)

        if per_class_f1 is not None:
            # ensure CPU tensor
            per_class_f1 = per_class_f1.detach().cpu()
            # macro = average across classes
            macro_f1 = float(per_class_f1.mean().item())
            self.log('val_f1_macro', macro_f1, prog_bar=True, logger=True)

            # log per-class F1 individually (may be many classes)
            per_class_dict = {f'val_f1_class_{i}': float(per_class_f1[i].item()) for i in range(per_class_f1.numel())}
            # log without flooding progress bar
            self.log_dict(per_class_dict, prog_bar=False, logger=True)

            # Print macro F1 to stdout (only on main process)
            try:
                rank = getattr(self, 'global_rank', 0)
            except Exception:
                rank = 0
            if rank == 0:
                print(f"Validation Macro F1: {macro_f1:.4f}")

            # reset metric state for next epoch
            try:
                self.val_f1_metric.reset()
            except Exception:
                pass
        else:
            # if torchmetrics failed, fallback to noting missing metric
            self.log('val_f1_macro', 0.0, prog_bar=True, logger=True)
        
        # Clear stored outputs for next epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step mirrors validation_step but stores test outputs."""
        x = batch['tokens']
        y = batch['label'].float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Log per-step test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        targets = (y > 0.5).int()
        try:
            self.test_f1_metric.update(preds, targets)
        except Exception:
            try:
                self.test_f1_metric.update(targets, preds)
            except Exception:
                pass

        self.test_step_outputs.append({'test_loss': loss.detach()})

    def on_test_epoch_end(self) -> None:
        """Aggregate test metrics at epoch end and log/print them."""
        if not self.test_step_outputs:
            return
        losses = torch.stack([o['test_loss'] for o in self.test_step_outputs])
        mean_loss = losses.mean()

        # compute per-class F1
        try:
            per_class_f1 = self.test_f1_metric.compute()
        except Exception:
            try:
                per_class_f1 = self.test_f1_metric(preds=None, target=None)
            except Exception:
                per_class_f1 = None

        self.log('test_loss_epoch', mean_loss, prog_bar=True, logger=True)

        if per_class_f1 is not None:
            per_class_f1 = per_class_f1.detach().cpu()
            macro_f1 = float(per_class_f1.mean().item())
            self.log('test_f1_macro', macro_f1, prog_bar=True, logger=True)
            per_class_dict = {f'test_f1_class_{i}': float(per_class_f1[i].item()) for i in range(per_class_f1.numel())}
            self.log_dict(per_class_dict, prog_bar=False, logger=True)
            # print on main rank
            try:
                rank = getattr(self, 'global_rank', 0)
            except Exception:
                rank = 0
            if rank == 0:
                print(f"Test Macro F1: {macro_f1:.4f}")
            try:
                self.test_f1_metric.reset()
            except Exception:
                pass
        else:
            self.log('test_f1_macro', 0.0, prog_bar=True, logger=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # Use AdamW; hyperparameters were saved in self.hparams by save_hyperparameters()
        lr = self.hparams.get('lr', 1e-4) if isinstance(self.hparams, dict) else getattr(self.hparams, 'lr', 1e-4)
        weight_decay = self.hparams.get('weight_decay', 1e-5) if isinstance(self.hparams, dict) else getattr(self.hparams, 'weight_decay', 1e-5)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Optional scheduler with linear warmup + cosine decay. Requires `total_steps` and `warmup_steps` in hparams.
        total_steps = self.hparams.get('total_steps', None) if isinstance(self.hparams, dict) else getattr(self.hparams, 'total_steps', None)
        warmup_steps = self.hparams.get('warmup_steps', 0) if isinstance(self.hparams, dict) else getattr(self.hparams, 'warmup_steps', 0)

        if total_steps is None or total_steps <= 0:
            return optimizer

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # step per training batch
                'frequency': 1,
            },
        }