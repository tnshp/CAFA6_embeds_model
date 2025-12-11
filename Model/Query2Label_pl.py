import torch
import torch.nn as nn
import pytorch_lightning as pl
from Utils.AsymetricLoss import AsymmetricLossOptimized, AsymmetricLoss
from Model.Query2Label import Query2Label
import numpy as np
import math


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
        # Loss selection: 'ASL' or 'BCE'
        loss_function: str = 'ASL',
        # Asymmetric loss parameters
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        loss_eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
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

    
        # Instantiate chosen loss function
        lf = (loss_function or 'ASL').upper()
        if lf in ('ASL', 'ASYMMETRIC', 'ASYMMETRIC_LOSS'):
            # prefer optimized implementation
            try:
                self.criterion = AsymmetricLossOptimized(
                    gamma_neg=gamma_neg,
                    gamma_pos=gamma_pos,
                    clip=clip,
                    eps=loss_eps,
                    disable_torch_grad_focal_loss=disable_torch_grad_focal_loss,
                )
            except Exception:
                self.criterion = AsymmetricLoss(
                    gamma_neg=gamma_neg,
                    gamma_pos=gamma_pos,
                    clip=clip,
                    eps=loss_eps,
                    disable_torch_grad_focal_loss=disable_torch_grad_focal_loss,
                )
        elif lf in ('BCE', 'BCEWITHLOGITS', 'BCEWITHLOGITSLOSS'):
            # Use BCEWithLogitsLoss; set reduction='sum' to match ASL sum-based scale
            self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            raise ValueError(f"Unsupported loss_function: {loss_function}. Supported: 'ASL', 'BCE'.")

        # Store validation step outputs for on_validation_epoch_end
        self.validation_step_outputs = []
        # Test step outputs
        self.test_step_outputs = []
        # For F-max computation (store probabilities and targets across val epoch)
        self._val_probs = []
        self._val_targets = []

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
        """Validation step computes loss and stores for F-max computation."""
        x = batch['tokens']
        y = batch['label'].float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Log per-step validation loss (will be reduced automatically)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Store for epoch end aggregation
        self.validation_step_outputs.append({'val_loss': loss.detach()})
        # store probabilities and targets for F-max computation (move to CPU to save GPU memory)
        probs = torch.sigmoid(logits)
        targets = (y > 0.5).int()
        try:
            self._val_probs.append(probs.detach().cpu())
            self._val_targets.append(targets.detach().cpu())
        except Exception:
            pass

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch. Compute F-max only."""
        # compute mean validation loss for the epoch
        if not self.validation_step_outputs:
            return
        losses = torch.stack([o['val_loss'] for o in self.validation_step_outputs])
        mean_loss = losses.mean()

        self.log('val_loss_epoch', mean_loss, prog_bar=True, logger=True)

        # Compute F-max (best F1 over thresholds) across validation epoch
        try:
            if len(self._val_probs) > 0:
                probs_all = torch.cat(self._val_probs, dim=0)  # (N, C)
                targets_all = torch.cat(self._val_targets, dim=0)  # (N, C)
                # thresholds from 0.0 to 1.0 (inclusive)
                thresholds = torch.linspace(0.0, 1.0, steps=101)
                C = targets_all.shape[1]
                per_class_f1_max = torch.zeros(C)
                eps = 1e-8
                for t in thresholds:
                    preds_t = (probs_all > t).int()
                    tp = (preds_t & targets_all).sum(dim=0).float()
                    fp = (preds_t & (1 - targets_all)).sum(dim=0).float()
                    fn = ((1 - preds_t) & targets_all).sum(dim=0).float()
                    precision = tp / (tp + fp + eps)
                    recall = tp / (tp + fn + eps)
                    f1 = 2 * precision * recall / (precision + recall + eps)
                    # replace NaNs with zeros
                    f1 = torch.nan_to_num(f1, nan=0.0)
                    per_class_f1_max = torch.maximum(per_class_f1_max, f1)

                val_fmax_macro = float(per_class_f1_max.mean().item())
                self.log('val_fmax_macro', val_fmax_macro, prog_bar=True, logger=True)
                # also log per-class fmax if desired (may be many classes)
                per_class_dict = {f'val_fmax_class_{i}': float(per_class_f1_max[i].item()) for i in range(per_class_f1_max.numel())}
                self.log_dict(per_class_dict, prog_bar=False, logger=True)
                # Print macro F-max to stdout (only on main process)
                try:
                    rank = getattr(self, 'global_rank', 0)
                except Exception:
                    rank = 0
                if rank == 0:
                    print(f"Validation Macro F-max: {val_fmax_macro:.4f}")
                try:
                    # clear stored buffers
                    self._val_probs.clear()
                    self._val_targets.clear()
                except Exception:
                    pass
        except Exception:
            # if anything goes wrong computing F-max, skip it
            pass
        
        # Clear stored outputs for next epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step computes loss and logs it."""
        x = batch['tokens']
        y = batch['label'].float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Log per-step test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append({'test_loss': loss.detach()})

    def on_test_epoch_end(self) -> None:
        """Aggregate test metrics at epoch end and log them."""
        if not self.test_step_outputs:
            return
        losses = torch.stack([o['test_loss'] for o in self.test_step_outputs])
        mean_loss = losses.mean()

        self.log('test_loss_epoch', mean_loss, prog_bar=True, logger=True)
        print(f"Test Loss: {mean_loss:.4f}")

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