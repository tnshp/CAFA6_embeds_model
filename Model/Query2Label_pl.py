import torch
import torch.nn as nn
import pytorch_lightning as pl
from Utils.AsymetricLoss import AsymmetricLossOptimized, AsymmetricLoss
from Utils.RankLoss import RankLossPair
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
        elif lf in ('RANK', 'RANKLOSS', 'RANK_LOSS'):
            # RankLoss uses RankLossPair class
            self.criterion = RankLossPair(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_function: {loss_function}. Supported: 'ASL', 'BCE', 'RankLoss'.")
        
        # Store loss function name for training step
        self.loss_function = lf

        # Store validation step outputs for on_validation_epoch_end
        self.validation_step_outputs = []
        # Test step outputs
        self.test_step_outputs = []
        # For GO term-based F1 computation (store predictions per GO term)
        self._val_go_predictions = {}  # {go_term_id: list of probabilities}
        self._val_go_targets = {}      # {go_term_id: list of binary labels}

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.model(x, f) 

    def training_step(self, batch, batch_idx):
        """Standard training step.

        Expects `batch` to be a dict with keys `'tokens'` and `'label'` where
        - `tokens` is a float Tensor of shape (B, L, C_in)
        - `label` is a binary Tensor of shape (B, num_classes)
        """
        x = batch['go_embed']   # (B, L, C_in)
        f = batch['feature']   
        y = batch['label']

        logits = self.forward(x, f)
        
        # Calculate loss based on loss function type
        if self.loss_function in ('RANK', 'RANKLOSS', 'RANK_LOSS'):
            # RankLoss: compute loss for each class separately and sum
            losses = []
            for class_idx in range(logits.shape[1]):
                scores = logits[:, class_idx]  # (B,)
                labels = y[:, class_idx]  # (B,)
                
                # Only compute loss if we have both positive and negative samples
                if labels.sum() > 0 and labels.sum() < len(labels):
                    class_loss = self.criterion(scores, labels)
                    losses.append(class_loss)
            
            if losses:
                loss = torch.stack(losses).mean()
        else:
            # Standard loss functions (ASL, BCE)
            loss = self.criterion(logits, y)

        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step computes loss and stores for F-max computation."""
        x = batch['go_embed']   # (B, L, C_in)
        f = batch['feature']   
        y = batch['label']

        logits = self.forward(x, f)
        
        # Calculate loss based on loss function type
        if self.loss_function in ('RANK', 'RANKLOSS', 'RANK_LOSS'):
            # RankLoss: compute loss for each class separately and sum
            losses = []
            for class_idx in range(logits.shape[1]):
                scores = logits[:, class_idx]  # (B,)
                labels = y[:, class_idx]  # (B,)
                
                # Only compute loss if we have both positive and negative samples
                if labels.sum() > 0 and labels.sum() < len(labels):
                    class_loss = self.criterion(scores, labels)
                    losses.append(class_loss)
            
            if losses:
                loss = torch.stack(losses).mean()
            else:
                # Fallback to BCE if no valid pairs
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='mean')
        else:
            # Standard loss functions (ASL, BCE)
            loss = self.criterion(logits, y)

        # Log per-step validation loss (will be reduced automatically)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Store for epoch end aggregation
        self.validation_step_outputs.append({'val_loss': loss.detach()})
        
        # Store GO term-specific predictions for macro/micro F1
        probs = torch.sigmoid(logits)
        targets = (y > 0.5).int()
        try:
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            predicted_terms = batch['predicted_terms']  # List of lists of GO term IDs
            
            # For each sample in the batch
            for sample_idx in range(len(predicted_terms)):
                sample_terms = predicted_terms[sample_idx]
                sample_probs = probs_np[sample_idx]  # Shape: (num_terms_for_sample,)
                sample_targets = targets_np[sample_idx]  # Shape: (num_terms_for_sample,)
                
                # For each GO term in this sample
                for term_idx, go_term in enumerate(sample_terms):
                    if go_term not in self._val_go_predictions:
                        self._val_go_predictions[go_term] = []
                        self._val_go_targets[go_term] = []
                    
                    self._val_go_predictions[go_term].append(float(sample_probs[term_idx]))
                    self._val_go_targets[go_term].append(int(sample_targets[term_idx]))
        except Exception as e:
            print(f"Warning: Could not store GO term predictions: {e}")
            pass

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch. Compute F-max only."""
        # compute mean validation loss for the epoch
        if not self.validation_step_outputs:
            return
        losses = torch.stack([o['val_loss'] for o in self.validation_step_outputs])
        mean_loss = losses.mean()

        self.log('val_loss_epoch', mean_loss, prog_bar=True, logger=True)
        
        # Compute GO term-based macro and micro F1 scores
        try:
            if len(self._val_go_predictions) > 0:
                macro_f1, micro_f1 = self._compute_go_f1_scores()
                self.log('val_f1_macro_go', macro_f1, prog_bar=True, logger=True)
                self.log('val_f1_micro_go', micro_f1, prog_bar=True, logger=True)
                
                try:
                    rank = getattr(self, 'global_rank', 0)
                except Exception:
                    rank = 0
                if rank == 0:
                    print(f"Validation GO-based Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
                
                # Clear GO term buffers
                self._val_go_predictions.clear()
                self._val_go_targets.clear()
        except Exception as e:
            print(f"Warning: Could not compute GO-based F1 scores: {e}")
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

    def _compute_go_f1_scores(self, threshold=0.5):
        """Compute macro and micro F1 scores based on GO term predictions.
        
        For each GO term, we have multiple predictions across different samples.
        Macro F1: Average F1 score across all GO terms
        Micro F1: F1 score computed from global TP, FP, FN counts
        """
        eps = 1e-8
        per_term_f1_scores = []
        
        # Global counts for micro F1
        global_tp = 0
        global_fp = 0
        global_fn = 0
        
        # Compute F1 for each GO term
        for go_term, predictions in self._val_go_predictions.items():
            targets = self._val_go_targets[go_term]
            
            # Convert to numpy for easier computation
            preds_array = np.array(predictions)
            targets_array = np.array(targets)
            
            # Binarize predictions
            preds_binary = (preds_array >= threshold).astype(int)
            
            # Compute TP, FP, FN for this GO term
            tp = np.sum((preds_binary == 1) & (targets_array == 1))
            fp = np.sum((preds_binary == 1) & (targets_array == 0))
            fn = np.sum((preds_binary == 0) & (targets_array == 1))
            
            # Accumulate for micro F1
            global_tp += tp
            global_fp += fp
            global_fn += fn
            
            # Compute F1 for this term
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            
            per_term_f1_scores.append(f1)
        
        # Compute macro F1 (average across all GO terms)
        macro_f1 = float(np.mean(per_term_f1_scores)) if per_term_f1_scores else 0.0
        
        # Compute micro F1 (global counts)
        micro_precision = global_tp / (global_tp + global_fp + eps)
        micro_recall = global_tp / (global_tp + global_fn + eps)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
        micro_f1 = float(micro_f1)
        
        return macro_f1, micro_f1

    def configure_optimizers(self):
        # Use AdamW; hyperparameters were saved in self.hparams by save_hyperparameters()
        lr = self.hparams.get('lr', 1e-4) if isinstance(self.hparams, dict) else getattr(self.hparams, 'lr', 1e-4)
        weight_decay = self.hparams.get('weight_decay', 1e-5) if isinstance(self.hparams, dict) else getattr(self.hparams, 'weight_decay', 1e-5)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Use ReduceLROnPlateau scheduler (monitoring validation metric 'val_fmax_macro').
        # Parameters can be provided via hparams: 'plateau_factor', 'plateau_patience', 'min_lr', 'plateau_threshold'.
        if isinstance(self.hparams, dict):
            plateau_factor = self.hparams.get('plateau_factor', 0.5)
            plateau_patience = int(self.hparams.get('plateau_patience', 3))
            plateau_min_lr = self.hparams.get('min_lr', 1e-6)
            plateau_threshold = self.hparams.get('plateau_threshold', 1e-4)
        else:
            plateau_factor = getattr(self.hparams, 'plateau_factor', 0.5)
            plateau_patience = int(getattr(self.hparams, 'plateau_patience', 3))
            plateau_min_lr = getattr(self.hparams, 'min_lr', 1e-6)
            plateau_threshold = getattr(self.hparams, 'plateau_threshold', 1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            min_lr=float(plateau_min_lr),
            threshold=float(plateau_threshold)
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro_go',
                'interval': 'epoch',
                'frequency': 1,
            },
        }