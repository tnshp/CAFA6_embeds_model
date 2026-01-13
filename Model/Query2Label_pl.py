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
        # Top-k filtering for validation
        top_k_predictions: int = None,
        # IA (Information Accretion) weights for F1 computation and WBCE loss
        ia_dict: dict = None,
        # Epsilon for WBCE loss (to handle zero IA scores)
        epsilon: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['ia_dict'])  # Don't save ia_dict in hparams
        self.ia_dict = ia_dict  # Store separately
        self.epsilon = epsilon  # Store epsilon for WBCE

        self.model = Query2Label(
            num_classes=num_classes,
            in_dim=in_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
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
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif lf in ('WBCE', 'WEIGHTED_BCE', 'WEIGHTEDBCE'):
            # Weighted BCE with IA scores - will compute weights in training/validation steps
            self.criterion = None  # Will use weighted BCE manually with ia_dict
        elif lf in ('RANK', 'RANKLOSS', 'RANK_LOSS'):
            # RankLoss uses RankLossPair class
            self.criterion = RankLossPair(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_function: {loss_function}. Supported: 'ASL', 'BCE', 'WBCE', 'RankLoss'.")
        
        # Store loss function name for training step
        self.loss_function = lf

        # Store validation step outputs for on_validation_epoch_end
        self.validation_step_outputs = []
        # Test step outputs
        self.test_step_outputs = []
        # For protein-based F1 computation (store predictions per protein)
        self._val_protein_predictions = {}  # {protein_id: {'pred_terms': set, 'true_terms': set}}
        self._rank_score_list = []
        self._cutoff_score_list = []
        # Store top-k parameter
        self.top_k_predictions = top_k_predictions

    def forward(self, x: torch.Tensor, f: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Invert mask: PyTorch expects True for positions to IGNORE, we have True for VALID positions
        if mask is not None:
            mask = ~mask
        return self.model(x, f, src_key_padding_mask=mask) 

    def training_step(self, batch, batch_idx):
        """Standard training step.

        Expects `batch` to be a dict with keys `'tokens'` and `'label'` where
        - `tokens` is a float Tensor of shape (B, L, C_in)
        - `label` is a binary Tensor of shape (B, num_classes)
        """
        x = batch['go_embed']   # (B, L, C_in)
        f = batch['features']   
        y = batch['label']
        mask = batch.get('mask', None)  # (B, L) attention mask

        logits = self.forward(x, f, mask)
        
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
        elif self.loss_function in ('WBCE', 'WEIGHTED_BCE', 'WEIGHTEDBCE'):
            # Weighted BCE with IA scores (vectorized)
            # Get predicted terms for this batch to compute weights
            predicted_terms = batch['predicted_terms']  # List of lists of GO terms
            
            # Vectorized weight computation
            batch_size = len(predicted_terms)
            num_classes = len(predicted_terms[0]) if predicted_terms else 0
            
            # Flatten all terms and look up IA scores in one pass
            all_terms = [term for terms_list in predicted_terms for term in terms_list]
            all_ia_scores = np.array([self.ia_dict.get(term, 0.0) if self.ia_dict else 0.0 for term in all_terms], dtype=np.float32)
            
            # Add epsilon and convert to tensor
            all_weights = torch.from_numpy(all_ia_scores + self.epsilon).to(y.device, dtype=y.dtype)
            
            # Reshape to (batch_size, num_classes)
            weights = all_weights.reshape(batch_size, num_classes)
            
            # Compute weighted BCE loss
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=weights, reduction='mean'
            )
            loss = bce_loss
        else:
            # Standard loss functions (ASL, BCE)
            loss = self.criterion(logits, y)

        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step computes loss and stores for F-max computation."""
        x = batch['go_embed']   # (B, L, C_in)
        f = batch['features']   
        y = batch['label']
        mask = batch.get('mask', None)  # (B, L) attention mask

        logits = self.forward(x, f, mask)
        
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
        elif self.loss_function in ('WBCE', 'WEIGHTED_BCE', 'WEIGHTEDBCE'):
            # Weighted BCE with IA scores (vectorized)
            # Get predicted terms for this batch to compute weights
            predicted_terms = batch['predicted_terms']  # List of lists of GO terms
            
            # Vectorized weight computation
            batch_size = len(predicted_terms)
            num_classes = len(predicted_terms[0]) if predicted_terms else 0
            
            # Flatten all terms and look up IA scores in one pass
            all_terms = [term for terms_list in predicted_terms for term in terms_list]
            all_ia_scores = np.array([self.ia_dict.get(term, 0.0) if self.ia_dict else 0.0 for term in all_terms], dtype=np.float32)
            
            # Add epsilon and convert to tensor
            all_weights = torch.from_numpy(all_ia_scores + self.epsilon).to(y.device, dtype=y.dtype)
            
            # Reshape to (batch_size, num_classes)
            weights = all_weights.reshape(batch_size, num_classes)
            
            # Compute weighted BCE loss
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=weights, reduction='mean'
            )
            loss = bce_loss
        else:
            # Standard loss functions (ASL, BCE)
            loss = self.criterion(logits, y)

        # Log per-step validation loss (will be reduced automatically)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Store for epoch end aggregation
        self.validation_step_outputs.append({'val_loss': loss.detach()})
        
        # Store protein-specific predictions for macro F1
        probs = torch.sigmoid(logits)
        targets = (y > 0.5).int()
        try:
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            predicted_terms = batch['predicted_terms']  # List of lists of GO term IDs
            entry_ids = batch['entryID']  # List of protein IDs
            true_terms = batch.get('true_terms', None)  # List of lists of true GO term IDs
            
            #Calculate rankscore and cutoff score
            rank_score = self._compute_rank_score(probs_np, targets_np, predicted_terms)
            cutoff_score = self._compute_cutoff_score(probs_np, targets_np, predicted_terms)

            self._rank_score_list.append(rank_score)
            self._cutoff_score_list.append(cutoff_score)

            # For each sample in the batch
            for sample_idx in range(len(predicted_terms)):
                entry_id = entry_ids[sample_idx]
                sample_terms = predicted_terms[sample_idx]
                sample_probs = probs_np[sample_idx]  # Shape: (num_terms_for_sample,)
                sample_targets = targets_np[sample_idx]  # Shape: (num_terms_for_sample,)
                
                # Get true terms for this sample
                sample_true_terms = set(true_terms[sample_idx]) if true_terms is not None else set()
                if not sample_true_terms:
                    # Fallback: derive from targets
                    sample_true_terms = set(term for i, term in enumerate(sample_terms) if sample_targets[i] > 0.5)
                
                # Apply top-k filtering if specified, otherwise use threshold
                if self.top_k_predictions is not None and self.top_k_predictions > 0:
                    # Get indices sorted by prediction score (descending)
                    sorted_indices = np.argsort(sample_probs)[::-1]
                    k = min(self.top_k_predictions, len(sample_probs))
                    top_k_indices = sorted_indices[:k]
                    sample_pred_terms = set(sample_terms[i] for i in top_k_indices)
                else:
                    # Use threshold (default 0.5)
                    threshold = 0.5
                    sample_pred_terms = set(term for i, term in enumerate(sample_terms) if sample_probs[i] >= threshold)
                
                # Store predictions and true terms for this protein
                if entry_id not in self._val_protein_predictions:
                    self._val_protein_predictions[entry_id] = {
                        'pred_terms': sample_pred_terms,
                        'true_terms': sample_true_terms
                    }
        except Exception as e:
            print(f"Warning: Could not store protein predictions: {e}")
            import traceback
            traceback.print_exc()
            pass

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch. Compute F-max only."""
        # compute mean validation loss for the epoch
        if not self.validation_step_outputs:
            return
        losses = torch.stack([o['val_loss'] for o in self.validation_step_outputs])
        mean_loss = losses.mean()

        
        self.log('val_loss_epoch', mean_loss, prog_bar=True, logger=True)
        
        mean_rank_score = float(np.mean(self._rank_score_list)) if self._rank_score_list else 0.0
        mean_cutoff_score = float(np.mean(self._cutoff_score_list)) if self._cutoff_score_list else 0.0
        self.log('val_rank_score', mean_rank_score, prog_bar=True, logger=True)
        self.log('val_cutoff_score', mean_cutoff_score, prog_bar=True, logger=True)


        # Compute protein-based macro F1 scores
        try:
            if len(self._val_protein_predictions) > 0:
                macro_f1, macro_precision, macro_recall = self._compute_protein_f1_scores()
                self.log('val_f1_macro_go', macro_f1, prog_bar=True, logger=True)
                self.log('val_precision_macro', macro_precision, prog_bar=True, logger=True)
                self.log('val_recall_macro', macro_recall, prog_bar=True, logger=True)
                
                try:
                    rank = getattr(self, 'global_rank', 0)
                except Exception:
                    rank = 0
                if rank == 0:
                    weighting_str = "IA-weighted" if self.ia_dict is not None else "Unweighted"
                    print(f"Validation Protein-based Macro ({weighting_str}) - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}, Rank Score: {mean_rank_score:.4f}, Cutoff Score: {mean_cutoff_score:.4f}")
                
                # Clear protein prediction buffers
                self._val_protein_predictions.clear()
                self._rank_score_list.clear()
                self._cutoff_score_list.clear()
                
        except Exception as e:
            print(f"Warning: Could not compute protein-based F1 scores: {e}")
            import traceback
            traceback.print_exc()
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

    def _compute_rank_score(self, probs_np, targets_np, predicted_terms):
        """Compute rank score based on GO term predictions.
        
        For each instance:
        1. Threshold = number of positive labels
        2. Sort predictions in descending order
        3. Count how many true labels are in top `threshold` positions
        4. Score = (true labels in top k) / (total true labels)
        """
        batch_scores = []
        
        for sample_idx in range(len(probs_np)):
            sample_probs = probs_np[sample_idx]
            sample_targets = targets_np[sample_idx]
            
            # Threshold is the number of positive labels
            num_positives = int(sample_targets.sum())
            
            # Skip if no positives
            if num_positives == 0:
                continue
            
            # Get indices of predictions sorted in descending order
            sorted_indices = np.argsort(sample_probs)[::-1]
            
            # Get top k indices where k = number of positive labels
            top_k_indices = sorted_indices[:num_positives]
            
            # Check how many of the top k predictions are true positives
            true_positives_in_top_k = sample_targets[top_k_indices].sum()
            
            # Calculate score for this sample
            score = true_positives_in_top_k / num_positives
            batch_scores.append(score)
        
        # Return average score across all samples in batch
        return float(np.mean(batch_scores)) if batch_scores else 0.0
        

    def _compute_cutoff_score(self, probs_np, targets_np, predicted_terms):
        """Compute cutoff score based on GO term predictions.
        
        For each instance:
        1. Sort predictions in descending order
        2. Find the index of the last (lowest ranked) true label
        3. That index is the cutoff score
        """
        batch_scores = []
        
        for sample_idx in range(len(probs_np)):
            sample_probs = probs_np[sample_idx]
            sample_targets = targets_np[sample_idx]
            
            # Skip if no positives
            num_positives = int(sample_targets.sum())
            if num_positives == 0:
                continue
            
            # Get indices of predictions sorted in descending order
            sorted_indices = np.argsort(sample_probs)[::-1]
            
            # Find positions of true labels in the sorted order
            true_label_positions = []
            for idx, sorted_idx in enumerate(sorted_indices):
                if sample_targets[sorted_idx] == 1:
                    true_label_positions.append(idx)
            
            # The cutoff score is the index of the last true label
            if true_label_positions:
                cutoff_score = max(true_label_positions)
                batch_scores.append(cutoff_score)
        
        # Return average score across all samples in batch
        return float(np.mean(batch_scores)) if batch_scores else 0.0


    def _compute_protein_f1_scores(self):
        """Compute macro F1 scores based on per-protein predictions.
        
        For each protein, compute precision, recall, and F1 by comparing
        predicted terms vs true terms. Then macro-average across all proteins.
        
        Uses IA (Information Accretion) weighting if ia_dict is provided.
        This matches the notebook implementation in test_model.ipynb.
        """
        eps = 1e-8
        per_protein_precisions = []
        per_protein_recalls = []
        per_protein_f1_scores = []
        
        use_ia_weighting = self.ia_dict is not None
        
        # Compute metrics for each protein
        for protein_id, protein_data in self._val_protein_predictions.items():
            pred_terms = protein_data['pred_terms']
            true_terms = protein_data['true_terms']
            
            # Compute TP, FP, FN for this protein
            if use_ia_weighting:
                # Weighted by IA scores
                tp = sum(self.ia_dict.get(term, 0.0) for term in (pred_terms & true_terms))
                fp = sum(self.ia_dict.get(term, 0.0) for term in (pred_terms - true_terms))
                fn = sum(self.ia_dict.get(term, 0.0) for term in (true_terms - pred_terms))
            else:
                # Unweighted (count)
                tp = len(pred_terms & true_terms)  # Intersection
                fp = len(pred_terms - true_terms)  # Predicted but not true
                fn = len(true_terms - pred_terms)  # True but not predicted
            
            # Compute precision, recall, F1 for this protein
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + eps) if (precision + recall) > 0 else 0.0
            
            per_protein_precisions.append(precision)
            per_protein_recalls.append(recall)
            per_protein_f1_scores.append(f1)
        
        # Macro-average: average across all proteins
        macro_precision = float(np.mean(per_protein_precisions)) if per_protein_precisions else 0.0
        macro_recall = float(np.mean(per_protein_recalls)) if per_protein_recalls else 0.0
        macro_f1 = float(np.mean(per_protein_f1_scores)) if per_protein_f1_scores else 0.0
        
        return macro_f1, macro_precision, macro_recall
        

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