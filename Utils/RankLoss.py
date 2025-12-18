import torch
import torch.nn as nn

class RankLossPair(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super(RankLossPair, self).__init__()    
        self.reduction = reduction

    def forward(self, scores: torch.Tensor,
                label: torch.Tensor
                ) -> torch.Tensor:
        """
        Rank loss that penalizes when negative scores are higher than positive scores.
        For each pair of (negative, positive), computes loss = log(1 + exp(s_neg - s_pos))
        
        Args:
            scores: (B,) scores for all items in a batch
            label: (B,) binary labels (1 for positive, 0 for negative)
        
        Returns:
            loss: scalar loss
        """
        s_pos = scores[label == 1]  # (P,) positive scores
        s_neg = scores[label == 0]  # (N,) negative scores
        
        # If no positive or negative samples, return 0 loss
        if s_pos.numel() == 0 or s_neg.numel() == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        
        # Compute all pairwise differences: s_neg - s_pos
        # s_neg: (N, 1), s_pos: (1, P) -> diff: (N, P)
        diff = s_neg.unsqueeze(1) - s_pos.unsqueeze(0)
        
        # Apply log(1 + exp(diff)) for each pair
        loss = torch.log1p(torch.exp(diff))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        

