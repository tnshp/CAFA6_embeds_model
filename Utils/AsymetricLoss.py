import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    
    Parameters
    ----------
    gamma_neg : float, default=4
        Focusing parameter for negative samples (gamma^-)
        Higher values increase down-weighting of easy negatives
    gamma_pos : float, default=0
        Focusing parameter for positive samples (gamma^+)
        Typically set to 0 for standard cross-entropy on positives
    clip : float, default=0.05
        Probability margin (m) for probability shifting
        Shifts negative probabilities to hard-threshold easy negatives
    eps : float, default=1e-8
        Small epsilon for numerical stability in log operations
    disable_torch_grad_focal_loss : bool, default=True
        Optimization flag to disable gradient computation for focal weight calculation
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(self, x, y):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input logits of shape (batch_size, num_classes)
            Raw outputs before sigmoid activation
        y : torch.Tensor
            Ground truth multi-label binary targets of shape (batch_size, num_classes)
            Values should be 0 or 1
            
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        # Calculate probabilities from logits
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping (Probability Shifting)
        # Shift negative probabilities: p_m = max(p - m, 0)
        # This is implemented as: (1-p) + m, then clamp to [0, 1]
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic Cross-Entropy calculation
        # For positives: y * log(p)
        # For negatives: (1-y) * log(1-p_m)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric Focusing
        # Apply (1-p)^gamma weighting, with different gamma for pos/neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Disable gradient for focal weight calculation (memory optimization)
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # Calculate pt = p if target=1, else 1-p
                    pt0 = xs_pos * y
                    pt1 = xs_neg * (1 - y)
                    pt = pt0 + pt1
                    
                    # one_sided_gamma selects gamma_pos for positives, gamma_neg for negatives
                    one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                    
                    # Calculate focal weight: (1 - pt)^gamma
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                # Calculate pt = p if target=1, else 1-p
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                
                # one_sided_gamma selects gamma_pos for positives, gamma_neg for negatives
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                
                # Calculate focal weight: (1 - pt)^gamma
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            # Apply focal weight to loss
            loss *= one_sided_w
        
        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """
    Optimized version of Asymmetric Loss
    
    This implementation minimizes memory allocations and GPU uploads,
    and favors inplace operations for better performance.
    Functionality is identical to AsymmetricLoss.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOptimized, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
        # Pre-allocate tensors to prevent memory allocation every iteration
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = None
        self.asymmetric_w = self.loss = None
    
    def forward(self, x, y):
        """
        Forward pass with memory-optimized operations
        
        Parameters
        ----------
        x : torch.Tensor
            Input logits of shape (batch_size, num_classes)
        y : torch.Tensor
            Ground truth multi-label binary targets of shape (batch_size, num_classes)
            
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        self.targets = y
        self.anti_targets = 1 - y
        
        # Calculate probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        
        # Asymmetric Clipping (inplace operations)
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        
        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            
            # Calculate asymmetric focal weight
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            self.loss *= self.asymmetric_w
        
        return -self.loss.sum()
