"""
Batch Mean Contrastive (BMC) regression loss.

Provides:
- `bmc_loss`: functional form
- `BMCLoss`: nn.Module wrapper with learnable or fixed noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bmc_loss(pred: torch.Tensor,
             target: torch.Tensor,
             noise_var: torch.Tensor,
             detach_scale: bool = True) -> torch.Tensor:
    """
    BMC loss for scalar regression.

    - Builds a pairwise squared-error matrix between predictions and targets.
    - Uses it as logits for cross-entropy over the batch (diagonal is the match).
    - Optionally detaches the scale factor to stabilize optimization.
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    # Pairwise negative squared errors as logits: shape [B, B]
    logits = - (pred.unsqueeze(1) - target.unsqueeze(0)).pow(2) / (2 * noise_var)

    # Each prediction is matched to its own target (diagonal)
    labels = torch.arange(pred.size(0), device=pred.device)

    ce     = F.cross_entropy(logits, labels, reduction='mean')

    # Scale explained variance back to MSE scale
    scale  = 2 * noise_var
    if detach_scale:
        scale = scale.detach()

    return ce * scale


class BMCLoss(nn.Module):
    """
    Module wrapper for BMC loss.

    Args:
        init_noise_sigma: initial sigma; variance = sigma^2
        learn_noise: if True, sigma is learnable
        detach_scale: if True, detaches the scale in the final multiplication
    """
    def __init__(self,
                 init_noise_sigma: float = 1.0,
                 learn_noise: bool  = True,
                 detach_scale: bool = True):
        super().__init__()
        self.detach_scale = detach_scale
        if learn_noise:
            # sigma is learnable
            self.noise_sigma = nn.Parameter(torch.tensor(init_noise_sigma,
                                                         dtype=torch.float))
        else:
            # sigma is fixed
            self.register_buffer('noise_sigma',
                                 torch.tensor(init_noise_sigma,
                                              dtype=torch.float))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        noise_var = self.noise_sigma.pow(2)
        return bmc_loss(pred, target, noise_var, self.detach_scale)
