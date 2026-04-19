"""mmdet loss function stubs -- PyTorch-native replacements."""
import torch.nn.functional as F

__all__ = ['l1_loss', 'smooth_l1_loss']


def l1_loss(input, target, reduction='mean'):
    return F.l1_loss(input, target, reduction=reduction)


def smooth_l1_loss(input, target, beta=1.0, reduction='mean'):
    return F.smooth_l1_loss(input, target, reduction=reduction)
