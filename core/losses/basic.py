"""Minimal L1 helpers compatible with map loss call sites."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred, target, weight=None, reduction: str = "mean", avg_factor=None):
    """Element-wise L1 with optional channel weighting (last-dim reduction)."""
    diff = (pred - target).abs()
    if weight is not None:
        diff = diff * weight
    loss = diff.flatten(1).sum(-1) if diff.dim() > 1 else diff
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    if avg_factor is not None:
        loss = loss / avg_factor
    return loss


def smooth_l1_loss(
    pred,
    target,
    weight=None,
    reduction: str = "mean",
    avg_factor=None,
    beta: float = 0.5,
):
    """Smooth L1 (Huber) with optional weighting."""
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    if weight is not None:
        diff = diff * weight
    loss = diff.flatten(1).sum(-1) if diff.dim() > 1 else diff
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    if avg_factor is not None:
        loss = loss / avg_factor
    return loss
