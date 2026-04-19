"""distributed — reduced distributed utilities for single-GPU inference.

All training-distributed operations are no-ops.  Only ``reduce_mean`` is
required by the codebase; for single-GPU it is the identity function.
"""
import torch


def reduce_mean(tensor):
    """Return the mean across all processes — single-GPU returns input unchanged."""
    return tensor


__all__ = [
    'reduce_mean',
]
