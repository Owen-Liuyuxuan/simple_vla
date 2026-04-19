"""Dropout layer factory — mmcv.cnn.bricks.drop.build_dropout replacement.

Builds torch.nn dropout modules from dict configs.  Minimal feature set to
support the VLA stack (Dropout, Dropout2d, Dropout3d).
"""
import torch.nn as nn

from .registry import Registry, build_from_cfg

DROPOUT_LAYERS: Registry = Registry("dropout")


def build_dropout(cfg: dict, default_args=None):
    """Create a dropout layer from a config dict.

    Examples
    --------
    >>> build_dropout({'type': 'Dropout', 'p': 0.1})
    Dropout(p=0.1, inplace=False)
    """
    if cfg is None:
        return nn.Identity()
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)


# Register standard torch dropout layers
DROPOUT_LAYERS.register_module(module=nn.Dropout)
DROPOUT_LAYERS.register_module(module=nn.Dropout2d)
DROPOUT_LAYERS.register_module(module=nn.Dropout3d)


__all__ = [
    'DROPOUT_LAYERS', 'build_dropout',
]
