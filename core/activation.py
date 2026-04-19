"""build_activation_layer — PyTorch-native activation factories.

mmcv.cnn.bricks.registry.ACTIVATION_LAYERS → replaced by local ACTIVATION_REGISTRY.
"""
import torch.nn as nn

from .registry import Registry, build_from_cfg

ACTIVATION_REGISTRY: Registry = Registry("activation")


def build_activation_layer(cfg: dict):
    """Factory matching mmcv.cnn.build_activation_layer.

    Supported cfg shapes (all lowercase keys are accepted by nn.Activation):
        {'type': 'ReLU', 'inplace': True}
        {'type': 'LeakyReLU', 'negative_slope': 0.1}
        {'type': 'GELU'}
        {'type': 'SiLU'}          # PyTorch name for Swish
        {'type': 'Mish'}
        {'type': 'Hardsigmoid'}
        {'type': 'Hardswish'}
        {'type': 'Tanh'}
        {'type': 'Sigmoid'}
        {'type': 'ELU', 'alpha': 1.0}
        {'type': 'PReLU', 'num_parameters': 1}
        {'type': 'Softplus', 'beta': 1, 'threshold': 20}
        {'type': 'Hardtanh', 'min_val', 'max_val'}
        {'type': 'GELU', 'approximate': 'tanh'}   # for torch>=1.10
    """
    if cfg is None:
        return nn.Identity()
    return build_from_cfg(cfg, ACTIVATION_REGISTRY)


# -------------------------------------------------------------------------
# Register PyTorch built-ins — no extra code required
# -------------------------------------------------------------------------
for _name in [
    'ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Mish',
    'Hardsigmoid', 'Hardswish', 'Tanh', 'Sigmoid',
    'ELU', 'PReLU', 'Softplus', 'Hardtanh',
]:
    ACTIVATION_REGISTRY.register_module(_name, getattr(nn, _name))


# Alias matching the original import name
def get_activation(name: str, **kwargs):
    """Return a fresh activation instance by name."""
    return build_activation_layer({'type': name, **kwargs})


__all__ = [
    'ACTIVATION_REGISTRY', 'build_activation_layer',
    'get_activation',
]
