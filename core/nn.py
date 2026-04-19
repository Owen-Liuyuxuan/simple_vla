"""nn: PyTorch-native replacements for mmcv.cnn + mmcv.runner utilities.

Exports
-------
- BaseModule : nn.Module with no-op init_weights() (mmcv.runner.BaseModule subset)
- Sequential : nn.Sequential alias (no mmcv tracking)
- Scale      : learnable scalar multiplier (mmcv.cnn.Scale)
- xavier_init, constant_init, bias_init_with_prob : weight-initializers

All symbols match the mmcv names so imports do not need to change.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class BaseModule(nn.Module):
    """Minimal drop-in for mmcv.runner.BaseModule.

    The original tracks parameter initialization and provides ``init_cfg``
    support.  For inference we only need the inheritance from ``nn.Module`` and
    a no-op ``init_weights`` method — that keeps ``@BaseModule`` subclasses
    happy without pulling in mmcv.
    """

    def __init__(self, init_cfg=None):
        super().__init__()
        # We deliberately ignore init_cfg — weights are either loaded from
        # checkpoint or randomly initialised at runtime.

    def init_weights(self):
        """No-op.  Used only during training."""
        pass


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize weights with Xavier/Glorot init.

    Parameters
    ----------
    module : nn.Module
        Target module (Linear, Conv2d, etc.)
    gain : float
        Multiplicative factor for the standard deviation.
    bias : float | None
        Constant bias value; set to ``None`` to skip bias init.
    distribution : str
        'normal' (Xavier normal) or 'uniform' (Xavier uniform).
    """
    assert distribution in ['normal', 'uniform']
    if distribution == 'normal':
        init.xavier_normal_(module.weight, gain=gain)
    else:
        init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        if bias is not None:
            init.constant_(module.bias, bias)
        else:
            module.bias.data.zero_()


def constant_init(module, val, bias=0):
    """Fill weights and bias with constants."""
    module.weight.data.fill_(val)
    if module.bias is not None:
        module.bias.data.fill_(bias)


def bias_init_with_prob(prob):
    """Initialize bias for classification heads given positive sample ratio.

    Returns a callable suitable for::

        cls_bias = bias_init_with_prob(prior_prob)(module)
    """
    def init(module):
        module.bias.data.fill_(torch.log(torch.tensor(prob / (1 - prob))))
        return module
    return init


class Scale(nn.Module):
    """Learnable scalar multiplier — matches mmcv.cnn.Scale.

    Used inside deformable attention to scale the attention scores.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


# Aliases for legacy imports that expect mmcv.cnn names
Sequential = nn.Sequential
Linear = nn.Linear

__all__ = [
    'BaseModule', 'Sequential', 'Linear', 'Scale',
    'xavier_init', 'constant_init', 'bias_init_with_prob',
]
