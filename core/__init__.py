"""simple_vla.core — zero-dependency foundation for the UniDriveVLA inference stack.

All symbols are re-exported here so downstream modules can use::

    from simple_vla.core import (
        Registry, build_from_cfg, build,
        ATTENTION, PLUGIN_LAYERS, POSITIONAL_ENCODING,
        FEEDFORWARD_NETWORK, NORM_LAYERS, HEADS, LOSSES,
        BBOX_SAMPLERS, BBOX_CODERS,
        BaseModule, Scale,
        xavier_init, constant_init, bias_init_with_prob,
        build_activation_layer, get_activation,
        build_norm_layer,
        FFN,
        build_dropout,
        auto_fp16, deprecated_api_warning,
        reduce_mean,
    )
"""
# Registry + build factory
from .registry import (
    Registry, build_from_cfg, build,
    ATTENTION, PLUGIN_LAYERS, POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK, NORM_LAYERS, HEADS, LOSSES,
    BBOX_SAMPLERS, BBOX_CODERS, BBOX_ASSIGNERS,
    DETECTORS, BACKBONES, NECKS,
    build_head, build_backbone, build_neck, build_loss, build_assigner,
)

# nn / initialisation
from .nn import (
    BaseModule, Sequential, Scale,
    xavier_init, constant_init, bias_init_with_prob,
)

# Activation / Norm factories
from .activation import ACTIVATION_REGISTRY, build_activation_layer, get_activation
from .norm import NORM_LAYERS, build_norm_layer

# Transformer building blocks
from .transformer import FFN

# Dropout factory
from .dropout import DROPOUT_LAYERS, build_dropout

# Decorator stubs
from .decorators import deprecated_api_warning

# FP16 helper
from .fp16_helper import auto_fp16

# Distributed utilities (identity in single-GPU inference)
from .distributed import reduce_mean

# Box3d index constants
from .box3d import (
    X, Y, Z, W, L, H,
    SIN_YAW, COS_YAW, VX, VY, YAW,
    CNS, YNS,
)

__all__ = [
    # Registry
    'Registry', 'build_from_cfg', 'build',
    'ATTENTION', 'PLUGIN_LAYERS', 'POSITIONAL_ENCODING',
    'FEEDFORWARD_NETWORK', 'NORM_LAYERS',
    'HEADS', 'LOSSES', 'BBOX_SAMPLERS', 'BBOX_CODERS', 'BBOX_ASSIGNERS',
    'DETECTORS', 'BACKBONES', 'NECKS',
    'build_head', 'build_backbone', 'build_neck', 'build_loss', 'build_assigner',
    # nn
    'BaseModule', 'Sequential', 'Scale',
    'xavier_init', 'constant_init', 'bias_init_with_prob',
    # activation / norm
    'ACTIVATION_REGISTRY', 'build_activation_layer', 'get_activation',
    'NORM_LAYERS', 'build_norm_layer',
    # transformer
    'FFN',
    # dropout
    'DROPOUT_LAYERS', 'build_dropout',
    # decorators
    'auto_fp16', 'deprecated_api_warning',
    # distributed
    'reduce_mean',
    # box3d constants
    'X', 'Y', 'Z', 'W', 'L', 'H',
    'SIN_YAW', 'COS_YAW', 'VX', 'VY', 'YAW',
    'CNS', 'YNS',
]