"""build_norm_layer — LayerNorm / BatchNorm factory.

Replaces mmcv.cnn.build_norm_layer which returns (norm_type, norm_layer)
tuples.  In practice most code only cares about the layer instance, so we
provide both styles for compatibility.
"""
import torch.nn as nn

from .registry import Registry, build_from_cfg

NORM_LAYERS: Registry = Registry("norm")


def build_norm_layer(cfg: dict, num_features: int) -> tuple[str, nn.Module]:
    """Build a normalization layer from config.

    Mimics mmcv: returns (name, layer).

    Examples
    --------
    >>> build_norm_layer({'type': 'LN'}, 256)
    ('LN', LayerNorm(256, eps=1e-05, elementwise_affine=True))
    >>> build_norm_layer({'type': 'LN', 'eps': 1e-6}, 256)
    ('LN', LayerNorm(256, eps=1e-06, elementwise_affine=True))
    >>> build_norm_layer({'type': 'BN1d'}, 256)
    ('BN1d', BatchNorm1d(256, eps=1e-05, momentum=0.1))
    """
    if cfg is None:
        return 'Identity', nn.Identity()

    # Accept both {'type': 'LN'} and {'type': 'layernorm'}
    raw_type = cfg.get('type', '').lower()

    # Normalise canonical names
    if raw_type in ['ln', 'layernorm']:
        norm_type = 'LN'
    elif raw_type in ('bn1d', 'batchnorm1d'):
        norm_type = 'BN1d'
    elif raw_type in ('bn2d', 'batchnorm2d'):
        norm_type = 'BN2d'
    elif raw_type in ('gn', 'groupnorm'):
        norm_type = 'GN'
    elif raw_type in ('syncbn', 'sync_bn'):
        norm_type = 'SyncBN'
    else:
        # Pass through as-is for custom registrations
        norm_type = cfg['type']

    # Copy constructor kwargs and drop 'type'
    kwargs = {k: v for k, v in cfg.items() if k != 'type'}

    # Common defaults
    if norm_type == 'LN':
        kwargs.setdefault('eps', 1e-5)
        layer = nn.LayerNorm(num_features, **kwargs)
    elif norm_type == 'BN1d':
        kwargs.setdefault('eps', 1e-5)
        kwargs.setdefault('momentum', 0.1)
        layer = nn.BatchNorm1d(num_features, **kwargs)
    elif norm_type == 'BN2d':
        kwargs.setdefault('eps', 1e-5)
        kwargs.setdefault('momentum', 0.1)
        layer = nn.BatchNorm2d(num_features, **kwargs)
    elif norm_type == 'GN':
        kwargs.setdefault('eps', 1e-5)
        # num_channels -> num_features
        group_num = kwargs.pop('num_groups', 32)
        layer = nn.GroupNorm(group_num, num_features, **kwargs)
    elif norm_type == 'SyncBN':
        # Distributed batch-norm is not needed for single-GPU inference
        # Fall back to BN1d but keep the naming for compatibility
        kwargs.setdefault('eps', 1e-5)
        kwargs.setdefault('momentum', 0.1)
        layer = nn.BatchNorm1d(num_features, **kwargs)
    else:
        # Custom norm registered in the registry
        layer = build_from_cfg(cfg, NORM_LAYERS)

    return norm_type, layer


__all__ = [
    'NORM_LAYERS', 'build_norm_layer',
]
