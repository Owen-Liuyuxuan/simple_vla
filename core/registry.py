"""Registry system — lightweight replacement for mmcv.utils.Registry + build_from_cfg.

All classes in this file are self-contained and have zero dependency on mmcv/mmdet.
This is the foundation of the plugin system: every @HEADS.register_module()
decorator writes into these global registry maps.
"""
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar

import torch.nn as nn

T = TypeVar('T')


class Registry:
    """Lightweight registry mapping string names → classes.

    Minimal subset of mmcv.utils.Registry used throughout UniDriveVLA:
    - ``@registry.register_module()`` decorator
    - ``registry.get(name)`` lookup
    - ``build_from_cfg(cfg, registry)`` factory
    """

    def __init__(self, name: str):
        self.name = name
        self._module_map: Dict[str, Callable] = {}

    def get(self, name: str) -> Optional[Callable]:
        """Return the class registered under *name* (or the class itself if already a type)."""
        if not isinstance(name, str):
            return name
        return self._module_map.get(name)

    def register_module(self, name: Optional[str] = None, module: Optional[Callable] = None):
        """Decorator: ``@registry.register_module()`` or ``@registry``."""
        if module is not None:
            # Direct registration: ``registry.register_module(module=MyClass)``
            self._module_map[module.__name__] = module
            return module

        def _register(cls: T) -> T:
            key = name if name is not None else cls.__name__
            self._module_map[key] = cls
            return cls
        return _register

    # Allow ``@registry`` shorthand
    def __call__(self, cls_or_name=None, **kwargs):
        if cls_or_name is None:
            return self.register_module(**kwargs)
        if isinstance(cls_or_name, str):
            return self.register_module(name=cls_or_name, **kwargs)
        return self.register_module(module=cls_or_name)


def build_from_cfg(
    cfg: Dict,
    registry: Registry,
    default_args: Optional[Dict] = None,
) -> Any:
    """Instantiate an object from a config dict — drop-in for mmcv.utils.build_from_cfg.

    Expected *cfg* format: ``{'type': 'ClassName', 'arg1': val1, ...}``.  The
    ``type`` key is looked up in *registry* and the remaining keys are passed
    to the class constructor.  *default_args* are merged via ``setdefault`` so
    explicit config values always win.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, got {type(cfg)}')

    args = cfg.copy()
    if default_args is not None:
        if not isinstance(default_args, dict):
            raise TypeError(f'default_args must be dict or None, got {type(default_args)}')
        for k, v in default_args.items():
            args.setdefault(k, v)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be str or class/function, got {type(obj_type)}')

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}')


def build(cfg: Optional[Dict], registry: 'Registry') -> Optional[Any]:
    """None-safe wrapper matching the local ``build()`` helpers used project-wide."""
    if cfg is None:
        return None
    return build_from_cfg(cfg, registry)


# -------------------------------------------------------------------------
# Global registry namespaces — mirror mmdet/mmcv naming
# -------------------------------------------------------------------------
ATTENTION = Registry("attention")
PLUGIN_LAYERS = Registry("plugin_layers")
POSITIONAL_ENCODING = Registry("positional_encoding")
FEEDFORWARD_NETWORK = Registry("feedforward_network")
NORM_LAYERS = Registry("norm_layers")


@NORM_LAYERS.register_module()
class LN(nn.LayerNorm):
    """LayerNorm under ``type='LN'`` — matches mmcv ``NORM_LAYERS`` registration."""

HEADS = Registry("heads")
LOSSES = Registry("losses")
BBOX_SAMPLERS = Registry("bbox_samplers")
BBOX_CODERS = Registry("bbox_coders")
BBOX_ASSIGNERS = Registry("bbox_assigners")

DETECTORS = Registry("detectors")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")


def build_head(cfg: Optional[Dict], default_args: Optional[Dict] = None) -> Any:
    if cfg is None:
        raise TypeError("build_head: cfg must not be None")
    return build_from_cfg(cfg, HEADS, default_args)


def build_backbone(cfg: Optional[Dict], default_args: Optional[Dict] = None) -> Any:
    if cfg is None:
        raise TypeError("build_backbone: cfg must not be None")
    return build_from_cfg(cfg, BACKBONES, default_args)


def build_neck(cfg: Optional[Dict], default_args: Optional[Dict] = None) -> Any:
    if cfg is None:
        raise TypeError("build_neck: cfg must not be None")
    return build_from_cfg(cfg, NECKS, default_args)


def build_loss(cfg: Optional[Dict], default_args: Optional[Dict] = None) -> Any:
    if cfg is None:
        raise TypeError("build_loss: cfg must not be None")
    return build_from_cfg(cfg, LOSSES, default_args)


def build_assigner(cfg: Optional[Dict], default_args: Optional[Dict] = None) -> Any:
    if cfg is None:
        raise TypeError("build_assigner: cfg must not be None")
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


# Convenience: re-export for ``from core import build_from_cfg``
__all__ = [
    'Registry', 'build_from_cfg', 'build',
    'ATTENTION', 'PLUGIN_LAYERS', 'POSITIONAL_ENCODING',
    'FEEDFORWARD_NETWORK', 'NORM_LAYERS', 'LN', 'HEADS', 'LOSSES',
    'BBOX_SAMPLERS', 'BBOX_CODERS', 'BBOX_ASSIGNERS',
    'DETECTORS', 'BACKBONES', 'NECKS',
    'build_head', 'build_backbone', 'build_neck', 'build_loss', 'build_assigner',
]
