# Copyright (c) OpenMMLab. All rights reserved.
from simple_vla.core.registry import (
    BBOX_SAMPLERS,
    BBOX_CODERS,
    BBOX_ASSIGNERS,
    build_from_cfg,
    build_assigner as _build_assigner_core,
)


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return _build_assigner_core(cfg, default_args if default_args else None)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)
