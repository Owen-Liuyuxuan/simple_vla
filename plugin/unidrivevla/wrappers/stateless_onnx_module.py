# Copyright 2026 The Xiaomi Corporation. All rights reserved.
"""Thin wrapper exposing an explicit ``(inputs, memory_in) -> (outputs, memory_out)`` surface.

Full Qwen3-VL + flow-matching ONNX is separate; this module bundles the planning head’s
:class:`~projects.mmdet3d_plugin.unidrivevla.dense_heads.qwenvl3_vla_planning_head.QwenVL3APlanningHead.forward_inference_stateless`
call for tensor-only temporal memory. Detector-level packaging is in
:class:`UniDriveVLATemporalMemoryDetectorWrapper`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from plugin.unidrivevla.wrappers.packed_temporal_memory import (
    PackedTemporalMemory,
)


class UniDriveVLATemporalMemoryModule(nn.Module):
    """Wraps a :class:`QwenVL3APlanningHead` and delegates to ``forward_inference_stateless``."""

    def __init__(self, planning_head: nn.Module) -> None:
        super().__init__()
        self.planning_head = planning_head

    def forward(
        self,
        temporal_memory_in: Optional[PackedTemporalMemory],
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[PackedTemporalMemory]]:
        """Returns ``(pred_dict, temporal_memory_out)`` — same keys as ``forward_test`` plus memory out."""
        pred = self.planning_head.forward_inference_stateless(
            temporal_memory_in=temporal_memory_in,
            **kwargs,
        )
        mem = pred.pop("temporal_memory_out", None) if isinstance(pred, dict) else None
        return pred, mem


class UniDriveVLATemporalMemoryDetectorWrapper(nn.Module):
    """Wraps :class:`~projects.mmdet3d_plugin.unidrivevla.detectors.unidrivevla.UniDriveVLA` for functional memory."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(
        self,
        temporal_memory_in: Optional[PackedTemporalMemory] = None,
        return_loss: bool = False,
        rescale: bool = True,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[PackedTemporalMemory]]:
        """Calls ``detector(return_loss=False, …)`` with ``use_functional_temporal_memory``."""
        if return_loss:
            raise NotImplementedError("Training is not supported on this wrapper.")
        kwargs = dict(kwargs)
        kwargs["use_functional_temporal_memory"] = True
        kwargs["temporal_memory_in"] = temporal_memory_in
        out = self.detector(return_loss=False, rescale=rescale, **kwargs)
        mod = self.detector
        mem = getattr(mod, "_last_temporal_memory_out", None)
        return out, mem
