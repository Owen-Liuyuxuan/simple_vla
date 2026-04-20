# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit export/import of temporal instance-bank memory for UniDriveVLA / UnifiedPerceptionDecoder.

Instance banks live on :class:`UnifiedPerceptionDecoder` (det/map) and ego bank. This module
snapshots their mutable state so callers can reset or carry memory across forwards without
relying on hidden module globals between processes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

# MMDataParallel imported lazily in helpers so importing this module for lightweight
# utilities (e.g. ``infer_batch_size_from_data``) does not require mmcv at import time.


def _clone_if_tensor(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _copy_metas(metas: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep-enough copy of ``metas`` dicts used by instance banks (tensors cloned)."""
    if metas is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in metas.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        elif k == "img_metas" and isinstance(v, list):
            out[k] = []
            for im in v:
                d: Dict[str, Any] = {}
                if not isinstance(im, dict):
                    out[k].append(copy.deepcopy(im))
                    continue
                for ik, iv in im.items():
                    if torch.is_tensor(iv):
                        d[ik] = iv.detach().clone()
                    elif isinstance(iv, dict):
                        d[ik] = copy.deepcopy(iv)
                    else:
                        d[ik] = copy.deepcopy(iv)
                out[k].append(d)
        elif isinstance(v, dict):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _move_tensor_to(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device)


@dataclass
class InstanceBankState:
    """Mutable cache for :class:`InstanceBank` (detection / map)."""

    cached_feature: Optional[torch.Tensor] = None
    cached_anchor: Optional[torch.Tensor] = None
    metas: Optional[Dict[str, Any]] = None
    mask: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    temp_confidence: Optional[torch.Tensor] = None
    instance_id: Optional[torch.Tensor] = None
    prev_id: int = 0


@dataclass
class EgoInstanceBankState:
    """Mutable cache for :class:`EgoInstanceBank`."""

    cached_feature: Optional[torch.Tensor] = None
    cached_anchor: Optional[torch.Tensor] = None
    metas: Optional[Dict[str, Any]] = None
    mask: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    instance_id: Optional[torch.Tensor] = None


@dataclass
class TemporalBankState:
    """Full snapshot of temporal memory for one :class:`UnifiedPerceptionDecoder`."""

    det: Optional[InstanceBankState] = None
    map_: Optional[InstanceBankState] = None  # map head instance bank (attribute: map_instance_bank)
    ego: Optional[EgoInstanceBankState] = None


def export_instance_bank(bank) -> InstanceBankState:
    return InstanceBankState(
        cached_feature=_clone_if_tensor(getattr(bank, "cached_feature", None)),
        cached_anchor=_clone_if_tensor(getattr(bank, "cached_anchor", None)),
        metas=_copy_metas(getattr(bank, "metas", None)),
        mask=_clone_if_tensor(getattr(bank, "mask", None)),
        confidence=_clone_if_tensor(getattr(bank, "confidence", None)),
        temp_confidence=_clone_if_tensor(getattr(bank, "temp_confidence", None)),
        instance_id=_clone_if_tensor(getattr(bank, "instance_id", None)),
        prev_id=int(getattr(bank, "prev_id", 0)),
    )


def import_instance_bank(bank, state: Optional[InstanceBankState], ref_device: torch.device) -> None:
    if state is None:
        bank.reset()
        return
    bank.cached_feature = _move_tensor_to(state.cached_feature, ref_device)
    bank.cached_anchor = _move_tensor_to(state.cached_anchor, ref_device)
    bank.metas = _copy_metas(state.metas)
    bank.mask = _move_tensor_to(state.mask, ref_device)
    bank.confidence = _move_tensor_to(state.confidence, ref_device)
    bank.temp_confidence = _move_tensor_to(state.temp_confidence, ref_device)
    bank.instance_id = _move_tensor_to(state.instance_id, ref_device)
    bank.prev_id = state.prev_id


def export_ego_instance_bank(bank) -> EgoInstanceBankState:
    return EgoInstanceBankState(
        cached_feature=_clone_if_tensor(getattr(bank, "cached_feature", None)),
        cached_anchor=_clone_if_tensor(getattr(bank, "cached_anchor", None)),
        metas=_copy_metas(getattr(bank, "metas", None)),
        mask=_clone_if_tensor(getattr(bank, "mask", None)),
        confidence=_clone_if_tensor(getattr(bank, "confidence", None)),
        instance_id=_clone_if_tensor(getattr(bank, "instance_id", None)),
    )


def import_ego_instance_bank(bank, state: Optional[EgoInstanceBankState], ref_device: torch.device) -> None:
    if state is None:
        bank.reset()
        return
    bank.cached_feature = _move_tensor_to(state.cached_feature, ref_device)
    bank.cached_anchor = _move_tensor_to(state.cached_anchor, ref_device)
    bank.metas = _copy_metas(state.metas)
    bank.mask = _move_tensor_to(state.mask, ref_device)
    if hasattr(bank, "confidence"):
        bank.confidence = _move_tensor_to(state.confidence, ref_device)
    if hasattr(bank, "instance_id"):
        bank.instance_id = _move_tensor_to(state.instance_id, ref_device)


def _unwrap_parallel(model: nn.Module) -> nn.Module:
    """Unwrap ``DataParallel`` / ``DistributedDataParallel`` / optional ``MMDataParallel``."""
    m = model
    while isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    try:
        from mmcv.parallel import MMDataParallel  # type: ignore

        if isinstance(m, MMDataParallel):
            m = m.module
    except ImportError:
        pass
    return m


def get_unified_decoder(model: nn.Module):
    """Resolve ``UnifiedPerceptionDecoder`` from ``UniDriveVLA`` (unwraps parallel wrappers)."""
    model = _unwrap_parallel(model)
    if not hasattr(model, "planning_head") or model.planning_head is None:
        raise AttributeError("Model has no planning_head.")
    head = model.planning_head
    if not hasattr(head, "unified_decoder"):
        raise AttributeError("planning_head has no unified_decoder (expected UniDriveVLA).")
    return head.unified_decoder


def _bank_ref_device(unified_decoder) -> torch.device:
    if hasattr(unified_decoder, "det_instance_bank") and unified_decoder.det_instance_bank is not None:
        return unified_decoder.det_instance_bank.anchor.device
    if hasattr(unified_decoder, "ego_instance_bank") and unified_decoder.ego_instance_bank is not None:
        return unified_decoder.ego_instance_bank.anchor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_temporal_bank_state(unified_decoder) -> TemporalBankState:
    det_st = None
    map_st = None
    ego_st = None
    if getattr(unified_decoder, "det_instance_bank", None) is not None:
        det_st = export_instance_bank(unified_decoder.det_instance_bank)
    if getattr(unified_decoder, "map_instance_bank", None) is not None:
        map_st = export_instance_bank(unified_decoder.map_instance_bank)
    if getattr(unified_decoder, "ego_instance_bank", None) is not None:
        ego_st = export_ego_instance_bank(unified_decoder.ego_instance_bank)
    return TemporalBankState(det=det_st, map_=map_st, ego=ego_st)


def import_temporal_bank_state(unified_decoder, state: Optional[TemporalBankState]) -> None:
    dev = _bank_ref_device(unified_decoder)
    if state is None:
        reset_temporal_banks(unified_decoder)
        return
    if unified_decoder.det_instance_bank is not None:
        import_instance_bank(unified_decoder.det_instance_bank, state.det, dev)
    if unified_decoder.map_instance_bank is not None:
        import_instance_bank(unified_decoder.map_instance_bank, state.map_, dev)
    if unified_decoder.ego_instance_bank is not None:
        import_ego_instance_bank(unified_decoder.ego_instance_bank, state.ego, dev)


def reset_temporal_banks(unified_decoder) -> None:
    if getattr(unified_decoder, "det_instance_bank", None) is not None:
        unified_decoder.det_instance_bank.reset()
    if getattr(unified_decoder, "map_instance_bank", None) is not None:
        unified_decoder.map_instance_bank.reset()
    if getattr(unified_decoder, "ego_instance_bank", None) is not None:
        unified_decoder.ego_instance_bank.reset()


def temporal_state_batch_size(state: Optional[TemporalBankState]) -> Optional[int]:
    """Infer batch size from cached anchors, if present."""
    if state is None:
        return None
    for sub in (state.det, state.map_, state.ego):
        if sub is None:
            continue
        ca = getattr(sub, "cached_anchor", None)
        if torch.is_tensor(ca) and ca.dim() >= 1:
            return int(ca.shape[0])
    return None


def validate_temporal_state_batch_size(state: Optional[TemporalBankState], batch_size: int) -> None:
    bs = temporal_state_batch_size(state)
    if bs is not None and bs != batch_size:
        raise ValueError(
            f"TemporalBankState batch size {bs} does not match current batch {batch_size}. "
            "Import None to reset banks or use matching batch size."
        )


def infer_batch_size_from_data(data_batch: dict) -> Optional[int]:
    """Infer batch size from collated dataloader output (unwraps ``DataContainer``).

    When ``img`` is an MMCV :class:`~mmcv.parallel.DataContainer` whose ``.data`` is a
    **list** of tensors (e.g. one chunk ``[1, 6, 3, H, W]``), batch size is the **sum**
    of each chunk's leading dimension (micro-batch / multi-chunk layout).
    """
    x = data_batch.get("img")
    if x is None:
        return None
    t = x.data if hasattr(x, "data") else x
    if torch.is_tensor(t):
        return int(t.shape[0])
    if isinstance(t, (list, tuple)) and len(t) > 0:
        if all(torch.is_tensor(x) for x in t):
            return int(sum(int(x.shape[0]) for x in t))
    return None


@torch.no_grad()
def stateless_forward_functional_memory(
    model: nn.Module,
    data_batch: dict,
    packed_memory_in: Optional[Any] = None,
    det_tracking_in: Optional[Any] = None,
) -> Tuple[Union[list, dict], Any, Any]:
    """Run inference with :meth:`forward_inference_stateless` (packed tensor memory, no bank import/export).

    The detector must be :class:`~projects.mmdet3d_plugin.unidrivevla.detectors.unidrivevla.UniDriveVLA`
    with a planning head that implements ``forward_inference_stateless``. Temporal features use
    ``packed_memory_in`` / ``packed_memory_out`` only; detection instance IDs use tensor state
    (``det_tracking_in`` / ``det_tracking_out`` on the detector) — banks are **not** reset here.

    When ``packed_memory_in`` is ``None`` but batch size is known, memory is replaced with
    :func:`~projects.mmdet3d_plugin.unidrivevla.wrappers.packed_temporal_memory.cold_packed_memory_like_unpack_none`
    (same tensor shapes as unpack with no history — empty temporal dim ``(B, 0, *)``, not full-``Kt`` zeros).

    Each bank slot in :class:`~projects.mmdet3d_plugin.unidrivevla.wrappers.packed_temporal_memory.PackedTemporalMemory`
    includes a per-batch ``valid`` tensor (see ``PackedInstanceBankMemory.valid``). That is fed as
    ``temporal_valid`` into :func:`~projects.mmdet3d_plugin.models.functional_instance_bank.functional_instance_bank_get`
    so temporal fusion uses ``mask = mask_time & temporal_valid`` (invalid rows skip fusion).

    Returns:
        ``results``, ``packed_memory_out`` (from ``model._last_temporal_memory_out``), and
        ``det_tracking_out`` (from ``model._last_det_tracking_out``).
    """
    from plugin.unidrivevla.wrappers.packed_temporal_memory import (
        cold_det_tracking_state,
        cold_packed_memory_like_unpack_none,
    )

    bs = infer_batch_size_from_data(data_batch)
    unified = None
    dev = None
    dt = None
    if packed_memory_in is None and bs is not None:
        unified = get_unified_decoder(model)
        dev = _bank_ref_device(unified)
        dt = next(unified.parameters()).dtype
        packed_memory_in = cold_packed_memory_like_unpack_none(
            bs,
            dev,
            dt,
            unified.embed_dims,
            11,
            20 * 3,
            11,
        )
    if det_tracking_in is None and bs is not None:
        if unified is None:
            unified = get_unified_decoder(model)
        if dev is None:
            dev = _bank_ref_device(unified)
        if dt is None:
            dt = next(unified.parameters()).dtype
        det_tracking_in = cold_det_tracking_state(
            bs,
            900,
            dev,
            dt,
        )
    
    if bs is not None and packed_memory_in is not None:
        validate_temporal_state_batch_size(
            _packed_to_temporal_state_for_batch_check(packed_memory_in), bs
        )
    data = dict(data_batch)
    data["use_functional_temporal_memory"] = True
    data["temporal_memory_in"] = packed_memory_in
    data["det_tracking_in"] = det_tracking_in
    model.eval()
    results = model(return_loss=False, rescale=True, **data)
    outputs, packed_out, det_track_out = results
    return outputs, packed_out, det_track_out


def _packed_to_temporal_state_for_batch_check(packed: Any) -> Optional[TemporalBankState]:
    """Reuse :func:`validate_temporal_state_batch_size` for packed tensor memory."""
    if packed is None:
        return None
    for sub in (
        getattr(packed, "det", None),
        getattr(packed, "map_", None),
        getattr(packed, "ego", None),
    ):
        if sub is None:
            continue
        ca = getattr(sub, "cached_anchor", None)
        if torch.is_tensor(ca) and ca.dim() >= 1 and ca.shape[1] > 0:
            return TemporalBankState(det=InstanceBankState(cached_anchor=ca))
    return None


@torch.no_grad()
def stateless_forward(
    model: nn.Module,
    data_batch: dict,
    bank_state_in: Optional[TemporalBankState] = None,
) -> Tuple[Union[list, dict], TemporalBankState]:
    """Run ``model(return_loss=False, rescale=True, **data_batch)`` with explicit bank memory.

    Args:
        model: Typically ``MMDataParallel`` wrapping ``UniDriveVLA``.
        data_batch: One collated batch from the dataloader (tensor values).
        bank_state_in: Memory from the previous step; ``None`` resets all temporal banks.

    Returns:
        ``results``: Same structure as the underlying detector (list of ``img_bbox`` dicts, etc.).
        ``bank_state_out``: Snapshot after forward; pass as ``bank_state_in`` on the next step.
    """
    bs = infer_batch_size_from_data(data_batch)
    if bs is not None and bank_state_in is not None:
        validate_temporal_state_batch_size(bank_state_in, bs)
    unified = get_unified_decoder(model)
    import_temporal_bank_state(unified, bank_state_in)
    model.eval()
    results = model(return_loss=False, rescale=True, **data_batch)
    bank_state_out = export_temporal_bank_state(unified)
    return results, bank_state_out
