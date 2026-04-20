# Copyright (c) OpenMMLab. All rights reserved.
"""Packed tensor-only temporal memory for stateless / ONNX-oriented inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

# Wall-clock timestamps and ego poses (``T_global`` / ``T_global_inv``) in packed memory and
# metas snapshots: float64 storage. In :func:`~projects.mmdet3d_plugin.models.functional_instance_bank.functional_instance_bank_get_from_tensors`,
# ``T_temp2cur`` is computed in float64 then cast to anchor dtype for projection.
_TS_DTYPE = torch.float64
_POSE_DTYPE = torch.float64


@dataclass
class PackedInstanceBankMemory:
    """One InstanceBank (det or map) temporal slot — tensor-only (no ``None`` history fields)."""

    valid: torch.Tensor  # (B,) float 0/1 — row has valid history
    cached_feature: torch.Tensor  # (B, Kt, C) or (B, 0, C) when empty
    cached_anchor: torch.Tensor  # (B, Kt, A)
    confidence: torch.Tensor  # (B, Kt); use (B, 0) when Kt==0
    timestamp_prev: torch.Tensor  # (B,) float64 — wall time stored at cache
    T_global_prev: torch.Tensor  # (B, 4, 4) float64
    T_global_inv_prev: torch.Tensor  # (B, 4, 4) float64


@dataclass
class PackedDetTrackingState:
    """Detection instance-ID state carried across stateless forwards (no ``InstanceBank`` mutation)."""

    prev_id: torch.Tensor  # scalar int64 — next ID base
    instance_id: torch.Tensor  # (B, num_anchor) int64, -1 padded
    temp_confidence: torch.Tensor  # (B, num_anchor) float — same semantics as ``InstanceBank.cache``


@dataclass
class PackedTemporalMemory:
    """All three banks; use zero-sized Kt when a bank has num_temp_instances==0."""

    det: Optional[PackedInstanceBankMemory] = None
    map_: Optional[PackedInstanceBankMemory] = None
    ego: Optional[PackedInstanceBankMemory] = None


def _batched_eye4(batch_size: int, ref: torch.Tensor) -> torch.Tensor:
    """``(B, 4, 4)`` identity matrices, same device/dtype as ``ref``."""
    e = torch.eye(4, device=ref.device, dtype=ref.dtype)
    return e.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _batched_eye4_pose(batch_size: int, device: torch.device) -> torch.Tensor:
    """``(B, 4, 4)`` identity for stored pose tensors (float64)."""
    e = torch.eye(4, device=device, dtype=_POSE_DTYPE)
    return e.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def cold_packed_memory_like_unpack_none(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    embed_dims: int,
    det_anchor_dim: int,
    map_anchor_dim: int,
    ego_anchor_dim: int,
) -> PackedTemporalMemory:
    """Cold-start packed memory for stateless injection.

    **Map** bank uses ``(B, 0, *)`` caches like :func:`unpack_packed_instance_bank_memory` with
    ``pb is None`` (``use_temp`` false). **Det** bank uses **600** temporal slots (zeros) so the det
    cache tensor shape matches the model's det ``Kt``; poses and ``timestamp_prev`` are float64.
    **Ego** uses one slot. This differs from :func:`empty_packed_memory`, which allocates full
    ``Kt`` for every bank.
    """

    def one_bank(anchor_dim: int, size: int) -> PackedInstanceBankMemory:
        z = torch.zeros(batch_size, device=device, dtype=dtype)
        eye_pose = _batched_eye4_pose(batch_size, device)
        return PackedInstanceBankMemory(
            valid=z.clone(),
            cached_feature=torch.zeros(
                batch_size, size, embed_dims, device=device, dtype=dtype
            ),
            cached_anchor=torch.zeros(
                batch_size, size, anchor_dim, device=device, dtype=dtype
            ),
            confidence=torch.zeros(batch_size, size, device=device, dtype=dtype),
            timestamp_prev=torch.zeros(batch_size, device=device, dtype=_TS_DTYPE),
            T_global_prev=eye_pose.clone(),
            T_global_inv_prev=eye_pose.clone(),
        )

    return PackedTemporalMemory(
        det=one_bank(det_anchor_dim, size=600),
        map_=one_bank(map_anchor_dim, size=0),
        ego=one_bank(ego_anchor_dim, size=1),
    )


def cold_det_tracking_state(
    batch_size: int,
    num_anchor: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedDetTrackingState:
    """Initial detection instance-ID carry state (matches ``det_tracking_in is None`` branch in stage2)."""
    return PackedDetTrackingState(
        prev_id=torch.zeros((), device=device, dtype=torch.long),
        instance_id=torch.full(
            (batch_size, num_anchor), -1, device=device, dtype=torch.long
        ),
        temp_confidence=torch.zeros(batch_size, num_anchor, device=device, dtype=dtype),
    )


def empty_packed_memory(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    det_Kt: int,
    map_Kt: int,
    ego_Kt: int,
    embed_dims: int,
    det_anchor_dim: int,
    map_anchor_dim: int,
    ego_anchor_dim: int,
) -> PackedTemporalMemory:
    """Fresh memory with **full** ``Kt``-sized cache tensors (zeros).

    Prefer :func:`cold_packed_memory_like_unpack_none` for cold start when you need the same
    behavior as ``packed_memory is None`` / unpack without history (empty temporal dim).
    """

    def pb(Kt: int, adim: int) -> PackedInstanceBankMemory:
        zt = torch.zeros(batch_size, device=device, dtype=dtype)
        eye_pose = _batched_eye4_pose(batch_size, device)
        return PackedInstanceBankMemory(
            valid=torch.zeros(batch_size, device=device, dtype=dtype),
            cached_feature=torch.zeros(batch_size, Kt, embed_dims, device=device, dtype=dtype),
            cached_anchor=torch.zeros(batch_size, Kt, adim, device=device, dtype=dtype),
            confidence=torch.zeros(batch_size, Kt, device=device, dtype=dtype),
            timestamp_prev=torch.zeros(batch_size, device=device, dtype=_TS_DTYPE),
            T_global_prev=eye_pose.clone(),
            T_global_inv_prev=eye_pose.clone(),
        )

    return PackedTemporalMemory(
        det=pb(det_Kt, det_anchor_dim),
        map_=pb(map_Kt, map_anchor_dim),
        ego=pb(ego_Kt, ego_anchor_dim),
    )


def metas_snapshot_tensors_from_metas(metas: Dict[str, Any], ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract batched ``timestamp`` (float64) and ego poses (same stacking as functional banks)."""
    from plugin.models.functional_instance_bank import (
        img_metas_list_to_pose_tensors,
    )

    ts = metas["timestamp"]
    if not torch.is_tensor(ts):
        ts = torch.as_tensor(ts, device=ref.device, dtype=_TS_DTYPE)
    else:
        ts = ts.to(device=ref.device, dtype=_TS_DTYPE)
    tg, ti = img_metas_list_to_pose_tensors(
        metas["img_metas"], ref, pose_dtype=_POSE_DTYPE
    )
    return {"timestamp": ts, "T_global": tg, "T_global_inv": ti}


def unpack_packed_instance_bank_memory(
    pb: Optional[PackedInstanceBankMemory],
    batch_size: int,
    ref: torch.Tensor,
    Kt: int,
    embed_dims: int,
    anchor_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack packed bank tensors for :func:`functional_instance_bank_get` / ego get.

    Always returns seven tensors. When ``pb is None`` or ``Kt == 0``, returns empty
    cache slots ``(B, 0, *)`` and ``valid`` all zeros (cold start / inactive bank).

    The ``valid`` tensor (row-wise 0/1) is passed as ``temporal_valid`` into
    :func:`~projects.mmdet3d_plugin.models.functional_instance_bank.functional_instance_bank_get`
    and combined with the time window: ``mask = mask_time & temporal_valid``.
    Rows with ``valid == 0`` skip temporal fusion even if cache tensors
    are non-empty (e.g. empty-packed-memory warmup shapes).
    """
    z = torch.zeros(batch_size, device=ref.device, dtype=ref.dtype)
    z_ts = torch.zeros(batch_size, device=ref.device, dtype=_TS_DTYPE)
    eye_pose = _batched_eye4_pose(batch_size, ref.device)

    if pb is None or Kt <= 0:
        return (
            torch.zeros(batch_size, 0, embed_dims, device=ref.device, dtype=ref.dtype),
            torch.zeros(batch_size, 0, anchor_dim, device=ref.device, dtype=ref.dtype),
            z_ts,
            eye_pose,
            eye_pose,
            z,
            torch.zeros(batch_size, 0, device=ref.device, dtype=ref.dtype),
        )

    valid = pb.valid.to(device=ref.device, dtype=ref.dtype)
    cf = pb.cached_feature.to(device=ref.device, dtype=ref.dtype)
    ca = pb.cached_anchor.to(device=ref.device, dtype=ref.dtype)
    ts = pb.timestamp_prev.to(device=ref.device, dtype=_TS_DTYPE)
    tg = pb.T_global_prev.to(device=ref.device, dtype=_POSE_DTYPE)
    tgi = pb.T_global_inv_prev.to(device=ref.device, dtype=_POSE_DTYPE)
    conf = pb.confidence.to(device=ref.device, dtype=ref.dtype)
    if cf.shape[1] == 0:
        return (
            torch.zeros(batch_size, 0, embed_dims, device=ref.device, dtype=ref.dtype),
            torch.zeros(batch_size, 0, anchor_dim, device=ref.device, dtype=ref.dtype),
            z_ts,
            eye_pose,
            eye_pose,
            z,
            torch.zeros(batch_size, 0, device=ref.device, dtype=ref.dtype),
        )
    return cf, ca, ts, tg, tgi, valid, conf


def expand_pose_to_batch(
    T: torch.Tensor, batch_size: int, ref: torch.Tensor
) -> torch.Tensor:
    if T.dim() == 2:
        return T.unsqueeze(0).expand(batch_size, -1, -1).to(device=ref.device, dtype=ref.dtype)
    return T.to(device=ref.device, dtype=ref.dtype)
