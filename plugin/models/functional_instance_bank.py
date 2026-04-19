# Copyright (c) OpenMMLab. All rights reserved.
"""Single-path functional instance bank: ``get`` / ``update`` / ``cache`` without ``self.cached_*``.

**Functional memory (stateless inference)** — all temporal state is **explicit tensors** passed in and
returned; callers (e.g. :class:`~projects.mmdet3d_plugin.unidrivevla.wrappers.packed_temporal_memory.PackedTemporalMemory`)
initialize, pack, and thread memory **outside** the model forward. The forward only sees packed I/O.

**Tensor contract (det/map bank)**

- **get** (see :func:`functional_instance_bank_get_from_tensors`):

  - ``ts_cur``: ``(B,)`` current timestamps (float64 for subtraction; see wrappers).
  - ``T_global_inv_cur``, ``T_global_prev``: ``(B, 4, 4)`` float64 in packed memory / metas;
    warp uses ``bmm`` in float64, then ``T_temp2cur`` is cast to anchor dtype for projection.
  - ``cached_feature``, ``cached_anchor``: ``(B, Kt, *)`` — previous cache; **cold** = ``Kt == 0`` (empty dim 1).
  - ``timestamp_prev``: ``(B,)`` float64 timestamps stored with the cache.
  - ``T_global_inv_prev``: ``(B, 4, 4)`` — unused in warp (symmetry with packed layout).
  - ``temporal_valid``: ``(B,)`` bool — per-row validity; AND with time window.

  Returns ``mask`` = ``mask_time & temporal_valid``; use the same ``mask`` in **update**.

- **update**: ``cached_feature.shape[1] == 0`` skips merge (matches :class:`~projects.mmdet3d_plugin.models.instance_bank.InstanceBank` when no cache). ``prev_confidence`` is always a tensor (use ``(B, 0)`` when no prior top-k conf).

- **cache**: top-k temporal slots; ``metas`` required (mutable bank parity); ``prev_confidence`` tensor (``(B, 0)`` when none).

Pose warps use ``torch.bmm``. Only ``dn_metas`` may be ``None`` (inference); training may supply DN dicts.

**Cold start:** ``Kt == 0`` **or** ``temporal_valid`` all false — no temporal fusion in **update** when
cache is empty.

**Detection instance IDs:** :func:`functional_det_instance_id_forward` — ONNX not in scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from simple_vla.plugin.models.instance_bank import topk as bank_topk
from simple_vla.plugin.unidrivevla.wrappers.packed_temporal_memory import (
    PackedDetTrackingState,
)


def img_metas_list_to_pose_tensors(
    img_metas: List[Dict[str, Any]],
    ref: torch.Tensor,
    pose_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack ``T_global`` and ``T_global_inv`` per batch row from nuScenes-style ``img_metas``.

    Default ``pose_dtype`` matches ``ref`` (mutable-bank style). Stateless snapshots and packed
    memory pass ``torch.float64`` so global poses stay in float64; only ``T_temp2cur`` is cast to
    activations dtype in :func:`functional_instance_bank_get_from_tensors`.
    """
    pd = pose_dtype if pose_dtype is not None else ref.dtype
    tgs = []
    tis = []
    for im in img_metas:
        tg = im["T_global"]
        ti = im["T_global_inv"]
        if not torch.is_tensor(tg):
            tg = torch.as_tensor(tg, device=ref.device, dtype=pd)
        else:
            tg = tg.to(device=ref.device, dtype=pd)
        if not torch.is_tensor(ti):
            ti = torch.as_tensor(ti, device=ref.device, dtype=pd)
        else:
            ti = ti.to(device=ref.device, dtype=pd)
        tgs.append(tg)
        tis.append(ti)
    return torch.stack(tgs, dim=0), torch.stack(tis, dim=0)


def compute_T_temp2cur_batched(
    T_global_inv_cur: torch.Tensor,
    T_global_prev: torch.Tensor,
) -> torch.Tensor:
    """``T_temp2cur[i] = T_global_inv_cur[i] @ T_global_prev[i]`` (matches ``InstanceBank.get``).

    Callers should pass float64 poses and cast the result to anchor dtype for ``anchor_projection``.
    """
    return torch.bmm(T_global_inv_cur, T_global_prev)


@dataclass
class FunctionalInstanceBankGetResult:
    instance_feature: torch.Tensor
    anchor: torch.Tensor
    temp_feature: torch.Tensor
    temp_anchor: torch.Tensor
    time_interval: torch.Tensor
    mask: torch.Tensor


def _empty_det_temp_tensors(bank, batch_size: int, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, 0, C)`` / ``(B, 0, A)`` when there is no temporal slot (export-friendly)."""
    return (
        torch.zeros(
            batch_size, 0, bank.embed_dims, device=ref.device, dtype=ref.dtype
        ),
        torch.zeros(
            batch_size, 0, bank.anchor.shape[-1], device=ref.device, dtype=ref.dtype
        ),
    )


def _as_bool_b(temporal_valid: torch.Tensor, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
    """``(B,)`` bool from 0/1 float or bool."""
    v = temporal_valid.to(device=ref.device)
    if v.dtype != torch.bool:
        v = v != 0
    if v.dim() == 0:
        v = v.expand(batch_size)
    return v.reshape(batch_size)


def _combine_masks(
    mask_time: torch.Tensor,
    temporal_valid: torch.Tensor,
    batch_size: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    """``mask = mask_time & temporal_valid``."""
    return mask_time & _as_bool_b(temporal_valid, batch_size, ref)


def functional_instance_bank_get_from_tensors(
    bank,
    batch_size: int,
    ts_cur: torch.Tensor,
    T_global_inv_cur: torch.Tensor,
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    timestamp_prev: torch.Tensor,
    T_global_prev: torch.Tensor,
    T_global_inv_prev: torch.Tensor,
    temporal_valid: torch.Tensor,
    metas_for_dn: Dict[str, Any],
    dn_metas: Optional[Dict[str, Any]] = None,
) -> FunctionalInstanceBankGetResult:
    """Tensor-only ``InstanceBank.get`` — single path; ``temporal_valid`` is required ``(B,)``.

    ``T_global_inv_prev`` is accepted for packed-memory symmetry; unused in warp.
    ``metas_for_dn`` must include ``img_metas`` when DN anchor projection runs.
    """
    _ = T_global_inv_prev
    instance_feature = torch.tile(bank.instance_feature[None], (batch_size, 1, 1))
    anchor = torch.tile(bank.anchor[None], (batch_size, 1, 1))
    ref = instance_feature

    ts_cur = ts_cur.to(device=ref.device, dtype=torch.float64)

    use_temp = (
        cached_feature.shape[1] > 0
        and cached_anchor.shape[1] > 0
        and batch_size == cached_anchor.shape[0]
    ) ## TODO: for now use_temp is False because the cold start feature has a shape of (B, 0, C)

    if not use_temp:
        time_interval = instance_feature.new_tensor(
            [bank.default_time_interval] * batch_size
        )
        mask = torch.zeros(batch_size, device=ref.device, dtype=torch.bool)
        etf, eta = _empty_det_temp_tensors(bank, batch_size, ref)
        return FunctionalInstanceBankGetResult(
            instance_feature=instance_feature,
            anchor=anchor,
            temp_feature=etf,
            temp_anchor=eta,
            time_interval=time_interval,
            mask=mask,
        )

    history_time = timestamp_prev.to(device=ref.device, dtype=torch.float64)
    time_interval = ts_cur - history_time
    time_interval = time_interval.to(dtype=instance_feature.dtype)
    mask_time = torch.abs(time_interval) <= bank.max_time_interval

    mask = _combine_masks(mask_time, temporal_valid, batch_size, ref)

    dev = cached_anchor.device
    T_prev_64 = T_global_prev.to(device=dev, dtype=torch.float64)
    T_inv_cur_64 = T_global_inv_cur.to(device=dev, dtype=torch.float64)
    T_temp2cur = compute_T_temp2cur_batched(T_inv_cur_64, T_prev_64)
    T_temp2cur = T_temp2cur.to(dtype=cached_anchor.dtype)

    warped_anchor = cached_anchor

    dn_metas_dict = metas_for_dn

    if bank.anchor_handler is not None:
        warped_anchor = bank.anchor_handler.anchor_projection(
            cached_anchor, [T_temp2cur], time_intervals=[-time_interval]
        )[0]

    if (
        bank.anchor_handler is not None
        and dn_metas is not None
        and batch_size == dn_metas["dn_anchor"].shape[0]
        and "img_metas" in dn_metas_dict
    ):
        num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
        dn_anchor = bank.anchor_handler.anchor_projection(
            dn_metas["dn_anchor"].flatten(1, 2), [T_temp2cur], time_intervals=[-time_interval]
        )[0]
        dn_metas["dn_anchor"] = dn_anchor.reshape(batch_size, num_dn_group, num_dn, -1)

    time_interval = torch.where(
        torch.logical_and(time_interval != 0, mask),
        time_interval,
        time_interval.new_tensor(bank.default_time_interval),
    )

    return FunctionalInstanceBankGetResult(
        instance_feature=instance_feature,
        anchor=anchor,
        temp_feature=cached_feature,
        temp_anchor=warped_anchor,
        time_interval=time_interval,
        mask=mask,
    )


def functional_instance_bank_get(
    bank,
    batch_size: int,
    metas: Dict[str, Any],
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    timestamp_prev: torch.Tensor,
    T_global_prev: torch.Tensor,
    T_global_inv_prev: torch.Tensor,
    temporal_valid: torch.Tensor,
    dn_metas: Optional[Dict[str, Any]] = None,
) -> FunctionalInstanceBankGetResult:
    """Thin wrapper: stacks poses / time from ``metas``, then :func:`functional_instance_bank_get_from_tensors`."""
    ref = bank.instance_feature
    ts_cur = metas["timestamp"]
    if not torch.is_tensor(ts_cur):
        ts_cur = torch.as_tensor(ts_cur, device=ref.device, dtype=torch.float64)
    else:
        ts_cur = ts_cur.to(device=ref.device, dtype=torch.float64)

    _, T_global_inv_cur = img_metas_list_to_pose_tensors(
        metas["img_metas"], ref, pose_dtype=torch.float64
    )

    return functional_instance_bank_get_from_tensors(
        bank,
        batch_size,
        ts_cur,
        T_global_inv_cur,
        cached_feature,
        cached_anchor,
        timestamp_prev,
        T_global_prev,
        T_global_inv_prev,
        temporal_valid,
        metas,
        dn_metas=dn_metas,
    )


def functional_instance_bank_update(
    bank,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    confidence: torch.Tensor,
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    mask: torch.Tensor,
    prev_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional ``InstanceBank.update`` (no side effects on ``bank``).

    ``mask`` must match :func:`functional_instance_bank_get` / :func:`functional_instance_bank_get_from_tensors``.
    ``prev_confidence`` is ``(B, Kt)`` or ``(B, 0)`` when cold.
    """
    if cached_feature.shape[1] == 0:
        return instance_feature, anchor, prev_confidence

    num_dn = 0
    if instance_feature.shape[1] > bank.num_anchor:
        num_dn = instance_feature.shape[1] - bank.num_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, : bank.num_anchor]
        anchor = anchor[:, : bank.num_anchor]
        confidence = confidence[:, : bank.num_anchor]

    N = bank.num_anchor - bank.num_temp_instances
    conf_vals = confidence.max(dim=-1).values
    _, (selected_feature, selected_anchor) = bank_topk(conf_vals, N, instance_feature, anchor)
    selected_feature = torch.cat([cached_feature, selected_feature], dim=1)
    selected_anchor = torch.cat([cached_anchor, selected_anchor], dim=1)

    m3 = mask[:, None, None]
    instance_feature = torch.where(m3, selected_feature, instance_feature)
    anchor = torch.where(m3, selected_anchor, anchor)

    new_conf = torch.where(
        mask[:, None],
        prev_confidence,
        prev_confidence.new_tensor(0),
    )

    if num_dn > 0:
        instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
        anchor = torch.cat([anchor, dn_anchor], dim=1)

    return instance_feature, anchor, new_conf


def functional_det_instance_id_forward(
    bank,
    confidence_logits: torch.Tensor,
    prev_id: torch.Tensor,
    instance_id_prev: torch.Tensor,
    temp_confidence_full: torch.Tensor,
    apply_score_threshold: bool,
    score_threshold: float,
) -> Tuple[torch.Tensor, PackedDetTrackingState]:
    """Tensor-only equivalent of ``InstanceBank.get_instance_id`` + ``update_instance_id``.

    ONNX export of dynamic instance-id assignment is not in scope; see planning docs.
    """
    confidence = confidence_logits.max(dim=-1).values.sigmoid()
    instance_id = confidence.new_full(confidence.shape, -1).long()
    n_prev = min(instance_id_prev.shape[1], instance_id.shape[1])
    if n_prev > 0:
        instance_id[:, :n_prev] = instance_id_prev[:, :n_prev]

    mask = instance_id < 0
    if apply_score_threshold:
        mask = mask & (confidence >= score_threshold)
    num_new_instance = mask.sum()
    pid = int(prev_id.item())
    new_ids = torch.arange(int(num_new_instance.item()), device=instance_id.device, dtype=instance_id.dtype) + pid
    instance_id[torch.where(mask)] = new_ids
    prev_id_next = pid + int(num_new_instance.item())

    if temp_confidence_full.shape == confidence.shape:
        temp_conf = temp_confidence_full
    else:
        temp_conf = confidence

    instance_id_k = bank_topk(temp_conf, bank.num_temp_instances, instance_id)[1][0]
    instance_id_k = instance_id_k.squeeze(dim=-1)
    instance_id_stored = F.pad(
        instance_id_k,
        (0, bank.num_anchor - bank.num_temp_instances),
        value=-1,
    )
    out = PackedDetTrackingState(
        prev_id=torch.tensor(prev_id_next, device=confidence.device, dtype=torch.long),
        instance_id=instance_id_stored,
        temp_confidence=temp_confidence_full,
    )
    return instance_id, out


def functional_instance_bank_cache(
    bank,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    confidence: torch.Tensor,
    prev_confidence: torch.Tensor,
    metas: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional ``InstanceBank.cache``.

    Returns:
        ``cached_feature``, ``cached_anchor``, ``confidence_topk``, ``confidence_full`` — full-grid
        ``(B, num_anchor)`` sigmoid confidences after temporal decay (for instance-ID tracking).
    """
    if bank.num_temp_instances <= 0:
        z = instance_feature.shape[0]
        dev = instance_feature.device
        dt = instance_feature.dtype
        empty = torch.zeros(z, 0, bank.embed_dims, device=dev, dtype=dt)
        N = bank.num_anchor
        conf_full = torch.zeros(z, N, device=dev, dtype=dt)
        return (
            empty,
            torch.zeros(z, 0, anchor.shape[-1], device=dev, dtype=dt),
            torch.zeros(z, 0, device=dev, dtype=dt),
            conf_full,
        )

    instance_feature = instance_feature.detach()
    anchor = anchor.detach()
    confidence = confidence.detach()

    conf = confidence.max(dim=-1).values.sigmoid()
    if prev_confidence.shape[1] > 0:
        conf[:, : bank.num_temp_instances] = torch.maximum(
            prev_confidence * bank.confidence_decay,
            conf[:, : bank.num_temp_instances],
        )

    conf_out, (cached_feature, cached_anchor) = bank_topk(
        conf, bank.num_temp_instances, instance_feature, anchor
    )
    _ = metas
    return cached_feature, cached_anchor, conf_out, conf


@dataclass
class FunctionalEgoBankGetResult:
    instance_feature: torch.Tensor
    anchor: torch.Tensor
    temp_feature: torch.Tensor
    temp_anchor: torch.Tensor
    mask: torch.Tensor


def _empty_ego_temp_tensors(ego_bank, batch_size: int, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(
            batch_size, 0, ego_bank.embed_dims, device=ref.device, dtype=ref.dtype
        ),
        torch.zeros(
            batch_size, 0, ego_bank.anchor.shape[-1], device=ref.device, dtype=ref.dtype
        ),
    )


def functional_ego_instance_bank_get_from_tensors(
    ego_bank,
    batch_size: int,
    ts_cur: torch.Tensor,
    T_global_inv_cur: torch.Tensor,
    feature_maps,
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    timestamp_prev: torch.Tensor,
    T_global_prev: torch.Tensor,
    T_global_inv_prev: torch.Tensor,
    temporal_valid: torch.Tensor,
    metas_for_dn: Dict[str, Any],
    dn_metas: Optional[Dict[str, Any]] = None,
) -> FunctionalEgoBankGetResult:
    """Tensor-only ``EgoInstanceBank.get`` — ``temporal_valid`` ``(B,)`` required."""
    _ = T_global_inv_prev
    instance_feature, anchor = ego_bank.prepare_ego(batch_size, feature_maps)
    ref = anchor
    ts_cur = ts_cur.to(device=ref.device, dtype=torch.float64)

    use_temp = (
        cached_feature.shape[1] > 0
        and cached_anchor.shape[1] > 0
        and batch_size == cached_anchor.shape[0]
    ) ## TODO: for now use_temp is False because the cold start feature has a shape of (B, 0, C)

    if not use_temp:
        etf, eta = _empty_ego_temp_tensors(ego_bank, batch_size, ref)
        return FunctionalEgoBankGetResult(
            instance_feature=instance_feature,
            anchor=anchor,
            temp_feature=etf,
            temp_anchor=eta,
            mask=torch.zeros(batch_size, device=ref.device, dtype=torch.bool),
        )

    history_time = timestamp_prev.to(device=ref.device, dtype=torch.float64)
    time_interval = ts_cur - history_time
    time_interval = time_interval.to(dtype=instance_feature.dtype)
    mask_time = torch.abs(time_interval) <= ego_bank.max_time_interval

    mask = _combine_masks(mask_time, temporal_valid, batch_size, ref)

    dev = cached_anchor.device
    T_prev_64 = T_global_prev.to(device=dev, dtype=torch.float64)
    T_inv_cur_64 = T_global_inv_cur.to(device=dev, dtype=torch.float64)
    T_temp2cur = compute_T_temp2cur_batched(T_inv_cur_64, T_prev_64)
    T_temp2cur = T_temp2cur.to(dtype=cached_anchor.dtype)

    warped = cached_anchor
    dn_metas_dict = metas_for_dn

    if ego_bank.anchor_handler is not None:
        warped = ego_bank.anchor_handler.anchor_projection(
            cached_anchor, [T_temp2cur], time_intervals=[-time_interval]
        )[0]

    if (
        ego_bank.anchor_handler is not None
        and dn_metas is not None
        and batch_size == dn_metas["dn_anchor"].shape[0]
        and "img_metas" in dn_metas_dict
    ):
        num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
        dn_anchor = ego_bank.anchor_handler.anchor_projection(
            dn_metas["dn_anchor"].flatten(1, 2), [T_temp2cur], time_intervals=[-time_interval]
        )[0]
        dn_metas["dn_anchor"] = dn_anchor.reshape(batch_size, num_dn_group, num_dn, -1)

    return FunctionalEgoBankGetResult(
        instance_feature=instance_feature,
        anchor=anchor,
        temp_feature=cached_feature,
        temp_anchor=warped,
        mask=mask,
    )


def functional_ego_instance_bank_get(
    ego_bank,
    batch_size: int,
    metas: Dict[str, Any],
    feature_maps,
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    timestamp_prev: torch.Tensor,
    T_global_prev: torch.Tensor,
    T_global_inv_prev: torch.Tensor,
    temporal_valid: torch.Tensor,
    dn_metas: Optional[Dict[str, Any]] = None,
) -> FunctionalEgoBankGetResult:
    """Thin wrapper for dict ``metas``; then :func:`functional_ego_instance_bank_get_from_tensors`."""
    ref = ego_bank.anchor
    ts_cur = metas["timestamp"]
    if not torch.is_tensor(ts_cur):
        ts_cur = torch.as_tensor(ts_cur, device=ref.device, dtype=torch.float64)
    else:
        ts_cur = ts_cur.to(device=ref.device, dtype=torch.float64)

    _, T_global_inv_cur = img_metas_list_to_pose_tensors(
        metas["img_metas"], ref, pose_dtype=torch.float64
    )

    return functional_ego_instance_bank_get_from_tensors(
        ego_bank,
        batch_size,
        ts_cur,
        T_global_inv_cur,
        feature_maps,
        cached_feature,
        cached_anchor,
        timestamp_prev,
        T_global_prev,
        T_global_inv_prev,
        temporal_valid,
        metas,
        dn_metas=dn_metas,
    )


def functional_ego_instance_bank_update(
    ego_bank,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    confidence: torch.Tensor,
    cached_feature: torch.Tensor,
    cached_anchor: torch.Tensor,
    mask: torch.Tensor,
    prev_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cached_feature.shape[1] == 0:
        return instance_feature, anchor, prev_confidence

    num_dn = 0
    if instance_feature.shape[1] > ego_bank.num_anchor:
        num_dn = instance_feature.shape[1] - ego_bank.num_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, : ego_bank.num_anchor]
        anchor = anchor[:, : ego_bank.num_anchor]
        confidence = confidence[:, : ego_bank.num_anchor]

    N = ego_bank.num_anchor - ego_bank.num_temp_instances
    conf_vals = confidence.max(dim=-1).values
    _, (selected_feature, selected_anchor) = bank_topk(conf_vals, N, instance_feature, anchor)
    selected_feature = torch.cat([cached_feature, selected_feature], dim=1)
    selected_anchor = torch.cat([cached_anchor, selected_anchor], dim=1)

    m3 = mask[:, None, None]
    instance_feature = torch.where(m3, selected_feature, instance_feature)
    anchor = torch.where(m3, selected_anchor, anchor)

    new_conf = torch.where(mask[:, None], prev_confidence, prev_confidence.new_tensor(0))

    if num_dn > 0:
        instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
        anchor = torch.cat([anchor, dn_anchor], dim=1)

    return instance_feature, anchor, new_conf


def functional_ego_instance_bank_cache(
    ego_bank,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    metas: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ego_bank.num_temp_instances <= 0:
        z = instance_feature.shape[0]
        dev = instance_feature.device
        dt = instance_feature.dtype
        _ = metas
        return (
            torch.zeros(z, 0, ego_bank.embed_dims, device=dev, dtype=dt),
            torch.zeros(z, 0, anchor.shape[-1], device=dev, dtype=dt),
        )
    _ = metas
    return instance_feature.detach(), anchor.detach()
