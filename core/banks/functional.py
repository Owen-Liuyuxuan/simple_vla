"""models/functional_instance_bank.py → banks/functional.py

Stateless functional instance bank — tensor-only API without mutating ``self.cached_*``.

Core idea: All temporal state is explicit tensors passed as args/returned as outputs.
Callers (PackedTemporalMemory) own and thread the memory tensors across frames.

This module is pure PyTorch, zero mmcv/mmdet dependency.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def topk(confidence, k, *inputs):
    """Helper: top-k selection with batched indexing."""
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1)
    outputs = []
    for inp in inputs:
        outputs.append(inp.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


def img_metas_list_to_pose_tensors(img_metas, ref):
    """Stack T_global and T_global_inv from img_metas."""
    tgs, tis = [], []
    for im in img_metas:
        tg = im["T_global"]
        ti = im["T_global_inv"]
        if not torch.is_tensor(tg):
            tg = torch.as_tensor(tg, device=ref.device, dtype=ref.dtype)
        else:
            tg = tg.to(device=ref.device, dtype=ref.dtype)
        if not torch.is_tensor(ti):
            ti = torch.as_tensor(ti, device=ref.device, dtype=ref.dtype)
        else:
            ti = ti.to(device=ref.device, dtype=ref.dtype)
        tgs.append(tg)
        tis.append(ti)
    return torch.stack(tgs, dim=0), torch.stack(tis, dim=0)


def compute_T_temp2cur_batched(T_global_inv_cur, T_global_prev):
    """Warp from temporal → current frame."""
    return torch.bmm(T_global_inv_cur, T_global_prev)


def _empty_det_temp_tensors(embed_dims, anchor_dim, batch_size, device, dtype):
    return (
        torch.zeros(batch_size, 0, embed_dims, device=device, dtype=dtype),
        torch.zeros(batch_size, 0, anchor_dim, device=device, dtype=dtype),
    )


def _as_bool_b(temporal_valid, batch_size, ref):
    v = temporal_valid.to(device=ref.device)
    if v.dtype != torch.bool:
        v = v != 0
    if v.dim() == 0:
        v = v.expand(batch_size)
    return v.reshape(batch_size)


def _combine_masks(mask_time, temporal_valid, batch_size, ref):
    return mask_time & _as_bool_b(temporal_valid, batch_size, ref)


class FunctionalInstanceBank:
    """Stateless wrapper around a *configured* InstanceBank module.

    Parameters
    ----------
    bank: torch.nn.Module
        A pre-built ``InstanceBank`` module (from ``PLUGIN_LAYERS``).
        Its parameters (anchor, instance_feature) are frozen during functional calls.
    """

    def __init__(self, bank):
        self.bank = bank
        self.embed_dims = bank.embed_dims
        self.num_anchor = bank.num_anchor
        self.num_temp_instances = bank.num_temp_instances
        self.default_time_interval = bank.default_time_interval
        self.max_time_interval = bank.max_time_interval
        self.confidence_decay = bank.confidence_decay
        self.anchor_handler = bank.anchor_handler

    def get_from_tensors(
        self,
        batch_size: int,
        ts_cur: torch.Tensor,
        T_global_inv_cur: torch.Tensor,
        cached_feature: torch.Tensor,
        cached_anchor: torch.Tensor,
        timestamp_prev: torch.Tensor,
        T_global_prev: torch.Tensor,
        temporal_valid: torch.Tensor,
        dn_metas: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure-tensor version of ``InstanceBank.get``.

        Returns
        -------
        instance_feature : (B, Na, C)
        anchor          : (B, Na, A)
        temp_feature    : (B, Kt, C) or (B, 0, C)
        temp_anchor     : (B, Kt, A) or (B, 0, A)
        time_interval   : (B,) float
        mask            : (B,) bool — which batch rows use temporal fusion
        """
        bank = self.bank
        instance_feature = torch.tile(bank.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(bank.anchor[None], (batch_size, 1, 1))
        ref = instance_feature

        ts_cur = ts_cur.to(device=ref.device, dtype=torch.float64)

        use_temp = (
            cached_feature.shape[1] > 0
            and cached_anchor.shape[1] > 0
            and batch_size == cached_anchor.shape[0]
        )

        if not use_temp:
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
            mask = torch.zeros(batch_size, device=ref.device, dtype=torch.bool)
            etf, eta = _empty_det_temp_tensors(self.embed_dims, anchor.shape[-1], batch_size, ref.device, ref.dtype)
            return instance_feature, anchor, etf, eta, time_interval, mask

        history_time = timestamp_prev.to(device=ref.device, dtype=torch.float64)
        time_interval = ts_cur - history_time
        time_interval = time_interval.to(dtype=instance_feature.dtype)
        mask_time = torch.abs(time_interval) <= self.max_time_interval

        mask = _combine_masks(mask_time, temporal_valid, batch_size, ref)

        T_global_prev_b = T_global_prev.to(device=cached_anchor.device, dtype=cached_anchor.dtype)
        T_global_inv_cur_b = T_global_inv_cur.to(device=cached_anchor.device, dtype=cached_anchor.dtype)
        T_temp2cur = compute_T_temp2cur_batched(T_global_inv_cur_b, T_global_prev_b)

        warped_anchor = cached_anchor
        if self.anchor_handler is not None:
            warped_anchor = self.anchor_handler.anchor_projection(
                cached_anchor, [T_temp2cur], time_intervals=[-time_interval]
            )[0]

        if (
            self.anchor_handler is not None
            and dn_metas is not None
            and batch_size == dn_metas["dn_anchor"].shape[0]
            and "img_metas" in dn_metas
        ):
            num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
            dn_anchor = self.anchor_handler.anchor_projection(
                dn_metas["dn_anchor"].flatten(1, 2), [T_temp2cur], time_intervals=[-time_interval]
            )[0]
            dn_metas["dn_anchor"] = dn_anchor.reshape(batch_size, num_dn_group, num_dn, -1)

        time_interval = torch.where(
            torch.logical_and(time_interval != 0, mask),
            time_interval,
            time_interval.new_tensor(self.default_time_interval),
        )

        return instance_feature, anchor, cached_feature, warped_anchor, time_interval, mask

    def update(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        cached_feature: torch.Tensor,
        cached_anchor: torch.Tensor,
        mask: torch.Tensor,
        prev_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional update — returns new (instance_feature, anchor, new_confidence)."""
        if cached_feature.shape[1] == 0:
            return instance_feature, anchor, prev_confidence

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        conf_vals = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(conf_vals, N, instance_feature, anchor)
        selected_feature = torch.cat([cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([cached_anchor, selected_anchor], dim=1)

        instance_feature = torch.where(mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(mask[:, None, None], selected_anchor, anchor)

        new_conf = torch.where(mask[:, None], prev_confidence, prev_confidence.new_tensor(0))

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        return instance_feature, anchor, new_conf

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        prev_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cached_feature, cached_anchor, confidence_topk, confidence_full)."""
        if self.num_temp_instances <= 0:
            B = instance_feature.shape[0]
            dev, dt = instance_feature.device, instance_feature.dtype
            empty = torch.zeros(B, 0, self.embed_dims, device=dev, dtype=dt)
            conf_full = torch.zeros(B, self.num_anchor, device=dev, dtype=dt)
            return empty, torch.zeros(B, 0, anchor.shape[-1], device=dev, dtype=dt), empty, conf_full

        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        conf = confidence.max(dim=-1).values.sigmoid()
        if prev_confidence.shape[1] > 0:
            conf[:, : self.num_temp_instances] = torch.maximum(
                prev_confidence * self.confidence_decay,
                conf[:, : self.num_temp_instances],
            )

        conf_out, (cached_feature, cached_anchor) = topk(conf, self.num_temp_instances, instance_feature, anchor)
        return cached_feature, cached_anchor, conf_out, conf
