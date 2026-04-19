# Copyright 2026 The Xiaomi Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""CUDA / SDPA runtime tweaks for numerical stability across PyTorch builds."""
from __future__ import annotations

import os

import torch


def maybe_configure_cuda_sdp() -> None:
    """Apply optional SDPA backend overrides before model forward.

    Environment variables:

    - ``TORCH_DISABLE_MEM_EFFICIENT_SDPA``: if ``1`` / ``true`` / ``yes``, calls
      ``torch.backends.cuda.enable_mem_efficient_sdp(False)``.
    - ``TORCH_DISABLE_MEM_EFFICIENT_SDPA=0`` explicitly keeps the default (even if
      auto-detection would disable it).

    **Auto-detection** (when the env var is unset): on **PyTorch 2.4.x** with
    **CUDA 12.4** wheels, ``scaled_dot_product_attention`` can emit **NaNs**
    when the **memory-efficient** SDPA backend handles **bf16** tensors with a
    non-null ``attn_mask``. Disabling only that backend forces math or flash
    attention (subject to mask support), restoring finite outputs. PyTorch
    2.5.x + cu121 does not show the same failure for the same tensors.

    This is a small performance trade-off on affected stacks only.
    """
    if not torch.cuda.is_available():
        return

    explicit = os.environ.get("TORCH_DISABLE_MEM_EFFICIENT_SDPA")
    if explicit is not None:
        v = explicit.strip().lower()
        if v in ("0", "false", "no", "off"):
            return
        if v in ("1", "true", "yes", "on"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            return

    ver = torch.__version__.split("+")[0]
    cuda = getattr(torch.version, "cuda", None) or ""
    if ver.startswith("2.4.") and cuda.startswith("12.4"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
