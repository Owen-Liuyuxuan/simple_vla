"""Model initialization for simple_vla inference (no mmcv monkey-patching).

Registers all plugin classes, then builds :class:`UniDriveVLA` from config dicts.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plugin.bootstrap import ensure_plugins_registered


def build_model(model_cfg: Dict[str, Any]):
    """Build UniDriveVLA from a config dict (same keys as MMDet-style ``model``)."""
    ensure_plugins_registered()
    from plugin.unidrivevla.detectors.unidrivevla import UniDriveVLA

    planning_head_cfg = model_cfg.pop("planning_head", None)
    model_type = model_cfg.pop("type", None)

    if model_type not in (None, "UniDriveVLA"):
        raise ValueError(f"Expected type='UniDriveVLA', got '{model_type}'")

    model = UniDriveVLA(
        planning_head=planning_head_cfg,
        task_loss_weight=model_cfg.get("task_loss_weight", {}),
    )
    return model


def load_model(model_cfg: Dict[str, Any], ckpt_path: str, map_location: str = "cpu"):
    """Build model and load checkpoint."""
    model = build_model(model_cfg)
    if not os.path.exists(ckpt_path):
        print(f"[load_model] WARNING: Checkpoint not found: {ckpt_path}, skipping.")
        return model
    from utils.checkpoint_loader import load_checkpoint

    ckpt = load_checkpoint(model, ckpt_path, map_location=map_location, strict=False)
    if isinstance(ckpt, dict) and "meta" in ckpt:
        if "CLASSES" in ckpt["meta"]:
            model.CLASSES = ckpt["meta"]["CLASSES"]
        if "PALETTE" in ckpt["meta"]:
            model.PALETTE = ckpt["meta"]["PALETTE"]
    return model
