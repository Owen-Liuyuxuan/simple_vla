#!/usr/bin/env python3
"""Minimal inference using simple_vla registries (no mmcv / mmdet).

Usage:
    python inference.py

Environment variables:
    CONFIG      Path to config file  (default: configs/simple_inference_stage2_2b.py)
    CHECKPOINT  Path to checkpoint    (default: ../UniDriveVLA_Stage3_Nuscenes_2B.pt)
    OCCWORLD_VAE_PATH  Path to occworld VAE weights
    VLM_PRETRAINED_PATH HuggingFace path or local path to VLM weights
    DEVICE      cuda or cpu           (default: cuda if available)
    NUM_SAMPLES Number of synthetic forward passes (each uses batch_size=1)
    BATCH_SIZE  Ignored; this script always runs with batch size 1

The script:
  1. Registers plugin classes (bootstrap) and builds UniDriveVLA via ``build_model``.
  2. Loads config via ``utils.config_parser``.
  3. Loads checkpoint via ``utils.checkpoint_loader``.
  4. Runs synthetic driving data through ``forward_test``.
  5. Prints trajectory shapes.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import torch


from utils.torch_runtime import maybe_configure_cuda_sdp

maybe_configure_cuda_sdp()

# 1. Load config first (uses our pure-Python parser)
CONFIG = os.environ.get(
    "CONFIG",
    str(Path(__file__).parent / "configs" / "simple_inference_stage2_2b.py")
)
CKPT = os.environ.get(
    "CHECKPOINT",
    str(Path(__file__).parent / "UniDriveVLA_Stage3_Nuscenes_2B.pt")
)

print("=" * 60)
print("  simple_vla inference (no mmcv/mmdet)")
print("=" * 60)
print(f"  Config:     {CONFIG}")
print(f"  Checkpoint: {CKPT}")
print(f"  GPU:        {torch.cuda.is_available()}")
print()

# 2. Model builder (bootstrap registers all plugin classes)
print("[1/5] Loading model_init...")
from wrappers.model_init import build_model, load_model
print("      Done.")

# 3. Load config
print("[2/5] Loading config...")
from utils.config_parser import load_config
cfg = load_config(CONFIG)
print(f"      Config loaded: model type = {cfg.get('model', {}).get('type', 'N/A')}")

# 4. Build + load model
print("[3/5] Building model...")
model_cfg = cfg["model"]

# Update paths from environment
ph = model_cfg.get("planning_head", {})
if "pretrained_path" in ph:
    ph["pretrained_path"] = os.environ.get(
        "VLM_PRETRAINED_PATH", ph["pretrained_path"]
    )
if "occworld_vae_path" in ph:
    ph["occworld_vae_path"] = os.environ.get(
        "OCCWORLD_VAE_PATH", ph["occworld_vae_path"]
    )

model = build_model(model_cfg)

if os.path.exists(CKPT):
    print(f"[4/5] Loading checkpoint: {CKPT}")
    from utils.checkpoint_loader import load_checkpoint
    ckpt = load_checkpoint(model, CKPT, map_location="cpu", strict=False)
    if isinstance(ckpt, dict) and 'meta' in ckpt:
        if 'CLASSES' in ckpt['meta']:
            model.CLASSES = ckpt['meta']['CLASSES']
        if 'PALETTE' in ckpt['meta']:
            model.PALETTE = ckpt['meta']['PALETTE']
    print("      Checkpoint loaded.")
else:
    print(f"[4/5] Checkpoint not found: {CKPT} — skipping (using random init).")

# 5. Move to device
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE).eval()

# 6. Synthetic data (batch size is fixed at 1: img_metas length must match B, and
# temporal InstanceBank state is exercised per forward; multi-batch smoke tests need
# collated img_metas / poses — see SyntheticDrivingDataset when batch_size > 1).
print("[5/5] Running inference on synthetic data...")
from datasets.synthetic_dataset import SyntheticDrivingDataset
from tools.simple_vis import run_visualization

num_samples = int(os.environ.get("NUM_SAMPLES", "4"))
img_height = int(os.environ.get("IMG_HEIGHT", "544"))
img_width = int(os.environ.get("IMG_WIDTH", "960"))

_batch_env = os.environ.get("BATCH_SIZE", "1").strip()
if _batch_env not in ("", "1"):
    print(
        f"  Note: BATCH_SIZE={_batch_env} ignored; inference.py always uses batch_size=1."
    )

dataset = SyntheticDrivingDataset(
    num_samples=num_samples,
    img_height=img_height,
    img_width=img_width,
    batch_size=1,
)

def _to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    return data

DEFAULT_PLOT_CHOICES = dict(
    draw_pred=True,
    det=True,
    track=True,
    motion=True,
    map=True,
    planning=True,
)

total_time = 0.0
with torch.no_grad():
    for i, sample in enumerate(dataset):
        batch = sample
        batch = _to_device(batch, DEVICE)

        t0 = time.time()
        result = model(return_loss=False, rescale=True, **batch)
        dt = time.time() - t0
        total_time += dt

        result = result[0] if isinstance(result, list) else result
        planning = result.get("planning", {})
        final_plan = result.get("img_bbox", {}).get("final_planning")

        run_visualization(result, i, 
            'output', plot_choices=DEFAULT_PLOT_CHOICES)

        print(
            f"  sample {i:02d}: {dt:.2f}s  "
            f"final_plan={getattr(final_plan, 'shape', None)}"
        )

avg_time = total_time / num_samples if num_samples > 0 else 0
print()
print(f"  Average time per sample: {avg_time:.2f}s")
print("  Done.")
