# simple_vla

A minimal **inference-only** slice of [UniDriveVLA](https://github.com/xiaomi-research/UniDriveVLA/) (same model/plugin code paths, trimmed for running a pretrained checkpoint **without** MMCV, MMDetection, or DeepSpeed).

## What this is

- **Purpose**: load a released checkpoint and run `forward_test` on prepared inputs (synthetic samples or future real samples).
- **Not included**: training pipelines, full nuScenes dataloaders, or distributed training (no DeepSpeed).

## Requirements

| Component | Notes |
|-----------|--------|
| **PyTorch** | GPU build recommended (`cuda`); Tested on `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`|
| **transformers** | Pin: `transformers==4.57.1`, then apply the **Qwen3-VL patch** shipped under `simple_vla/qwenvl3/transformers_replace/` (see [Transformers patch](#transformers-patch)). |
| **pyhelp** | [Owen-Liuyuxuan/pyhelp](https://github.com/Owen-Liuyuxuan/pyhelp) — used for `load_data` when loading packaged sample tensors (see `datasets/synthetic_dataset.py`). Install from the repo (e.g. `pip install -e .` from a clone, or follow that project’s `install.sh`). |
| **Other Python deps** | See `requirements.txt` (`einops`, `flash-attn`, `matplotlib`, `numpy`, `opencv-python`, `pillow`, `safetensors`, `scipy`, `timm`, `tqdm`, etc.). |

**Intentionally omitted**: `mmcv`, `mmdet`, `mmengine`, `deepspeed`.

## Transformers patch

After installing `transformers==4.57.1`, merge the vendored Qwen3-VL overrides into your environment’s `transformers` package so imports resolve to the patched `modeling_qwen3_vl` (and related) code.

Example (adjust the destination to your **active** site-packages or venv):

```bash
# From repo root; destination must be the transformers package you import at runtime
cp -r simple_vla/qwenvl3/transformers_replace/models <path-to-site-packages>/transformers/
```

`simple_vla/patch.sh` is a **template** with a machine-specific path — edit it or use the command above.

## Data preparation

Download from https://drive.google.com/drive/folders/1bqF_iZtNAjinooTgHqimM0h5gV9lDx2j?usp=drive_link

1. **K-means anchors** (used by the unified decoder `InstanceBank` configs): paths in `configs/simple_inference_stage2_2b.py` expect files such as:
   - `kmeans/kmeans_det_900.npy`
   - `kmeans/kmeans_map_100.npy`  
   These (and any other assets referenced by the config) should live under your working tree or be symlinked so those paths resolve.

2. **Synthetic / sample tensors** for the bundled loader: `SyntheticDrivingDataset` loads several `.npz` files via `pyhelp` (e.g. `sample_data/img0.npz`, `sample_data/timestamp.npz`, `sample_data/projection_mat.npz`, …) relative to the **current working directory**. A full curated pack will be **uploaded and documented later**; until then, place compatible files at the paths used in `datasets/synthetic_dataset.py` or adjust paths locally.

## Pretrained weights

Download the Stage-3 NuScenes 2B checkpoint from Hugging Face:

- **Model repo**: [owl10/UniDriveVLA_Nusc_Base_Stage3](https://huggingface.co/owl10/UniDriveVLA_Nusc_Base_Stage3/tree/main)  
- Typical file: `UniDriveVLA_Stage3_Nuscenes_2B.pt`

Point `CHECKPOINT` (see below) at the downloaded file. The planning head also needs VLM backbone weights; the default config uses a Hugging Face id for `pretrained_path` — override with `VLM_PRETRAINED_PATH` if you use a local cache or another revision.

## Running inference

**Main entry**: `simple_vla/inference.py` (run from the **repository root** so imports and default paths work).

```bash
cd /path/to/unidrivevla   # parent of simple_vla/
python simple_vla/inference.py
```

### Environment variables

| Variable | Role | Default |
|----------|------|---------|
| `CONFIG` | Python config file for the model | `simple_vla/configs/simple_inference_stage2_2b.py` |
| `CHECKPOINT` | Path to `*.pt` checkpoint | `../UniDriveVLA_Stage3_Nuscenes_2B.pt` (relative to CWD: repo root) |
| `VLM_PRETRAINED_PATH` | HF repo id or local path for VLM weights | Value in config / env |
| `OCCWORLD_VAE_PATH` | OccWorld VAE checkpoint if enabled in config | From config |
| `DEVICE` | `cuda` or `cpu` | `cuda` if available |
| `NUM_SAMPLES` | Number of forward passes | `4` |
| `IMG_HEIGHT`, `IMG_WIDTH` | Input image size | `544`, `960` |

`BATCH_SIZE` is ignored; this script runs with batch size 1.

Outputs (e.g. visualizations) go under `output/` by default (see `tools/simple_vis.py`).

## Layout (high level)

- `inference.py` — bootstrap, load config/checkpoint, run synthetic loop.
- `configs/` — pure-Python model config (no mmcv registry).
- `plugin/`, `core/`, `wrappers/` — UniDriveVLA model pieces wired for this tree.
- `datasets/synthetic_dataset.py` — sample batch construction via `pyhelp.debug_utils.load_data`.
- `qwenvl3/transformers_replace/` — patches for `transformers==4.57.1`.

For questions about the upstream full training stack, see the parent UniDriveVLA project documentation.
