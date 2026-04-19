"""Shared OCC semantic class colors (matplotlib tab20 discrete swatches) for PNG/BEV/PCD."""
from __future__ import annotations

import numpy as np

_TAB20_N = 20


def _tab20_palette():
    import matplotlib.pyplot as plt

    return np.asarray(plt.get_cmap("tab20").colors, dtype=np.float64)


def occ_class_ids_to_rgb_u8(class_ids: np.ndarray) -> np.ndarray:
    """(H,W) or (...) int class ids -> RGB uint8, same palette everywhere."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    pal = _tab20_palette()
    idx = np.mod(class_ids, _TAB20_N)
    return (pal[idx] * 255.0).astype(np.uint8)


def occ_class_ids_to_bgr_u8(class_ids: np.ndarray) -> np.ndarray:
    """(H,W) int class ids -> BGR uint8 for OpenCV."""
    rgb = occ_class_ids_to_rgb_u8(class_ids)
    return rgb[..., ::-1].copy()


def occ_class_ids_to_rgb01_flat(class_ids: np.ndarray) -> np.ndarray:
    """(N,) int class ids -> (N,3) float RGB in [0,1]."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    pal = _tab20_palette()
    idx = np.mod(class_ids, _TAB20_N)
    return pal[idx][:, :3].astype(np.float64)
