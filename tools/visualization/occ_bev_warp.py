"""Resample OCC BEV class map from LiDAR voxel grid to ego / base_link BEV axes.

Must match ``bev_render.py`` **human** BEV axes (``ego_xy_to_bev_axes``):

- Horizontal matplotlib axis ``u = -ego_y`` (vehicle right → screen right).
- Vertical matplotlib axis ``v = ego_x`` (forward → screen up).

``imshow`` uses ``extent=(-ylim, ylim, -xlim, xlim)`` as ``(u_left, u_right, v_bottom, v_top)``.
"""
from __future__ import annotations
from re import X

import numpy as np


def warp_occ_semantic_lidar_grid_to_ego_bev(
    sem_hw: np.ndarray,
    point_cloud_range,
    T_ego_from_lidar: np.ndarray,
    xlim: float,
    ylim: float,
) -> np.ndarray:
    """
    Sample ``sem_hw`` at ego ground ``z = 0`` for each output pixel in human-BEV layout.

    Parameters
    ----------
    sem_hw :
        Full LiDAR ``(H, W)`` class map (row ↔ lidar y, col ↔ lidar x). Do not crop.
    point_cloud_range :
        ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in **LiDAR** (m).
    T_ego_from_lidar :
        ``p_ego = T @ p_lidar`` (nuScenes ``lidar2ego``).
    xlim, ylim :
        Ego half-ranges: ``X ∈ [-xlim, xlim]``, ``Y ∈ [-ylim, ylim]``.

    Returns
    -------
    ``(H, W)`` same as ``sem_hw``. ``out[i, j]`` ↔ ``(ego_x, ego_y)`` with row ``i`` along
    **forward** (``ego_x``) and column ``j`` along **lateral** (``ego_y``), matching
    ``imshow(..., extent=(-ylim, ylim, -xlim, xlim), origin='lower')``.
    """
    sem_hw = np.asarray(sem_hw)
    if sem_hw.ndim != 2:
        return sem_hw
    h, w = sem_hw.shape
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    T_inv = np.linalg.inv(np.asarray(T_ego_from_lidar, dtype=np.float64))

    T_add = np.array([[0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    ox, oy = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64),
            indexing="xy",
        )
    ego_x = - xlim + (ox + 0.5) / max(h, 1) * (2.0 * xlim)
    ego_y = - ylim + (oy + 0.5) / max(w, 1) * (2.0 * ylim)
    flat = h * w
    pe = np.zeros((flat, 4), dtype=np.float64)
    pe[:, 0] = ego_x.ravel()
    pe[:, 1] = ego_y.ravel()
    pe[:, 3] = 1.0
    pl = (T_add @ T_inv @ pe.T).T
    lx = pl[:, 0]
    ly = pl[:, 1]

    x_rng = x_max - x_min
    y_rng = y_max - y_min
    ix = np.floor((lx - x_min) / (x_rng + 1e-9) * w).astype(np.int32)
    iy = np.floor((ly - y_min) / (y_rng + 1e-9) * h).astype(np.int32)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    out = np.full(sem_hw.shape, 17, dtype=sem_hw.dtype)
    out.ravel()[valid] = sem_hw[iy[valid], ix[valid]]
    return out
