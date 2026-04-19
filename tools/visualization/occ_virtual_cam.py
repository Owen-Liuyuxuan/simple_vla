"""Synthetic pinhole view: ego-frame OCC points projected to a rear-elevated camera."""
from __future__ import annotations

import numpy as np
import cv2


def _R_world_to_cam_opencv(camera_pos: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Rotation 3x3: p_cam = R @ (p_world - camera_pos). OpenCV: +X right, +Y down, +Z forward.

    Ego/base_link world: x forward, y left, z up. Camera basis: right = up×forward, down = right×forward.
    """
    forward = target.astype(np.float64) - camera_pos.astype(np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(world_up, forward)
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        right = right / nr
    down = np.cross(right, forward)
    down = down / (np.linalg.norm(down) + 1e-9)
    R = np.stack([right, down, forward], axis=0)
    return R


def _project_ego_points(
    pts: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc = (R @ (pts - C).T).T
    z = pc[:, 2]
    u = fx * pc[:, 0] / (z + 1e-9) + cx
    v = fy * pc[:, 1] / (z + 1e-9) + cy
    return u, v, z


def render_occ_virtual_rear_elevated(
    points_ego: np.ndarray,
    colors_rgb01: np.ndarray,
    width: int = 960,
    height: int = 540,
    behind_m: float = 8.0,
    above_m: float = 3.0,
    look_ahead_m: float = 15.0,
    voxel_extent_m: float = 0.5,
) -> np.ndarray:
    """
    Points in ego/base_link (x forward, y left, z up). Colors (N,3) in [0,1] RGB.
    Camera sits behind the vehicle (-x) and above (+z), looking toward +x.

    Each occupied voxel is drawn as a filled axis-aligned **square** in the image, sized from
    projecting ±half-voxel offsets along ego x/y (matches LiDAR grid resolution in meters).
    """
    if points_ego.size == 0:
        return np.full((height, width, 3), 24, dtype=np.uint8)

    C = np.array([-abs(behind_m), 0.0, abs(above_m)], dtype=np.float64)
    target = np.array([look_ahead_m, 0.0, 0.5], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R = _R_world_to_cam_opencv(C, target, world_up)

    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    col = np.asarray(colors_rgb01, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if col.shape[0] != n:
        raise ValueError("colors must match points")

    fx = fy = float(min(width, height)) * 0.72
    cx = width * 0.5
    cy = height * 0.5

    u0, v0, z0 = _project_ego_points(pts, R, C, fx, fy, cx, cy)
    mask = z0 > 0.12
    if not np.any(mask):
        return np.full((height, width, 3), 24, dtype=np.uint8)

    e = float(max(voxel_extent_m, 1e-6)) * 0.5
    offs = np.array(
        [[e, 0.0, 0.0], [-e, 0.0, 0.0], [0.0, e, 0.0], [0.0, -e, 0.0]],
        dtype=np.float64,
    )
    half_px = np.zeros(n, dtype=np.float64)
    for k in range(4):
        u1, v1, z1 = _project_ego_points(pts + offs[k], R, C, fx, fy, cx, cy)
        ok = mask & (z1 > 0.05)
        du = np.abs(u1 - u0)
        dv = np.abs(v1 - v0)
        half_px = np.maximum(half_px, np.where(ok, np.maximum(du, dv), 0.0))
    half_px = np.clip(np.maximum(half_px, 1.0), 1.0, 128.0)

    ui = np.round(u0).astype(np.int32)
    vi = np.round(v0).astype(np.int32)
    inb = (
        mask
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (24, 24, 24)

    bgr = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    bgr = bgr[:, ::-1]

    idx = np.where(inb)[0]
    order = idx[np.argsort(-z0[idx])]
    for i in order:
        color = (int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]))
        s = int(np.round(half_px[i]))
        uu, vv = int(ui[i]), int(vi[i])
        cv2.rectangle(
            img,
            (uu - s, vv - s),
            (uu + s + 1, vv + s + 1),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return img
