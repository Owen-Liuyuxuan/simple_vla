from typing import Optional
import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
import os.path as osp

def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(
        img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_NEAREST
    )


def resize_to_height_area(img: np.ndarray, target_h: int) -> np.ndarray:
    """Like :func:`resize_to_height` but INTER_AREA (better for camera JPEG mosaics)."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(
        img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA
    )

def combine_frames_layout(
    cam_pred_path: str,
    bev_pred_path: str,
    bev_gt_path: Optional[str],
    save_path: str,
) -> None:
    """Horizontal strip: cam | bev_pred [| bev_gt].
    """
    cam_image = cv2.imread(cam_pred_path)
    bev_image = cv2.imread(bev_pred_path)
    if cam_image is None or bev_image is None:
        return
    bh = bev_image.shape[0]
    # cv2.hconcat requires identical row counts; cam mosaic height can differ from BEV jpg.
    if cam_image.shape[0] != bh:
        cam_image = resize_to_height_area(cam_image, bh)
    imgs = [cam_image, bev_image]

    merge_image = cv2.hconcat(imgs)
    cv2.imwrite(save_path, merge_image)

def run_visualization(
    result, idx,
    out_dir,
    plot_choices,
):
    from tools.visualization.bev_render import BEVRender

    combine_dir = osp.join(out_dir, "combine")
    os.makedirs(combine_dir, exist_ok=True)
    bev_render = BEVRender(plot_choices, out_dir)


    bev_pred_path = bev_render.render_pred_only(
        result['img_bbox'], idx
    )
