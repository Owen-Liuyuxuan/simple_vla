"""Synthetic driving dataset — zero mmcv dependency.

Provides fake multi-camera images and metadata that match the shape expectations
of :class:`QwenVL3APlanningHead.forward_test`.
"""
import numpy as np
from typing import Dict, List, Any, Iterator
import torch
from pyhelp.debug_utils import load_data


def _identity(x):
    return x


class SyntheticDrivingDataset:
    """Random tensors matching real NuScenes dataloader output shapes.

    Only fields required by ``forward_test`` are provided.
    The head normalizes / tokenizes internally in ``forward_pre_stage1``.
    """

    def __init__(
        self,
        num_samples: int = 4,
        img_height: int = 544,
        img_width: int = 960,
        num_cameras: int = 6,
        num_history: int = 3,
        batch_size: int = 1,
    ):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.num_cameras = num_cameras
        self.num_history = num_history
        self.batch_size = batch_size
        self._idx = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        B = self.batch_size
        H, W = self.img_height, self.img_width
        C = self.num_cameras

        # (B, T, 6, 3, H, W)  — last timestamp
        img = load_data("nuScenes/img0.npz")
        # timestamp (B, )
        timestamp = load_data("nuScenes/timestamp.npz")
        # camera-to-ego projection (B, 6, 4, 4)
        projection_mat = load_data("nuScenes/projection_mat.npz")
        # image sizes (B, 6, 2)
        image_wh = torch.tensor([[[W, H]]], dtype=torch.float32).expand(B, C, -1)
        # ego status: (B, 10) — [ax, ay, az, wx, wy, wz, vx, vy, vz, steer]
        ego_status = load_data("nuScenes/ego_status.npz")
        # navigation command: (B, 3) — one-hot over
        #   [Turn Right, Turn Left, Go Straight]
        # Matches nuScenes data converter (see tools/data_converter/nuscenes_converter.py),
        # which stores `gt_ego_fut_cmd` as a 3-dim one-hot float vector. The planning head
        # (`_maybe_get_status_features`) concatenates it on `dim=-1` with a 2-D
        # `ego_status_pred`, so a 1-D index tensor would break the concat.
        cmd_idx = torch.ones(B, dtype=torch.int64)
        gt_ego_fut_cmd = torch.nn.functional.one_hot(cmd_idx, num_classes=3).float()
        # history trajectory: (B, T, 2) in meters
        hist_traj = load_data("nuScenes/hist_traj.npz") #torch.randn(B, 4, 2) * 0.1

        # One img_meta dict per batch row; each T_* is (4, 4) — matches nuScenes / InstanceBank.
        img_metas = []
        for b in range(B):
            #T_global = np.eye(4, dtype=np.float64)
            #T_global[0, 3] = float(hist_traj[b, 0, 0])
            #T_global[1, 3] = float(hist_traj[b, 0, 1])
            T_global = load_data("nuScenes/T_global.npz")
            T_global_inv = load_data("nuScenes/T_global_inv.npz")
            img_metas.append(
                {
                    "sample_idx": idx,
                    "timestamp": timestamp[b],
                    "T_global": T_global,
                    "T_global_inv": T_global_inv,
                }
            )

        # Optional GT tensors (only used in train; safe to provide None or random)
        gt_depth = None
        focal = None
        gt_bboxes_3d = None
        gt_labels_3d = None
        gt_map_labels = None
        gt_map_pts = None
        gt_agent_fut_trajs = None
        gt_agent_fut_masks = None
        gt_ego_fut_trajs = None
        gt_ego_fut_masks = None
        gt_occ_dense = None

        return {
            'img': img,
            'timestamp': timestamp,
            'projection_mat': projection_mat,
            'image_wh': image_wh,
            'gt_depth': gt_depth,
            'focal': focal,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'gt_map_labels': gt_map_labels,
            'gt_map_pts': gt_map_pts,
            'gt_agent_fut_trajs': gt_agent_fut_trajs,
            'gt_agent_fut_masks': gt_agent_fut_masks,
            'gt_ego_fut_trajs': gt_ego_fut_trajs,
            'gt_ego_fut_masks': gt_ego_fut_masks,
            'gt_ego_fut_cmd': gt_ego_fut_cmd,
            'ego_status': ego_status,
            'gt_occ_dense': gt_occ_dense,
            'hist_traj': hist_traj,
            'img_metas': img_metas,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of single-sample dicts into a batch dict.

        Handles None values gracefully.
        """
        if len(batch) == 1:
            return batch[0]

        keys = batch[0].keys()
        result = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if vals[0] is None:
                result[k] = None
            elif isinstance(vals[0], torch.Tensor):
                result[k] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], (list, dict)):
                # flatten one level
                result[k] = [item for sublist in vals for item in (sublist if isinstance(sublist, list) else [sublist])]
            else:
                result[k] = vals
        return result
