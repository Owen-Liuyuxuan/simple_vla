"""losses/plan_map_loss.py → losses/plan_map.py

GT map-based planning losses (boundary and direction).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES


def _segments_intersect(line1_start, line1_end, line2_start, line2_end):
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    det = dx1 * dy2 - dx2 * dy1
    parallel = det == 0

    safe_det = det.clone()
    safe_det[parallel] = 1.0

    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / safe_det
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / safe_det

    intersect = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
    intersect[parallel] = False
    return intersect


def _prepare_gt_map(gt_map_pts_b, gt_map_labels_b, cls_idx, device, dtype):
    if not torch.is_tensor(gt_map_pts_b):
        gt_map_pts_b = torch.tensor(gt_map_pts_b, dtype=dtype, device=device)
    else:
        gt_map_pts_b = gt_map_pts_b.to(device=device, dtype=dtype)

    if not torch.is_tensor(gt_map_labels_b):
        gt_map_labels_b = torch.tensor(gt_map_labels_b, dtype=torch.long, device=device)
    else:
        gt_map_labels_b = gt_map_labels_b.to(device=device)

    mask = gt_map_labels_b == cls_idx
    if mask.sum() == 0:
        return None

    lines = gt_map_pts_b[mask, 0, :, :]
    return lines


@LOSSES.register_module()
class GTMapBoundLoss(nn.Module):
    """Penalise predicted ego trajectory crossing road boundaries."""

    def __init__(self, boundary_cls_idx: int = 2, dis_thresh: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.boundary_cls_idx = boundary_cls_idx
        self.dis_thresh = dis_thresh
        self.weight = weight

    def forward(
        self,
        pred_abs: torch.Tensor,
        gt_map_pts: list,
        gt_map_labels: list,
        gt_ego_fut_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, _ = pred_abs.shape
        device = pred_abs.device
        dtype = torch.float32
        pred_abs = pred_abs.to(dtype)

        total_loss = pred_abs.new_zeros(1)
        n_valid = 0

        for b in range(B):
            lines = _prepare_gt_map(gt_map_pts[b], gt_map_labels[b],
                                    self.boundary_cls_idx, device, dtype)
            if lines is None:
                continue

            M, P, _ = lines.shape
            lines_flat = lines.reshape(M * P, 2)

            origin = pred_abs.new_zeros(1, 2)
            traj_ext = torch.cat([origin, pred_abs[b]], dim=0)

            for t in range(T):
                if gt_ego_fut_masks is not None and gt_ego_fut_masks[b, t] < 0.5:
                    continue

                ego_pt = pred_abs[b, t]

                dist = torch.linalg.norm(
                    ego_pt.unsqueeze(0) - lines_flat, dim=-1)
                min_dist = dist.min()
                loss_t = F.relu(self.dis_thresh - min_dist)

                seg_start = traj_ext[t].unsqueeze(0).expand(M * (P - 1), 2)
                seg_end   = traj_ext[t + 1].unsqueeze(0).expand(M * (P - 1), 2)
                bd_starts = lines[:, :-1, :].reshape(M * (P - 1), 2)
                bd_ends   = lines[:, 1:,  :].reshape(M * (P - 1), 2)
                crossed = _segments_intersect(seg_start, seg_end, bd_starts, bd_ends)
                if crossed.any():
                    break

                total_loss = total_loss + loss_t
                n_valid += 1

        if n_valid == 0:
            return pred_abs.new_zeros(1)
        return (total_loss / n_valid) * self.weight


@LOSSES.register_module()
class GTMapDirectionLoss(nn.Module):
    """Penalise ego heading angle inconsistent with nearest lane divider direction."""

    def __init__(self, divider_cls_idx: int = 1, dis_thresh: float = 2.0, weight: float = 1.0):
        super().__init__()
        self.divider_cls_idx = divider_cls_idx
        self.dis_thresh = dis_thresh
        self.weight = weight

    def forward(
        self,
        pred_abs: torch.Tensor,
        gt_map_pts: list,
        gt_map_labels: list,
        gt_ego_fut_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, _ = pred_abs.shape
        device = pred_abs.device
        dtype = torch.float32
        pred_abs = pred_abs.to(dtype)

        total_loss = pred_abs.new_zeros(1)
        n_valid = 0

        for b in range(B):
            lines = _prepare_gt_map(gt_map_pts[b], gt_map_labels[b],
                                    self.divider_cls_idx, device, dtype)
            if lines is None:
                continue

            traj_disp = torch.linalg.norm(pred_abs[b, -1] - pred_abs[b, 0])
            if traj_disp < 1.0:
                continue

            M, P, _ = lines.shape

            origin = pred_abs.new_zeros(1, 2)
            traj_ext = torch.cat([origin, pred_abs[b]], dim=0)
            diff = traj_ext[1:] - traj_ext[:-1]
            ego_yaw = torch.atan2(diff[:, 1], diff[:, 0])

            for t in range(T):
                if gt_ego_fut_masks is not None and gt_ego_fut_masks[b, t] < 0.5:
                    continue

                ego_pt = pred_abs[b, t]

                dist_to_pts = torch.linalg.norm(
                    ego_pt.unsqueeze(0).unsqueeze(0) - lines, dim=-1)
                min_dist_per_inst = dist_to_pts.min(dim=-1).values
                nearest_inst = min_dist_per_inst.argmin()
                nearest_dist = min_dist_per_inst[nearest_inst]

                if nearest_dist > self.dis_thresh:
                    continue

                inst_pts = lines[nearest_inst]
                dist_to_inst = dist_to_pts[nearest_inst]
                pt_idx = dist_to_inst.argmin().clamp(0, P - 2)
                seg_dir = inst_pts[pt_idx + 1] - inst_pts[pt_idx]
                lane_yaw = torch.atan2(seg_dir[1], seg_dir[0])

                yaw_diff = ego_yaw[t] - lane_yaw
                yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
                yaw_diff = yaw_diff.abs()
                if yaw_diff > math.pi / 2:
                    yaw_diff = math.pi - yaw_diff

                total_loss = total_loss + yaw_diff
                n_valid += 1

        if n_valid == 0:
            return pred_abs.new_zeros(1)
        return (total_loss / n_valid) * self.weight
