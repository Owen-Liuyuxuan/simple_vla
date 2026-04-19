"""models/map/target.py → samplers/map.py

SparsePoint3DTarget + HungarianLinesAssigner for map prediction.
"""
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from simple_vla.core.registry import BBOX_SAMPLERS, BBOX_ASSIGNERS
from simple_vla.core.samplers.base import BaseTargetWithDenoising
from simple_vla.core.samplers.match_cost import build_match_cost
from simple_vla.core.bbox import build_assigner



@BBOX_SAMPLERS.register_module()
class SparsePoint3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner) if assigner else None
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size

    def sample(self, cls_preds, pts_preds, cls_targets, pts_targets):
        pts_targets = [x.flatten(2, 3) if len(x.shape) == 4 else x for x in pts_targets]
        indices = []
        for (cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            pts_pred_norm = self.normalize_line(pts_pred)
            pts_target_norm = self.normalize_line(pts_target)
            preds = dict(lines=pts_pred_norm, scores=cls_pred)
            gts = dict(lines=pts_target_norm, labels=cls_target)
            if self.assigner is not None:
                indice = self.assigner.assign(preds, gts)
                indices.append(indice)
            else:
                indices.append((None, None, None))

        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            if gt_permute_index is not None:
                permute_idx = gt_permute_index[pred_idx, target_idx]
                output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
                output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx].to(dtype=output_box_target.dtype)
            else:
                output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_reg_weights[i, pred_idx] = output_reg_weights.new_tensor(1, dtype=output_reg_weights.dtype)

        return output_cls_target, output_box_target, output_reg_weights

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        origin = line.new_tensor([-self.roi_size[0] / 2, -self.roi_size[1] / 2])
        line = line - origin
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)
        return line


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner:
    """One-to-one matching between line predictions and ground truth."""

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self, preds, gts, ignore_cls_cost=False, gt_bboxes_ignore=None, eps=1e-7):
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        gt_permute_idx = None
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost)
        else:
            cost = self.cost(preds, gts, ignore_cls_cost)

        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx