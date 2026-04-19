"""core/bbox/match_costs/match_cost.py → match_cost.py

Lightweight match cost utilities for Hungarian assignment.
"""
import torch
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss

from simple_vla.core.registry import Registry

MATCH_COST = Registry("match_cost")


def build_match_cost(cfg):
    """Build a match cost from config dict."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        cost_type = cfg.get('type')
        if cost_type == 'BBox3DL1Cost':
            return BBox3DL1Cost(weight=cfg.get('weight', 1.0))
        elif cost_type == 'DiceCost':
            return DiceCost(weight=cfg.get('weight', 1.0))
        elif cost_type == 'LinesL1Cost':
            return LinesL1Cost(
                weight=cfg.get('weight', 1.0),
                beta=cfg.get('beta', 0.0),
                permute=cfg.get('permute', False),
            )
        elif cost_type == 'MapQueriesCost':
            return MapQueriesCost(
                cls_cost=cfg.get('cls_cost'),
                reg_cost=cfg.get('reg_cost'),
                iou_cost=cfg.get('iou_cost'),
            )
        else:
            raise ValueError(f"Unknown match cost type: {cost_type}")
    return cfg


@MATCH_COST.register_module()
class BBox3DL1Cost:
    """L1 cost between predicted and target boxes."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes, ignore_cls_cost=False):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


class LinesL1Cost(object):
    """L1 / smooth-L1 cost between predicted line queries and GT polylines."""

    def __init__(self, weight=1.0, beta=0.0, permute=False):
        self.weight = weight
        self.permute = permute
        self.beta = beta

    def __call__(self, lines_pred, gt_lines, **kwargs):
        if self.permute:
            assert len(gt_lines.shape) == 3
        else:
            assert len(gt_lines.shape) == 2

        num_pred, num_gt = len(lines_pred), len(gt_lines)
        if self.permute:
            gt_lines = gt_lines.flatten(0, 1)

        num_pts = lines_pred.shape[-1] // 2

        if self.beta > 0:
            lines_pred = lines_pred.unsqueeze(1).repeat(1, len(gt_lines), 1)
            gt_lines = gt_lines.unsqueeze(0).repeat(num_pred, 1, 1)
            dist_mat = smooth_l1_loss(
                lines_pred, gt_lines, reduction='none', beta=self.beta
            ).sum(-1)
        else:
            dist_mat = torch.cdist(lines_pred, gt_lines, p=1)

        dist_mat = dist_mat / num_pts

        if self.permute:
            dist_mat = dist_mat.view(num_pred, num_gt, -1)
            dist_mat, gt_permute_index = torch.min(dist_mat, 2)
            return dist_mat * self.weight, gt_permute_index

        return dist_mat * self.weight


class DynamicLinesCost:
    """Placeholder for mask-based line cost (training); not used in minimal inference."""

    permute = False


class MapQueriesCost(object):
    """Combined cls + reg (+ optional iou) cost for map query matching."""

    def __init__(self, cls_cost, reg_cost, iou_cost=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost) if iou_cost is not None else None

    def __call__(self, preds: dict, gts: dict, ignore_cls_cost: bool):
        cls_cost = self.cls_cost(preds['scores'], gts['labels'])

        regkwargs = {}
        if 'masks' in preds and 'masks' in gts:
            assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!'
            regkwargs = {
                'masks_pred': preds['masks'],
                'masks_gt': gts['masks'],
            }

        reg_cost = self.reg_cost(preds['lines'], gts['lines'], **regkwargs)
        if self.reg_cost.permute:
            reg_cost, gt_permute_idx = reg_cost

        if ignore_cls_cost:
            cost = reg_cost
        else:
            cost = cls_cost + reg_cost

        if self.iou_cost is not None:
            iou_cost = self.iou_cost(preds['lines'], gts['lines'])
            cost += iou_cost

        if self.reg_cost.permute:
            return cost, gt_permute_idx
        return cost


@MATCH_COST.register_module()
class DiceCost:
    """Soft Dice cost for line map matching."""

    def __init__(self, weight=1.0, permute=True):
        self.weight = weight
        self.permute = permute

    def __call__(self, preds, gts, ignore_cls_cost=False):
        input_lines = preds['lines']
        target_lines = gts['lines']

        N1, H1, W1 = input_lines.shape
        N2, H2, W2 = target_lines.shape

        if H1 != H2 or W1 != W2:
            target_lines = F.interpolate(
                target_lines.unsqueeze(0), size=(H1, W1), mode='bilinear'
            ).squeeze(0)

        input_flat = input_lines.contiguous().view(N1, -1)[:, None, :]
        target_flat = target_lines.contiguous().view(N2, -1)[None, :, :]

        a = torch.sum(input_flat * target_flat, -1)
        b = torch.sum(input_flat * input_flat, -1) + 0.001
        c = torch.sum(target_flat * target_flat, -1) + 0.001
        d = (2 * a) / (b + c)
        return (1 - d) * self.weight
