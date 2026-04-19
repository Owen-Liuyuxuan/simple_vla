"""models/base_target.py → samplers/base.py

Base class for target samplers with denoising support.
"""
from abc import ABC, abstractmethod


__all__ = ["BaseTargetWithDenoising"]


class BaseTargetWithDenoising(ABC):
    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        super(BaseTargetWithDenoising, self).__init__()
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None

    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """Perform Hungarian matching, returning matched GT."""

    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """Generate noisy instances for the current frame."""
        return None

    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """Insert previously saved dn_metas into current frame's noisy instances."""
        pass

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        """Randomly save info for temporal noisy groups."""
        if self.num_temp_dn_groups < 0:
            return
        self.dn_metas = dict(dn_anchor=dn_anchor[:, : self.num_temp_dn_groups])
