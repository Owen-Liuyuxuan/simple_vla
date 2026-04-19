"""mmdet assigner/sampler stubs -- no-op replacements for training-only constructs."""
__all__ = ['build_match_cost', 'build_assigner', 'build_sampler', 'AssignResult', 'BaseAssigner']


class BaseAssigner:
    pass


class AssignResult:
    def __init__(self, num_gt=0, *args, **kwargs):
        self.num_gt = num_gt


def build_match_cost(cfg):
    return None


def build_assigner(cfg):
    return None


def build_sampler(cfg):
    return None
