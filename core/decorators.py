"""Decorator utilities — mmcv.runner / mmcv.utils replacements.

During inference we never need the complex mixed-precision bookkeeping or
deprecation warnings, so the implementations are intentionally lightweight.
"""
import functools
import warnings

def deprecated_api_warning(msg=None, custom_cls=None, cls_name=None, **kwargs):
    """No-op deprecation wrapper — accepts mmcv positional/kw variants."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if msg is not None and not callable(msg):
                warnings.warn(str(msg), DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper

    if callable(msg):
        return decorator(msg)
    return decorator
