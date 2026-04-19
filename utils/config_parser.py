"""Parse Python-style config files without mmcv.Config."""
import os
import re


def load_config(config_path: str) -> dict:
    """Parse a Python config file to a plain dict.

    Handles the same Python-dict syntax as mmcv.Config.
    Registries are pre-registered and available as names in the namespace.
    """
    abs_path = os.path.abspath(config_path)

    with open(config_path) as f:
        content = f.read()

    # Remove mmcv sys.path manipulations that configs inject
    content = re.sub(r'^import sys\b.*?\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^sys\.path\.insert.*?\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^sys\.path\.append.*?\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^from collections import .*?\n', '', content)
    content = re.sub(r'^import collections\b.*?\n', '', content)

    namespace = {
        '__file__': abs_path,
        '__name__': '__config__',
        '__builtins__': __builtins__,
        'os': os,
    }

    # Inject env helper (configs do: VLM_PATH = os.environ.get(...))
    def _get_env(key, default=None):
        return os.environ.get(key, default)
    namespace['_get_env'] = _get_env

    # Inject pre-registered registries so config can reference them by name
    from simple_vla.utils.registry import (
        ATTENTION, PLUGIN_LAYERS, POSITIONAL_ENCODING,
        FEEDFORWARD_NETWORK, NORM_LAYERS, HEADS, LOSSES,
        BBOX_SAMPLERS, BBOX_CODERS,
        build_from_cfg, build,
    )
    for _name in ('ATTENTION', 'PLUGIN_LAYERS', 'POSITIONAL_ENCODING',
                   'FEEDFORWARD_NETWORK', 'NORM_LAYERS', 'HEADS', 'LOSSES',
                   'BBOX_SAMPLERS', 'BBOX_CODERS',
                   'build_from_cfg', 'build'):
        namespace[_name] = locals()[_name]

    exec(compile(content, config_path, 'exec'), namespace)

    cfg = {
        k: v for k, v in namespace.items()
        if not k.startswith('_') and k not in ('__file__', '__name__', '__builtins__')
    }
    return cfg
