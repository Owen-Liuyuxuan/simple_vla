"""Register all classes — delegates to :func:`plugin.bootstrap.ensure_plugins_registered`."""

from plugin.bootstrap import ensure_plugins_registered
from utils.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
    HEADS,
    LOSSES,
    BBOX_SAMPLERS,
    BBOX_CODERS,
)


def verify_registration():
    """Debug helper: print registered class counts after bootstrap."""
    ensure_plugins_registered()
    for name, reg in [
        ("ATTENTION", ATTENTION),
        ("PLUGIN_LAYERS", PLUGIN_LAYERS),
        ("POSITIONAL_ENCODING", POSITIONAL_ENCODING),
        ("FEEDFORWARD_NETWORK", FEEDFORWARD_NETWORK),
        ("NORM_LAYERS", NORM_LAYERS),
        ("HEADS", HEADS),
        ("LOSSES", LOSSES),
        ("BBOX_SAMPLERS", BBOX_SAMPLERS),
        ("BBOX_CODERS", BBOX_CODERS),
    ]:
        keys = list(reg._module_map.keys())
        print(f"  {name}: {len(keys)} classes — {keys[:5]}{'...' if len(keys) > 5 else ''}")


if __name__ == "__main__":
    verify_registration()
