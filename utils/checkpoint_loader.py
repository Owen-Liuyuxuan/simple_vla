"""Load model checkpoint without mmcv."""
import torch


def load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False):
    """Drop-in replacement for mmcv.runner.load_checkpoint.

    Returns the full checkpoint dict (so callers can read meta info like CLASSES).
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model', 'module', 'ema']:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Remove 'module.' DDP prefix (may appear once or twice)
    cleaned = {}
    for k, v in state_dict.items():
        kk = k
        while kk.startswith('module.'):
            kk = kk[7:]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"[load_checkpoint] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    return ckpt
