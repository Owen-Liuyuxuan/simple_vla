#!/usr/bin/env python3
"""Bisect which `scaled_dot_product_attention` backend produces NaNs.

Expects the same `.npz` layout as `test.py` under ``torch_functional_test/``.

Usage (from repo root)::

    python simple_vla/tools/debug_sdpa_backends.py
    TORCH_FUNCTIONAL_TEST_DIR=mydir python simple_vla/tools/debug_sdpa_backends.py
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from pyhelp.debug_utils import load_data

    data_dir = os.environ.get(
        "TORCH_FUNCTIONAL_TEST_DIR", "torch_functional_test"
    )
    dropout = 0.0
    scaling = 0.08838834764831845
    is_causal = False

    q = load_data(f"{data_dir}/query.npz")
    k = load_data(f"{data_dir}/key.npz")
    v = load_data(f"{data_dir}/value.npz")
    mask = load_data(f"{data_dir}/attention_mask.npz")

    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("dtypes:", q.dtype, mask.dtype)
    print("inputs finite:", bool(torch.isfinite(q).all()), bool(torch.isfinite(mask).all()))

    def stats(name: str, out: torch.Tensor) -> None:
        frac = torch.isfinite(out).float().mean().item()
        nan = bool(torch.isnan(out).any().item())
        mx = (
            out.detach().abs().max().item()
            if torch.isfinite(out).any()
            else float("nan")
        )
        print(f"  {name:28s}  finite_frac={frac:.8f}  nan={nan}  max_abs={mx}")

    trials = [
        ("default", dict(enable_flash=True, enable_mem_efficient=True, enable_math=True, enable_cudnn=True)),
        ("math_only", dict(enable_flash=False, enable_mem_efficient=False, enable_math=True, enable_cudnn=False)),
        ("mem_eff_only", dict(enable_flash=False, enable_mem_efficient=True, enable_math=False, enable_cudnn=False)),
        ("flash_math_no_mem", dict(enable_flash=True, enable_mem_efficient=False, enable_math=True, enable_cudnn=False)),
        ("mem_math", dict(enable_flash=False, enable_mem_efficient=True, enable_math=True, enable_cudnn=False)),
    ]

    for name, kwargs in trials:
        print(f"\n{name} {kwargs}")
        try:
            with torch.backends.cuda.sdp_kernel(**kwargs):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=dropout,
                    scale=scaling,
                    is_causal=is_causal,
                )
            stats("out", out)
        except RuntimeError as e:
            print(f"  RuntimeError: {e}")

    print(
        "\nTip: if `mem_eff_only` / `mem_math` show nan=True but `math_only` is clean, "
        "use `maybe_configure_cuda_sdp()` from utils.torch_runtime "
        "or export TORCH_DISABLE_MEM_EFFICIENT_SDPA=1."
    )


if __name__ == "__main__":
    main()
