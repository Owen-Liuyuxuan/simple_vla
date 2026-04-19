"""Minimal detector base for inference-only UniDriveVLA (no mmdet dependency)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn


class BaseDetector(nn.Module):
    """Subset of mmdet ``BaseDetector`` API used by :class:`UniDriveVLA`.

    Training hooks are omitted; subclasses implement ``forward`` / ``forward_test`` / ``simple_test``.
    """

    CLASSES: Optional[Union[tuple, list]] = None
    PALETTE: Optional[List] = None

    def forward(self, return_loss: bool = True, **kwargs) -> Any:
        raise NotImplementedError

    def forward_train(self, **kwargs) -> Any:
        raise NotImplementedError

    def forward_test(self, **kwargs) -> Any:
        raise NotImplementedError

    def simple_test(self, img: torch.Tensor, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def extract_feat(self, img: torch.Tensor) -> Any:
        raise NotImplementedError
