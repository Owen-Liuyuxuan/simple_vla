"""Feed-Forward Network (FFN) — mmcv.cnn.bricks.transformer.FFN replacement.

Matches the mmcv FFN signature used throughout the codebase:
    FFN(embed_dims, feedforward_channels, num_fcs, act_cfg, ffn_drop, add_identity=True)

The implementation builds two Linear layers (or three if ``num_fcs > 2``) with
an activation layer in between and an optional final dropout.
"""
from __future__ import annotations

import torch.nn as nn
from typing import Optional

from .activation import build_activation_layer
from .dropout import build_dropout
from .nn import BaseModule, xavier_init


class FFN(BaseModule):
    """Position-wise feed-forward network, identical to mmcv's FFN."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        act_cfg: Optional[dict] = None,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
        dropout_layer: Optional[dict] = None,
    ):
        """Create FFN.

        Parameters
        ----------
        embed_dims : int
            Input & output feature dimension.
        feedforward_channels : int
            Hidden dimension (typically 2-4× embed_dims).
        num_fcs : int
            Number of fully-connected layers (2 standard; >2 adds more).
        act_cfg : dict | None
            Activation config e.g. ``{'type': 'GELU'}``.
        ffn_drop : float
            Drop probability after the activation (deprecated kw kept for compat).
        add_identity : bool
            When True a residual identity connection adds the input to output.
        dropout_layer : dict | None
            Alternative dropout cfg; if None uses ``ffn_drop``.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.add_identity = add_identity

        # Build the sequential stack
        layers = []
        in_channels = embed_dims
        for i in range(num_fcs):
            out_channels = (
                feedforward_channels if i < num_fcs - 1 else embed_dims
            )
            layers.append(nn.Linear(in_channels, out_channels))
            if i < num_fcs - 1:
                layers.append(build_activation_layer(act_cfg))
                layers.append(build_dropout(dropout_layer or {'type': 'Dropout', 'p': ffn_drop}))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.dropout = build_dropout(dropout_layer or {'type': 'Dropout', 'p': ffn_drop})
        self.init_weights()

    def forward(self, x, **kwargs):
        """Forward pass — optionally passes residual to the output."""
        out = self.layers(x)
        if self.add_identity:
            out = out + x
        return self.dropout(out)

    def init_weights(self):
        """Xavier initialisation — training-time only, kept for parity."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform', bias='zero')


__all__ = ['FFN']
