"""attention.py — flash-attention and multi-head attention modules.

This is a copy+import-rewrite of
`nuScenes/projects/mmdet3d_plugin/models/attention.py`.
All mmcv/mmdet dependencies are replaced with simple_vla.core.
"""
import warnings
import math

import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_

from core.decorators import deprecated_api_warning
from core.fp16_helper import auto_fp16
from core.nn import BaseModule
from core.dropout import build_dropout
from core.registry import ATTENTION
import torch.utils.checkpoint as cp

from einops import rearrange

# FlashAttention backend — try to import the fastest available variant
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_unpadded_kvpacked_func')
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_varlen_kvpacked_func')

from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Low-level scaled-dot-product attention using flash-attn kernels.

    Matches the original mmcv implementation: fp16-only, CUDA-only.
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv,
                causal=False,
                key_padding_mask=None):
        """Flash-attn forward with packed KV.

        q:       (B, T, H, D)
        kv:      (B, S, 2, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]

        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=kv.device)
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):
    """Internal MHA that wraps FlashAttention with packed QKV."""

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True,
                 attention_dropout=0.0, causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, \
            "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None):
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)

        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


@ATTENTION.register_module()
class MultiheadFlashAttention(BaseModule):
    """Wrapper around FlashMHA with identity residual and positional encodings.

    Keeps the exact same signature as the mmcv version so that configs
    work without changes.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', p=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(MultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['p'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True
        self.attn = FlashMHA(
            embed_dim=embed_dims,
            num_heads=num_heads,
            attention_dropout=attn_drop,
            dtype=torch.float16,
            device='cuda',
            **kwargs
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'}, cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        assert attn_mask is None, 'attn mask not supported now.'
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=key_padding_mask)[0]

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Positional sine embedding — mostly copy-paste from DAB-DETR."""
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos
