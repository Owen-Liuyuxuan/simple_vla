# Copyright 2026 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, List

import torch
from torch import nn
import torch.nn.functional as F
import logging
from simple_vla.core.registry import HEADS
from simple_vla.core.registry import build_head
from timm.models.layers import Mlp
from einops import rearrange
from .unidrivevla_vlm_qwenvl3 import Qwen3VLWithExpertModel
from torch.nn.utils.rnn import pad_sequence
from simple_vla.plugin.unidrivevla.dense_heads.flex_attention_opt import build_blockmask_unidrive
from .constants import (
    NUSCENES_SYSTEM_PROMPT,
    NUSCENES_USER_PROMPT_TEMPLATE,
    NUSCENES_VIEW_TOKENS,
    TARGET_SENSOR_ORDER,
    OPENPI_ATTENTION_MASK_VALUE,
    DEFAULT_PERM_INDICES,
    _NAV_CMD_FIXED,
)
from .utils import (
    make_att_2d_masks,
    sample_beta,
    create_sinusoidal_pos_embedding,
    permute_metas_per_camera_fields,
)
from .modules import OccLatentDecoder, DenseDepthNet

from simple_vla.core.box3d import *
from simple_vla.ops import feature_maps_format
from .unified_perception_decoder import UnifiedPerceptionDecoder
from simple_vla.plugin.unidrivevla.wrappers.packed_temporal_memory import (
    PackedTemporalMemory,
    cold_det_tracking_state,
    cold_packed_memory_like_unpack_none,
)

@dataclass
class DrivingBatch:
    images: torch.Tensor
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    command: Optional[torch.Tensor]
    ego_status: Optional[torch.Tensor]
    view_token_ids: Optional[torch.Tensor] = None
    traj_answer_ids: Optional[torch.Tensor] = None
    traj_answer_mask: Optional[torch.Tensor] = None
    traj_labels: Optional[torch.Tensor] = None
    prompt_lens: Optional[List[int]] = None

class QwenConfig:
    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:
    num_hidden_layers = int(variant.split('_')[-1][:-1])
    if variant.startswith("qwen3_vl_8b"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_8b_expert"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=2048,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024,
            intermediate_size=2048,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

@HEADS.register_module()
class QwenVL3APlanningHead(nn.Module):
    def __init__(
        self,
        pretrained_path,
        vlm_variant: Literal["2b", "8b"] = "2b",
        action_dim: int = 2,
        action_horizon: int = 6,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        num_sample_steps: int = 10,
        enable_knowledge_insulation: bool = False,
        train_vlm: bool = False,
        enable_traj_ar: bool = False,
        traj_ar_target: Literal["waypoint", "raw_delta", "norm_delta"] = "waypoint",
        map_bound_dis_thresh: float = 1.0,
        map_dir_dis_thresh: float = 2.0,
        x_min: float = -13.97,
        x_max: float = 11.77,
        y_min: float = -2.02,
        y_max: float = 55.79,
        occ_aux_layers_1based: Optional[List[int]] = None,
        attn_implementation: Literal["eager", "sdpa", "flex"] = "flex",
        inference_attn_impl: Literal["eager", "sdpa"] = "eager",
        unified_decoder_cfg: dict = None,
        occworld_vae_config: Optional[dict] = None,
        occworld_vae_path: Optional[str] = None,
        with_depth_supervision: bool = False,
        num_depth_bins: int = 80,
        depth_range: tuple = (1.0, 60.0),
        depth_supervision_source: Literal["input", "output"] = "input",
        feature_source: Literal["raw", "deepstack"] = "deepstack",
        feat_grad: Optional[bool] = None,
        use_tau0_pred: bool = False,
        vlm_fusion_cfg: Optional[dict] = None,
        feature_fusion_cfg: Optional[dict] = None,
        vlm_grad_scale: float = 1.0,
        lora_cfg: Optional[dict] = None,
        lora_merge_save_dir: Optional[str] = None,
        driving_deepstack: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.min_period = min_period
        self.max_period = max_period
        self.num_sample_steps = num_sample_steps

        assert not (enable_knowledge_insulation and not enable_traj_ar), \
            "enable_knowledge_insulation=True requires enable_traj_ar=True"
        self.enable_knowledge_insulation = enable_knowledge_insulation
        self._inference_attn_impl = inference_attn_impl
        self.train_vlm = train_vlm
        self.enable_traj_ar = enable_traj_ar
        assert traj_ar_target in ("waypoint", "raw_delta", "norm_delta"), \
            f"traj_ar_target must be one of waypoint/raw_delta/norm_delta, got {traj_ar_target}"
        self.traj_ar_target = traj_ar_target

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        if occ_aux_layers_1based is None:
            occ_aux_layers_1based = [4, 14, 24]
        self.occ_aux_layers = [int(x) - 1 for x in occ_aux_layers_1based]

        self.use_tau0_pred = False

        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            perception_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
            action_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            perception_expert_cfg = get_qwen_config('qwen3_28l')
            action_expert_cfg = get_qwen_config('qwen3_28l')

        self.lora_merge_save_dir = lora_merge_save_dir

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            qwen3_vl_cfg,
            perception_expert_cfg,
            action_expert_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
        )

        self.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert._vla_attn_impl = self.attn_implementation

        if occworld_vae_config is None:
            raise ValueError("occworld_vae_config must be provided via config.")
        # if occworld_vae_path is None:
        #     raise ValueError("occworld_vae_path must be provided via config.")

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg must be provided via config.")

        self.embed_dims = unified_decoder_cfg.get("embed_dims", 256)
        self.vlm_hidden_size = perception_expert_cfg.hidden_size

        self.num_det_queries = unified_decoder_cfg.get("det_instance_bank", {}).get("num_anchor", 900)
        self.num_map_queries = unified_decoder_cfg.get("map_instance_bank", {}).get("num_anchor", 100)
        self.num_occ_queries = 625

        self.with_motion = "motion" in unified_decoder_cfg.get("task_select", [])

        if self.with_motion:
            self.num_motion_queries = unified_decoder_cfg.get("motion_instance_bank", {}).get("num_anchor", 900)
            self.motion_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
            self.motion_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        else:
            self.num_motion_queries = 0
            self.motion_proj_up = None
            self.motion_proj_down = None

        self.ego_status_dim = unified_decoder_cfg.get("ego_refine_layer", {}).get("status_dims", 10)

        self.det_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.map_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)

        self.det_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        self.map_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        self.ego_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.ego_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        vision_hidden_size = self.qwen3_vl_with_expert.qwen3_vl.config.vision_config.hidden_size
        self.feature_source = feature_source

        if feat_grad is None:
            raise ValueError("feat_grad must be provided via config.")
        self.feat_grad = bool(feat_grad)

        if self.feature_source == "raw":
            proj_input_dim = vision_hidden_size
            num_proj_layers = 4
        elif self.feature_source == "deepstack":
            proj_input_dim = vision_hidden_size * 2
            num_proj_layers = 3
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        self.num_feature_scales = num_proj_layers

        self.feature_map_proj = nn.ModuleList([
            nn.Linear(proj_input_dim, self.embed_dims)
            for _ in range(num_proj_layers)
        ])

        self.fusion_weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dims, num_proj_layers),
                nn.Softmax(dim=-1)
            ) for _ in range(num_proj_layers)
        ])
        self.num_occ_queries = 625
        self.occ_queries = nn.Parameter(torch.randn(1, self.num_occ_queries, self.vlm_hidden_size))

        self.with_depth_supervision = with_depth_supervision
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range
        self.depth_supervision_source = depth_supervision_source

        if self.with_depth_supervision:
            self.depth_net = DenseDepthNet(
                embed_dims=self.embed_dims,
                in_channels=qwen3_vl_cfg.hidden_size,
                num_depth_layers=1,
                equal_focal=100,
                max_depth=60,
                loss_weight=1.0,
            )

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg is required. Legacy det_vla_head/map_vla_head are no longer supported.")

        self.unified_decoder = build_head(unified_decoder_cfg)

        self.occ_decoder = OccLatentDecoder(
            qwen_dim=perception_expert_cfg.hidden_size,
            occworld_vae_config=occworld_vae_config,
            # pretrained_vae_path=occworld_vae_path,
        )
        self.action_in_proj = nn.Linear(action_dim, action_expert_cfg.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_cfg.hidden_size, action_dim)

        status_in_features = 3 + self.ego_status_dim

        self.status_mlp = Mlp(
            in_features=status_in_features,
            hidden_features=action_expert_cfg.hidden_size,
            out_features=action_expert_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.hist_traj_steps = 4
        self.hist_traj_dim = 2
        self.hist_traj_encoder = Mlp(
            in_features=self.hist_traj_steps * self.hist_traj_dim,
            hidden_features=action_expert_cfg.hidden_size,
            out_features=action_expert_cfg.hidden_size,
            norm_layer=nn.LayerNorm,
        )

        self.action_time_mlp_in = nn.Linear(2 * action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_cfg.hidden_size, action_expert_cfg.hidden_size)

        if dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float16

        self.action_in_proj.to(target_dtype)
        self.status_mlp.to(target_dtype)
        self.hist_traj_encoder.to(target_dtype)
        self.action_time_mlp_in.to(target_dtype)
        self.action_time_mlp_out.to(target_dtype)

        self.det_proj_up.to(target_dtype)
        self.det_proj.to(target_dtype)
        self.map_proj_up.to(target_dtype)
        self.map_proj.to(target_dtype)
        self.ego_proj_up.to(target_dtype)
        self.ego_proj_down.to(target_dtype)
        if self.motion_proj_up is not None:
            self.motion_proj_up.to(target_dtype)
        if self.motion_proj_down is not None:
            self.motion_proj_down.to(target_dtype)

        self.unified_decoder.to(target_dtype)
        self.feature_map_proj.to(target_dtype)
        self.fusion_weight_generators.to(target_dtype)

        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        self.gradient_checkpointing_enable()
        self.gradient_checkpointing_enabled = True

        self.vlm_grad_scale = vlm_grad_scale
        self.driving_deepstack = driving_deepstack

        self.view_token_str_list = NUSCENES_VIEW_TOKENS
        self.view_token_ids = None

        self._cached_block_mask = None
        self._cached_block_mask_key = None
        self._cached_q_len_rounded = None

        self.adaptive_feature_fusion = torch.compile(
            self.adaptive_feature_fusion, mode="default", fullgraph=False, dynamic=True
        )

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = []
            for t in self.view_token_str_list:
                tid = tokenizer.convert_tokens_to_ids(t)
                ids.append(tid)
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        gc_kwargs = {"use_reentrant": False}
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_action_expert.gradient_checkpointing_disable()

    def merge_and_save_lora(self, save_dir: Optional[str] = None) -> None:
        target = save_dir or self.lora_merge_save_dir
        if target is None:
            logging.warning(
                "[LoRA] merge_and_save_lora called but no save_dir configured. "
                "Set lora_merge_save_dir in the config or pass it explicitly."
            )
            return
        self.qwen3_vl_with_expert.merge_and_save_lora(target)

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        t = sample_beta(self.time_beta_alpha, self.time_beta_beta, bsize, device)
        t = t * 0.999 + 0.001
        return t.to(dtype=torch.float32, device=device)

    def norm_delta(self, delta_meter: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([0.0233, 2.2707], device=delta_meter.device, dtype=delta_meter.dtype)
        std = torch.tensor([0.3427, 1.8668], device=delta_meter.device, dtype=delta_meter.dtype)
        return (delta_meter - mu) / (std + 1e-6)

    def denorm_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([0.0233, 2.2707], device=delta_norm.device, dtype=delta_norm.dtype)
        std = torch.tensor([0.3427, 1.8668], device=delta_norm.device, dtype=delta_norm.dtype)
        return delta_norm * (std + 1e-6) + mu

    def embed_perception(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage1_outs: dict,
    ):
        query_select = self.unified_decoder.query_select
        motion_token_256 = stage1_outs.get('motion_token', None)

        proj_dtype = self.det_proj_up.weight.dtype

        parts_embs = []
        parts_pad  = []
        parts_att  = []
        perception_lengths = {'det': 0, 'map': 0, 'occ': 0, 'ego': 0, 'motion': 0}

        # ── det ──────────────────────────────────────────────────────────────
        if 'det' in query_select and len(stage1_outs.get('det_predictions', [])) > 0:
            det_feat   = stage1_outs['det_instance_feature']
            det_anchor = stage1_outs['det_predictions'][-1]
            det_embs   = self.det_proj_up(det_feat.to(dtype=proj_dtype))
            anchor_embed_256 = self.unified_decoder.det_anchor_encoder(det_anchor)
            anchor_embed_vlm = self.det_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            det_embs = (det_embs + anchor_embed_vlm).to(dtype)
            det_n = det_embs.shape[1]
            parts_embs.append(det_embs)
            parts_pad.append(torch.ones( (batch_size, det_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, det_n), dtype=torch.bool, device=device))
            perception_lengths['det'] = det_n

        # ── map ──────────────────────────────────────────────────────────────
        if 'map' in query_select and len(stage1_outs.get('map_predictions', [])) > 0:
            map_feat   = stage1_outs['map_instance_feature']
            map_anchor = stage1_outs['map_predictions'][-1]
            map_embs   = self.map_proj_up(map_feat.to(dtype=proj_dtype))
            anchor_embed_out = self.unified_decoder.map_anchor_encoder(map_anchor)
            anchor_embed_256 = anchor_embed_out[0] if isinstance(anchor_embed_out, (tuple, list)) else anchor_embed_out
            anchor_embed_vlm = self.map_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            map_embs = (map_embs + anchor_embed_vlm).to(dtype)
            map_n = map_embs.shape[1]
            parts_embs.append(map_embs)
            parts_pad.append(torch.ones( (batch_size, map_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, map_n), dtype=torch.bool, device=device))
            perception_lengths['map'] = map_n

        # ── occ (always included when the module exists) ──────────────────────
        occ_embs = self.occ_queries.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        occ_n = occ_embs.shape[1]
        parts_embs.append(occ_embs)
        parts_pad.append(torch.ones( (batch_size, occ_n), dtype=torch.bool, device=device))
        parts_att.append(torch.zeros((batch_size, occ_n), dtype=torch.bool, device=device))
        perception_lengths['occ'] = occ_n

        # ── ego ──────────────────────────────────────────────────────────────
        if 'ego' in query_select:
            ego_feat   = stage1_outs['ego_instance_feature']
            ego_anchor = stage1_outs.get('ego_anchor', None)
            ego_embs   = self.ego_proj_up(ego_feat.to(dtype=proj_dtype))
            if hasattr(self.unified_decoder, 'ego_anchor_encoder') and ego_anchor is not None:
                ego_anchor_embed_256 = self.unified_decoder.ego_anchor_encoder(ego_anchor)
                ego_anchor_embed_vlm = self.ego_proj_up(ego_anchor_embed_256.to(dtype=proj_dtype))
                ego_embs = (ego_embs + ego_anchor_embed_vlm).to(dtype)
            else:
                ego_embs = ego_embs.to(dtype)
            ego_n = ego_embs.shape[1]
            parts_embs.append(ego_embs)
            parts_pad.append(torch.ones( (batch_size, ego_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, ego_n), dtype=torch.bool, device=device))
            perception_lengths['ego'] = ego_n

        # ── motion ───────────────────────────────────────────────────────────
        if motion_token_256 is not None:
            motion_token_vlm = self.motion_proj_up(motion_token_256.to(dtype=proj_dtype)).to(dtype)
            motion_n = motion_token_vlm.shape[1]
            parts_embs.append(motion_token_vlm)
            parts_pad.append(torch.ones( (batch_size, motion_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, motion_n), dtype=torch.bool, device=device))
            perception_lengths['motion'] = motion_n

        # ── assemble ─────────────────────────────────────────────────────────
        if parts_embs:
            perception_embs      = torch.cat(parts_embs, dim=1)
            perception_pad_masks = torch.cat(parts_pad,  dim=1)
            perception_att_masks = torch.cat(parts_att,  dim=1)
        else:
            perception_embs      = torch.empty((batch_size, 0, self.vlm_hidden_size), dtype=dtype, device=device)
            perception_pad_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            perception_att_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)

        return perception_embs, perception_pad_masks, perception_att_masks, perception_lengths

    def project_and_reshape_features(
        self,
        source_features,
        bsz: int,
        all_image_grids,
        feature_source: str,
    ):
        feature_maps = []

        if source_features is None:
            return feature_maps

        if not isinstance(source_features, list):
            source_features = [source_features]

        projected_features = []
        for i, feat in enumerate(source_features):
            if i < len(self.feature_map_proj):
                feat = feat.to(self.feature_map_proj[i].weight.dtype)
                feat_proj = self.feature_map_proj[i](feat)
                projected_features.append(feat_proj)
            else:
                projected_features.append(feat)

        if all_image_grids is not None and len(all_image_grids) > 0:
            h_grid = int(all_image_grids[0, 1].item())
            w_grid = int(all_image_grids[0, 2].item())
            num_views = 6

            for ds_feat in projected_features:
                if ds_feat.dim() != 2:
                    continue

                feat_reshaped = None

                if feature_source == "raw":
                    merge_size = 2
                    h_block = h_grid // merge_size
                    w_block = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_grid * w_grid

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_vis = ds_feat.view(bsz, num_views, h_block, w_block, merge_size, merge_size, -1)
                            feat_vis = feat_vis.permute(0, 1, 2, 4, 3, 5, 6)
                            feat_reshaped = feat_vis.reshape(bsz, num_views, h_grid, w_grid, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                elif feature_source == "deepstack":
                    merge_size = 2
                    h_ds = h_grid // merge_size
                    w_ds = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_ds * w_ds

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_reshaped = ds_feat.view(bsz, num_views, h_ds, w_ds, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                if feat_reshaped is not None:
                    feature_maps.append(feat_reshaped)

        if len(feature_maps) > 0:
            feature_maps = self.adaptive_feature_fusion(feature_maps)

        return feature_maps

    def adaptive_feature_fusion(self, feature_maps):
        if len(feature_maps) <= 1:
            return feature_maps

        B, N, C, H, W = feature_maps[0].shape
        fused_maps = []

        for i, feat in enumerate(feature_maps):
            feat_flat = feat.view(B * N, C, H, W)

            fusion_weights = self.fusion_weight_generators[i](feat_flat)

            weights = fusion_weights.view(B, N, len(feature_maps), 1, 1, 1)

            current_fused = 0
            for j in range(len(feature_maps)):
                other_feat = feature_maps[j]
                w = weights[:, :, j]

                if other_feat.shape[-2:] != (H, W):
                    ref = F.interpolate(
                        other_feat.flatten(0, 1), size=(H, W), mode='bilinear'
                    ).view(B, N, C, H, W)
                else:
                    ref = other_feat

                current_fused += ref * w

            fused_maps.append(current_fused)

        return fused_maps

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def _build_driving_batch(
        self,
        img: torch.Tensor,
        command=None,
        ego_status=None,
        hist_traj=None,
        gt_trajs=None,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        permute_indices = [0, 2, 1, 4, 5, 3]
        images = img[:, permute_indices]

        image_masks = {f"cam{i}": torch.ones((b,), device=device, dtype=torch.bool) for i in range(6)}

        view_token_ids = self._get_view_token_ids(device)

        if command is not None:
            if not torch.is_tensor(command):
                 try:
                     command = torch.stack(command)
                 except:
                     command = torch.tensor(command)

            command = command.to(device)
            cmd_idx = command.view(-1).long()

            idx_list = cmd_idx.tolist()
        else:
            idx_list = [2] * b

        nav_cmd_texts = [_NAV_CMD_FIXED.get(i, _NAV_CMD_FIXED[2]) for i in idx_list]

        hist_traj_np = hist_traj.detach().cpu().numpy()

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        sys_content_ids = tokenizer.encode(NUSCENES_SYSTEM_PROMPT, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]

        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            points_str = [f"({pt[0]:+07.2f}, {pt[1]:+07.2f})" for pt in hist_traj_np[i]]
            hist_traj_str = f"[PT_HIST, {', '.join(points_str)}]"

            user_prompt_text = NUSCENES_USER_PROMPT_TEMPLATE.format(
                nav_cmd=nav_cmd_texts[i],
                hist_traj_str=hist_traj_str,
            )
            user_content_ids = tokenizer.encode(user_prompt_text, add_special_tokens=False)

            full_ids = sys_part + user_start_part + user_content_ids + user_end_assistant_start_part

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        tokenized_prompt = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        tokenized_prompt_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        traj_answer_ids = None
        traj_answer_mask = None
        traj_labels = None
        if gt_trajs is not None:
            if self.traj_ar_target == "waypoint":
                gt_trajs_target = torch.cumsum(gt_trajs, dim=1)
            elif self.traj_ar_target == "norm_delta":
                gt_trajs_target = self.norm_delta(gt_trajs)
            else:
                gt_trajs_target = gt_trajs

            gt_trajs_np = gt_trajs_target.detach().cpu().numpy()
            answer_ids_list = []
            for i in range(b):
                pts = gt_trajs_np[i]
                pts_str = ", ".join(f"({p[0]:+07.2f}, {p[1]:+07.2f})" for p in pts)
                answer_text = f"[PT, {pts_str}]"
                answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
                answer_token_ids = answer_token_ids + [im_end_id, nl_id]
                answer_ids_list.append(torch.tensor(answer_token_ids, dtype=torch.long))

            traj_answer_ids = pad_sequence(
                answer_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)

            FIXED_ANSWER_MAX_LEN = 128
            curr_ans_len = traj_answer_ids.shape[1]
            if curr_ans_len < FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = F.pad(
                    traj_answer_ids, (0, FIXED_ANSWER_MAX_LEN - curr_ans_len),
                    value=tokenizer.pad_token_id
                )
            elif curr_ans_len > FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = traj_answer_ids[:, :FIXED_ANSWER_MAX_LEN]

            traj_answer_mask = (traj_answer_ids != tokenizer.pad_token_id).long()
            traj_labels = traj_answer_ids.masked_fill(traj_answer_mask == 0, -100)

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            view_token_ids=view_token_ids,
            traj_answer_ids=traj_answer_ids,
            traj_answer_mask=traj_answer_mask,
            traj_labels=traj_labels,
        )


    def embed_prefix(self, batch: DrivingBatch):
        images_tensor = batch.images
        device = self.action_in_proj.weight.device

        image_features, feature_lens, all_image_grids, deepstack_features, raw_features = self.qwen3_vl_with_expert.embed_image_tensor(images_tensor)

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        vision_end_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_end_token_id
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        nl_id = self.qwen3_vl_with_expert.processor.tokenizer.encode("\n", add_special_tokens=False)[0]

        bs = images_tensor.shape[0]
        num_views_per_sample = 6
        view_token_ids = self._get_view_token_ids(device)

        prefix_input_ids_list = []

        FIXED_PREFIX_MAX_LEN = 3740

        for b_idx in range(bs):
            sample_input_ids = []

            for v_idx in range(num_views_per_sample):
                img_len = feature_lens[b_idx * num_views_per_sample + v_idx]
                ids = [view_token_ids[v_idx].item(), nl_id, vision_start_id] + \
                    [image_token_id] * img_len + \
                    [vision_end_id, nl_id]
                sample_input_ids.extend(ids)

            prompt_mask = batch.tokenized_prompt_mask[b_idx].bool()
            prompt_ids = batch.tokenized_prompt[b_idx][prompt_mask].tolist()
            sample_input_ids.extend(prompt_ids)

            prefix_input_ids_list.append(torch.tensor(sample_input_ids, dtype=torch.long, device=device))

        prompt_lens_list = [ids.shape[0] for ids in prefix_input_ids_list]

        FIXED_TOTAL_LEN = FIXED_PREFIX_MAX_LEN + (
            batch.traj_answer_ids.shape[1] if batch.traj_answer_ids is not None else 0
        )

        if batch.traj_answer_ids is not None:
            full_ids_list = []
            for b_idx in range(bs):
                p_ids = prefix_input_ids_list[b_idx]
                a_ids = batch.traj_answer_ids[b_idx]
                if p_ids.shape[0] > FIXED_PREFIX_MAX_LEN:
                    p_ids = p_ids[:FIXED_PREFIX_MAX_LEN]
                    prompt_lens_list[b_idx] = FIXED_PREFIX_MAX_LEN
                full_ids_list.append(torch.cat([p_ids, a_ids], dim=0))

            prefix_input_ids = pad_sequence(
                full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            curr_len = prefix_input_ids.shape[1]
            if curr_len < FIXED_TOTAL_LEN:
                prefix_input_ids = F.pad(prefix_input_ids, (0, FIXED_TOTAL_LEN - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > FIXED_TOTAL_LEN:
                prefix_input_ids = prefix_input_ids[:, :FIXED_TOTAL_LEN]
        else:
            prefix_input_ids = pad_sequence(prefix_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            curr_len = prefix_input_ids.shape[1]
            if curr_len < FIXED_PREFIX_MAX_LEN:
                prefix_input_ids = F.pad(prefix_input_ids, (0, FIXED_PREFIX_MAX_LEN - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > FIXED_PREFIX_MAX_LEN:
                prefix_input_ids = prefix_input_ids[:, :FIXED_PREFIX_MAX_LEN]
                truncated_mask = (prefix_input_ids == image_token_id)
                valid_img_tokens = truncated_mask.sum().item()
                if valid_img_tokens < image_features.shape[0]:
                    image_features = image_features[:valid_img_tokens]

        prompt_only_len = max(prompt_lens_list)

        prefix_pad_masks = (prefix_input_ids != tokenizer.pad_token_id)

        input_embeds = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()(prefix_input_ids)

        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.sum() != image_features.shape[0]:
            target_count = image_mask.sum()
            current_count = image_features.shape[0]
            if current_count > target_count:
                image_features = image_features[:target_count]
            else:
                raise ValueError(f"Visual features mismatch! Feat: {current_count}, Tokens: {target_count}")

        input_embeds = input_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features.to(input_embeds.dtype))

        prefix_att_marks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
        for b_idx in range(bs):
            p_len = prompt_lens_list[b_idx]
            prefix_att_marks[b_idx, :p_len] = 1
        if batch.traj_answer_ids is not None:
            for b_idx in range(bs):
                p_len = prompt_lens_list[b_idx]
                real_ans_len = batch.traj_answer_mask[b_idx].sum().item()
                prefix_att_marks[b_idx, p_len:p_len + real_ans_len] = 1

        batch.prompt_lens = prompt_lens_list

        return input_embeds, prefix_pad_masks, prefix_att_marks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len

    def _maybe_get_status_features(
        self,
        batch: DrivingBatch,
        ego_status_pred: Optional[torch.Tensor],
        *,
        use_gt: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = batch.command.device
        cmd_onehot = batch.command.to(device=device, dtype=dtype)

        if use_gt:
            if batch.ego_status is None:
                raise ValueError("ego_status is required during training")
            ego = batch.ego_status.to(device=device, dtype=dtype)
            if ego.shape[-1] >= self.ego_status_dim:
                ego = ego[..., : self.ego_status_dim]
            else:
                pad = cmd_onehot.new_zeros((ego.shape[0], self.ego_status_dim - ego.shape[-1]))
                ego = torch.cat([ego, pad], dim=-1)
            return torch.cat([cmd_onehot, ego], dim=-1)

        if ego_status_pred is None:
            ego_status_pred = cmd_onehot.new_zeros((cmd_onehot.shape[0], self.ego_status_dim))
        return torch.cat([cmd_onehot, ego_status_pred.to(dtype=dtype)], dim=-1)

    def embed_suffix(
        self,
        batch: DrivingBatch,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        ego_status_pred: Optional[torch.Tensor] = None,
        use_gt_status: bool = False,
        hist_traj: Optional[torch.Tensor] = None,
    ):
        device = actions.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype

        B = actions.shape[0]

        status_input = self._maybe_get_status_features(
            batch, ego_status_pred, use_gt=use_gt_status, dtype=dtype,
        )
        status_emb = self._apply_checkpoint(self.status_mlp, status_input)
        status_emb = status_emb.unsqueeze(1)

        if hist_traj is not None:
            if hist_traj.dtype != dtype:
                hist_traj = hist_traj.to(dtype=dtype)
            hist_traj_emb = self._apply_checkpoint(self.hist_traj_encoder, hist_traj.flatten(1))
            status_emb = torch.cat([hist_traj_emb.unsqueeze(1), status_emb], dim=1)

        num_status_tokens = status_emb.shape[1]

        if actions.dtype != dtype:
            actions = actions.to(dtype=dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, self.min_period, self.max_period, device=device
        )
        fused_input = torch.cat([action_emb, time_emb.unsqueeze(1).expand_as(action_emb)], dim=-1).to(dtype)

        def mlp_func(x):
            x = self.action_time_mlp_in(x)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, fused_input)
        suffix_emb = torch.cat([status_emb, action_time_emb], dim=1)

        suffix_len = suffix_emb.shape[1]
        pad_masks = torch.ones((B, suffix_len), dtype=torch.bool, device=device)
        att_marks = [1] * num_status_tokens + [0] * (suffix_len - num_status_tokens)
        att_marks = torch.tensor(att_marks, dtype=torch.bool, device=device)[None, :].expand(B, -1)

        return suffix_emb, pad_masks, att_marks

    def get_position_ids(self, input_ids, image_grid_thw, pad_masks):
        attention_mask = pad_masks.long()
        position_ids, rope_deltas = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def prepare_for_deformable_aggregation(self, feature_maps):
        if not feature_maps:
            return []
        return feature_maps_format(feature_maps)

    def forward_pre_stage1(self, img, timestamp, projection_mat, image_wh, gt_ego_fut_cmd, ego_status, hist_traj, **kwargs):
        permute_indices = [0, 2, 1, 4, 5, 3]
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        batch = self._build_driving_batch(
            img=img,
            command=gt_ego_fut_cmd,
            ego_status=ego_status,
            hist_traj=hist_traj,
        )
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.embed_prefix(batch)

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
        )

        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_dtype = next(self.unified_decoder.parameters()).dtype
        head_device = device

        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': timestamp,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_dtype) if projection_mat is not None else None,
            'image_wh': image_wh.to(device=head_device, dtype=head_dtype) if image_wh is not None else None,
        }

        return feature_maps_daf, perception_metas, batch, prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len

    def forward_pre_stage2(self, stage1_outs, batch, prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len):
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, device, dtype, stage1_outs
        )

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(dtype)

        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        _ds_embeds = deepstack_features if self.driving_deepstack and deepstack_features is not None else None
        _vis_masks = (
            (prefix_input_ids == self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id)
            if _ds_embeds is not None else None
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = self._inference_attn_impl

        _, past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
            deepstack_visual_embeds=_ds_embeds,
            visual_pos_masks=_vis_masks,
        )

        perception_len = perception_embs.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
        perception_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
        perception_full_att_2d = torch.cat([prefix_pad_2d, perception_att_2d], dim=2)

        perception_full_att_2d_4d = self._prepare_attention_masks_4d(perception_full_att_2d).to(dtype)

        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.qwen3_perception_expert.config._attn_implementation = self._inference_attn_impl

        (_, perception_out, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=perception_full_att_2d_4d,
            position_ids=perception_pos_ids_3d,
            past_key_values=past_key_values,
            inputs_embeds=[None, perception_embs, None],
            use_cache=True,
        )

        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        occ_len = perception_lengths['occ']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        o0 = m1
        o1 = o0 + occ_len
        e0 = o1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        stage2_outs = None

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            target_dtype = self.det_proj.weight.dtype
            proj_dtype = self.det_proj.weight.dtype
            motion_token_256 = stage1_outs.get('motion_token', None)
            ego_feat_stage1 = stage1_outs['ego_instance_feature']

            det_feat_fused = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            map_feat_fused = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            ego_feat_fused = (
                self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if ego_out is not None
                else ego_feat_stage1.to(torch.float32)
            )
            motion_feat_fused = (
                self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                if motion_token_256 is not None
                else None
            )

            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)
        return vlm_enhanced, perception_pad_masks, perception_len, perception_pos_ids_1d, max_prefix_pos, past_key_values, occ_out, occ_len

    def prepare_output(self, stage2_outs, batch, prefix_pad_masks, perception_pad_masks, num_steps, device, noise, use_gt_ego_status, hist_traj, perception_len, perception_pos_ids_1d, max_prefix_pos, bsz, past_key_values, dtype):
        det_result, map_result = self.unified_decoder.post_process(stage2_outs)

        cached_pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
        dt_val = -1.0 / num_steps
        dt_val = torch.tensor(dt_val, dtype=torch.float32, device=device)

        x_t = noise
        time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)

        if use_gt_ego_status:
            ego_status_pred = None
        elif stage2_outs and 'ego_status_list' in stage2_outs and len(stage2_outs['ego_status_list']) > 0:
            ego_status_pred = stage2_outs['ego_status_list'][-1].squeeze(1).to(torch.float32)
        else:
            ego_status_pred = x_t.new_zeros((bsz, self.ego_status_dim), dtype=torch.float32)

        if perception_len > 0:
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        else:
            max_perception_pos = max_prefix_pos

        while time_tensor >= -dt_val / 2:
            expanded_time = time_tensor.expand(bsz)
            v_t = self._denoise_step(
                batch, cached_pad_masks, past_key_values, x_t.to(dtype), expanded_time.to(dtype),
                max_perception_pos, ego_status_pred=ego_status_pred, use_gt_status=use_gt_ego_status, hist_traj=hist_traj,
            )
            x_t = x_t + dt_val * v_t
            time_tensor += dt_val

        traj_pred_meter_deltas = self.denorm_delta(x_t)
        zeros = torch.zeros((bsz, 1, 2), device=device, dtype=x_t.dtype)
        traj_pred_points_meter = zeros + torch.cumsum(traj_pred_meter_deltas, dim=1)


        return traj_pred_points_meter, det_result, map_result


    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        hist_traj=None,
        use_gt_ego_status: bool = False,
        return_occ_pred: bool = False,
        **kwargs,
    ):

        feature_maps_daf, perception_metas, batch, prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len = self.forward_pre_stage1(img, timestamp, projection_mat, image_wh, gt_ego_fut_cmd, ego_status, hist_traj, **kwargs)
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype
        num_steps = int(self.num_sample_steps if num_steps is None else num_steps)

        if noise is None:
            noise = self.sample_noise((bsz, self.action_horizon, self.action_dim), device)

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)

        vlm_enhanced, perception_pad_masks, perception_len, perception_pos_ids_1d, max_prefix_pos, past_key_values, occ_out, occ_len = self.forward_pre_stage2(
            stage1_outs, batch, prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len)
            
        stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)

        traj_pred_points_meter, det_result, map_result = self.prepare_output(
            stage2_outs, batch, prefix_pad_masks, perception_pad_masks, num_steps, device, noise, use_gt_ego_status, hist_traj, perception_len, perception_pos_ids_1d, max_prefix_pos, bsz, past_key_values, dtype)

        out = {
            "traj": traj_pred_points_meter,
            "det": det_result,
            "map": map_result,
        }
        if return_occ_pred and occ_len > 0:
            out["occ_logits"] = self.occ_decoder(occ_out.to(torch.float32))
        return out


    def _denoise_step(self, batch, cached_pad_masks, past_key_values, x_t, timestep, max_cached_position_ids, *, ego_status_pred: Optional[torch.Tensor] = None, use_gt_status: bool = False, hist_traj=None):

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            batch,
            x_t,
            timestep,
            ego_status_pred=ego_status_pred,
            use_gt_status=use_gt_status,
            hist_traj=hist_traj,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = cached_pad_masks.shape[0]
        cached_len = cached_pad_masks.shape[1]

        cached_pad_2d_masks = cached_pad_masks[:, None, :].expand(batch_size, suffix_len, cached_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([cached_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(x_t.dtype)

        suffix_range = torch.arange(1, suffix_len + 1, device=max_cached_position_ids.device).view(1, -1).expand(batch_size, -1)
        suffix_pos_ids_1d = max_cached_position_ids + suffix_range

        position_ids = torch.stack([
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
            suffix_pos_ids_1d,
        ], dim=0)

        self.qwen3_vl_with_expert.qwen3_action_expert.config._attn_implementation = self._inference_attn_impl

        outputs_embeds, _, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]

        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)

        model_out = self.action_out_proj(suffix_out)

        return model_out

    def forward_ar_batch(self, ar_batch):
        input_ids = ar_batch['ar_input_ids']
        labels    = ar_batch['ar_labels']
        device    = input_ids.device
        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        raw_pv = ar_batch.get('ar_pixel_values', None)
        pixel_values_tensor = None
        if raw_pv is not None:
            flat = []
            for item in raw_pv:
                if isinstance(item, (list, tuple)):
                    for t in item:
                        flat.append(t.to(device))
                else:
                    flat.append(item.to(device))
            if flat:
                pixel_values_tensor = torch.cat(flat, dim=0)

        raw_thw = ar_batch.get('ar_image_grid_thw', None)
        flat_image_grid_thw = None
        if raw_thw is not None:
            if isinstance(raw_thw, (list, tuple)):
                parts = [t.to(device) for t in raw_thw if t is not None]
                if parts:
                    flat_image_grid_thw = torch.cat(parts, dim=0)
            elif torch.is_tensor(raw_thw):
                if raw_thw.dim() == 3:
                    flat_image_grid_thw = raw_thw.reshape(-1, 3).to(device)
                else:
                    flat_image_grid_thw = raw_thw.to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        position_ids, _ = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=flat_image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.qwen3_vl_with_expert.qwen3_vl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_tensor,
                image_grid_thw=flat_image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"
        loss_vlm = outputs.loss

        return dict(
            loss_ar=self.ar_loss_weight * loss_vlm,
            loss_vlm_raw=loss_vlm.detach(),
        )
