import os

vlm_pretrained_path = os.environ.get("VLM_PRETRAINED_PATH", "owl10/UniDriveVLA_Nusc_Base_Stage1")
occworld_vae_path = None #os.environ.get("OCCWORLD_VAE_PATH", "/home/ukenryu/python_try_new/unidrivevla/occvae_latest.pth")

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
decouple_attn = True
with_quality_estimation = True
ego_feature_map_scale = (544 // 16, 960 // 16)

num_classes = 10
num_map_classes = 3
num_sample = 20
roi_size = (30, 60)

single_frame_layer = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine',
]
temporal_frame_layer = [
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine',
]
unified_decoder_operation_order = single_frame_layer * num_single_frame_decoder + \
                                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

unified_decoder_cfg = dict(
    type="UnifiedPerceptionDecoder",
    embed_dims=embed_dims,
    task_select=["det", "map", "ego", "motion"],
    query_select=["det", "map", "ego", "motion"],
    num_stage1_layers=3,
    num_stage2_layers=3,
    num_single_frame_decoder=1,
    cls_threshold_to_reg=0.05,
    decouple_attn=decouple_attn,
    use_vlm_in_stage2=True,
    operation_order=unified_decoder_operation_order,
    det_instance_bank=dict(
        type="InstanceBank",
        num_anchor=900,
        embed_dims=embed_dims,
        anchor="nuScenes/data/kmeans/kmeans_det_900.npy",
        anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        num_temp_instances=600,
        confidence_decay=0.6,
        feat_grad=False,
    ),
    map_instance_bank=dict(
        type="InstanceBank",
        num_anchor=100,
        embed_dims=embed_dims,
        anchor="nuScenes/data/kmeans/kmeans_map_100.npy",
        anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
        num_temp_instances=0,
        confidence_decay=0.6,
        feat_grad=True,
    ),
    ego_instance_bank=dict(
        type="EgoInstanceBank",
        embed_dims=embed_dims,
        anchor_type='nus',
        num_temp_instances=1,
        feature_map_scale=ego_feature_map_scale,
    ),
    det_anchor_encoder=dict(
        type="SparseBox3DEncoder",
        vel_dims=3,
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    ),
    map_anchor_encoder=dict(
        type="SparsePoint3DEncoder",
        embed_dims=embed_dims,
        num_sample=num_sample,
    ),
    motion_anchor_encoder=dict(
        type="SparseBox3DEncoder",
        vel_dims=3,
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    ),
    graph_model=dict(
        type="SeparateAttention",
        query_select=["det", "map", "ego", "motion"],
        separate_list=[["det"], ["map"]],
        decouple_list=[True, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
        ],
    ),
    temp_graph_model=dict(
        type="TemporalSeparateAttention",
        query_select=["det", "map", "ego", "motion"],
        query_list=[["det"], ["map"], ["ego"]],
        key_list=[["det"], ["map"], ["det", "map"]],
        decouple_list=[True, False, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
        ],
    ),
    inter_graph_model=dict(
        type="InteractiveAttention",
        query_select=["det", "map", "ego", "motion"],
        query_list=[["ego"]],
        key_list=[["det", "map"]],
        decouple_list=[False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=0.1),
        ],
    ),
    det_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]],
        ),
    ),
    map_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparsePoint3DKeyPointsGenerator",
            embed_dims=embed_dims,
            num_sample=num_sample,
            num_learnable_pts=3,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        ),
    ),
    ego_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=4,
        num_cams=6,
        attn_drop=0.15,
        use_deformable_func=True,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]],
        ),
    ),
    ffn=dict(
        type="AsymmetricFFN",
        in_channels=embed_dims * 2,
        pre_norm=dict(type="LN"),
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 4,
        num_fcs=2,
        ffn_drop=0.1,
        act_cfg=dict(type="ReLU", inplace=True),
    ),
    norm_layer=dict(type="LN", normalized_shape=embed_dims),
    det_refine_layer=dict(
        type="SparseBox3DRefinementModule",
        embed_dims=embed_dims,
        num_cls=num_classes,
        refine_yaw=True,
        with_quality_estimation=with_quality_estimation,
    ),
    map_refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=embed_dims,
        num_sample=num_sample,
        num_cls=num_map_classes,
    ),
    ego_refine_layer=dict(
        type="EgoStatusRefinementModule",
        embed_dims=embed_dims,
        status_dims=6,
    ),
    motion_refine_layer=dict(
        type="SparseMotionRefinementModule",
        embed_dims=embed_dims,
        fut_ts=12,
        fut_mode=6,
    ),
    det_decoder=dict(type="SparseBox3DDecoder"),
    map_decoder=dict(type="SparsePoint3DDecoder"),
    motion_decoder=dict(type="SparseMotionDecoder"),
    det_reg_weights=[2.0] * 3 + [1.0] * 7,
    map_reg_weights=[1.0] * 40,
    motion_anchor="nuScenes/data/kmeans/kmeans_motion_6.npy",
)

lora_cfg = dict(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
)

model = dict(
    type='UniDriveVLA',
    planning_head=dict(
        type='QwenVL3APlanningHead',
        pretrained_path=vlm_pretrained_path,
        action_dim=2,
        action_horizon=6,
        dtype='bfloat16',
        train_vlm=False,
        with_depth_supervision=False,
        depth_loss_weight=0.2,
        occworld_vae_config=dict(
            type='VAERes3D',
            encoder_cfg=dict(
                type='Encoder2D',
                ch=64,
                out_ch=64,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=128,
                double_z=False,
            ),
            decoder_cfg=dict(
                type='Decoder3D',
                ch=64,
                out_ch=128,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=64,
                give_pre_end=False,
            ),
            num_classes=18,
            expansion=8,
            vqvae_cfg=None,
        ),
        occworld_vae_path=occworld_vae_path,
        feat_grad=True,
        feature_source="raw",
        unified_decoder_cfg=unified_decoder_cfg,
        det_vla_head_cfg=None,
        map_vla_head_cfg=None,
        lora_cfg=lora_cfg,
        driving_deepstack=True,
        vlm_fusion_cfg=dict(type='direct'),
        feature_fusion_cfg=dict(type='none'),
        inference_attn_impl="sdpa",
    ),
)
