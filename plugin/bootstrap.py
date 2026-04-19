"""Import side effects: register all classes used by Stage-2 inference configs.

Call :func:`ensure_plugins_registered` before ``build_model`` / ``build_head`` so every
``type=...`` string in the config resolves in ``simple_vla.core.registry``.
"""
from __future__ import annotations


def ensure_plugins_registered() -> None:
    # ATTENTION registry (MultiheadFlashAttention + temporal/interactive graph ops)
    import simple_vla.plugin.models.attention  # noqa: F401
    import simple_vla.core.attention.separate_attn  # noqa: F401

    # PLUGIN_LAYERS: deformable aggregation, FFN, instance banks, coders, decoders
    import simple_vla.plugin.models.blocks  # noqa: F401
    import simple_vla.plugin.models.instance_bank  # noqa: F401
    import simple_vla.plugin.models.detection3d.detection3d_blocks  # noqa: F401
    import simple_vla.plugin.models.detection3d.decoder  # noqa: F401
    import simple_vla.plugin.models.detection3d.target  # noqa: F401
    import simple_vla.plugin.models.map.map_blocks  # noqa: F401
    import simple_vla.plugin.models.map.decoder  # noqa: F401
    import simple_vla.plugin.models.map.target  # noqa: F401
    import simple_vla.plugin.models.motion.decoder  # noqa: F401
    import simple_vla.plugin.models.motion.blocks  # noqa: F401
    import simple_vla.plugin.models.ego.instance_bank  # noqa: F401
    import simple_vla.plugin.models.ego.blocks  # noqa: F401

    import simple_vla.core.banks.instance_bank  # noqa: F401
    import simple_vla.core.banks.ego  # noqa: F401
    import simple_vla.core.banks.motion  # noqa: F401
    import simple_vla.core.blocks.blocks  # noqa: F401
    import simple_vla.core.blocks.motion  # noqa: F401
    import simple_vla.core.blocks.ego  # noqa: F401
    import simple_vla.core.coders.detection3d  # noqa: F401
    import simple_vla.core.coders.map  # noqa: F401
    import simple_vla.core.coders.motion  # noqa: F401

    # OccWorld VAE submodules (HEADS registry, used by OccLatentDecoder)
    import simple_vla.plugin.unidrivevla.vae.vae_2d_resnet  # noqa: F401
    import simple_vla.plugin.unidrivevla.vae.quantizer  # noqa: F401

    # Planning stack (order: unified decoder → planning head → detector)
    import simple_vla.plugin.unidrivevla.dense_heads.unified_perception_decoder  # noqa: F401
    import simple_vla.plugin.unidrivevla.dense_heads.modules  # noqa: F401
    import simple_vla.plugin.unidrivevla.dense_heads.qwenvl3_vla_planning_head  # noqa: F401
    import simple_vla.plugin.unidrivevla.detectors.unidrivevla  # noqa: F401
