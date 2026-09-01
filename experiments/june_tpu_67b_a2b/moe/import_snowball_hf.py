# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the public Snowball HF export into the vendored training pytree.

The Snowball trainer uses an ``ArrayStacked`` vendored model plus a separate
``pending_qb_betas`` router state. Public HF exports contain the already-applied
router bias and use one tensor group per layer. This importer reverses only the
layout transformation and initializes ``pending_qb_betas`` to zero so training
does not apply the exported bias a second time.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import levanter
from haliax.partitioning import set_mesh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.jax_utils import use_cpu_device

from experiments.grug.moe.model import GrugModelConfig as HfGrugModelConfig
from experiments.grug.moe.model import Transformer as HfTransformer
from experiments.grug.moe.model import grugmoe_inference_state_dict
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import SNOWBALL_CHAT_MODEL_CONFIG

logger = logging.getLogger(__name__)


def _linear_training_tensor(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, -1, -2)


def _unstacked_blocks(model: Transformer) -> tuple[Any, ...]:
    if model.stacked_blocks is None:
        raise ValueError("Snowball HF import/export requires use_array_stacked_blocks=True.")

    def take_layer(value: Any, layer_index: int) -> Any:
        if isinstance(value, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        if isinstance(value, jax.Array):
            return value[layer_index]
        return value

    return tuple(
        jax.tree.map(lambda value, layer_index=layer_index: take_layer(value, layer_index), model.stacked_blocks.stacked)
        for layer_index in range(model.stacked_blocks.num_layers)
    )


def _expected_hf_keys(model: Transformer) -> set[str]:
    keys = {
        "model.embed_tokens.weight",
        "model.embed_norm.weight",
        "model.embed_gated_norm.down_proj.weight",
        "model.embed_gated_norm.up_proj.weight",
        "model.norm.weight",
        "model.final_gated_norm.down_proj.weight",
        "model.final_gated_norm.up_proj.weight",
        "lm_head.weight",
    }
    per_layer = {
        "input_layernorm.weight",
        "attn_gated_norm.down_proj.weight",
        "attn_gated_norm.up_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.attn_gate.weight",
        "post_attention_layernorm.weight",
        "mlp_gated_norm.down_proj.weight",
        "mlp_gated_norm.up_proj.weight",
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_proj.weight",
        "mlp.experts.up_proj.weight",
        "mlp.experts.down_proj.weight",
    }
    if model.config.shared_expert_intermediate_dim > 0:
        per_layer.update(
            {
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
            }
        )
    for layer_index in range(model.config.num_layers):
        keys.update(f"model.layers.{layer_index}.{suffix}" for suffix in per_layer)
    return keys


def snowball_hf_state_dict(model: Transformer) -> dict[str, jax.Array]:
    """Return the public-HF tensor layout for a vendored Snowball model."""
    if model.stacked_blocks is None:
        raise ValueError("Snowball HF import/export requires use_array_stacked_blocks=True.")
    source = cast(Any, model)
    unstacked_view = SimpleNamespace(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=_unstacked_blocks(model),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
    )
    return grugmoe_inference_state_dict(cast(HfTransformer, unstacked_view))


def _checked_tensor(
    state_dict: dict[str, jax.Array],
    name: str,
    expected: jax.Array,
    *,
    linear: bool = False,
) -> jax.Array:
    value = state_dict[name]
    if linear:
        value = _linear_training_tensor(value)
    if value.shape != expected.shape:
        raise ValueError(f"HF tensor {name!r} has shape {value.shape}; expected {expected.shape}.")
    return value


def snowball_from_hf_state_dict(
    template: Transformer,
    state_dict: dict[str, jax.Array],
) -> tuple[Transformer, jax.Array]:
    """Load HF tensors into the exact vendored training pytree.

    Returns the imported model and a zero-valued ``pending_qb_betas`` tensor.
    The latter is intentional: the HF router-bias tensors already include the
    pending QB adjustment that existed at export time.
    """
    if template.stacked_blocks is None:
        raise ValueError("Snowball HF import requires use_array_stacked_blocks=True.")

    expected_keys = _expected_hf_keys(template)
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(f"HF tensor schema mismatch: missing={missing}, unexpected={unexpected}")

    imported_blocks = []
    for layer_index, block in enumerate(_unstacked_blocks(template)):
        prefix = f"model.layers.{layer_index}"
        selectors = [
            lambda b: b.rms_attn.weight,
            lambda b: b.attn_gated_norm.w_down,
            lambda b: b.attn_gated_norm.w_up,
            lambda b: b.attn.w_q,
            lambda b: b.attn.w_k,
            lambda b: b.attn.w_v,
            lambda b: b.attn.w_o,
            lambda b: b.attn.attn_gate,
            lambda b: b.rms_mlp.weight,
            lambda b: b.mlp_gated_norm.w_down,
            lambda b: b.mlp_gated_norm.w_up,
            lambda b: b.mlp.router,
            lambda b: b.mlp.router_bias,
            lambda b: b.mlp.expert_mlp.w_gate,
            lambda b: b.mlp.expert_mlp.w_up,
            lambda b: b.mlp.expert_mlp.w_down,
        ]
        values = [
            _checked_tensor(state_dict, f"{prefix}.input_layernorm.weight", block.rms_attn.weight),
            _checked_tensor(
                state_dict,
                f"{prefix}.attn_gated_norm.down_proj.weight",
                block.attn_gated_norm.w_down,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                f"{prefix}.attn_gated_norm.up_proj.weight",
                block.attn_gated_norm.w_up,
                linear=True,
            ),
            _checked_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight", block.attn.w_q, linear=True),
            _checked_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight", block.attn.w_k, linear=True),
            _checked_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight", block.attn.w_v, linear=True),
            _checked_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight", block.attn.w_o, linear=True),
            _checked_tensor(
                state_dict,
                f"{prefix}.self_attn.attn_gate.weight",
                block.attn.attn_gate,
                linear=True,
            ),
            _checked_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight", block.rms_mlp.weight),
            _checked_tensor(
                state_dict,
                f"{prefix}.mlp_gated_norm.down_proj.weight",
                block.mlp_gated_norm.w_down,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                f"{prefix}.mlp_gated_norm.up_proj.weight",
                block.mlp_gated_norm.w_up,
                linear=True,
            ),
            _checked_tensor(state_dict, f"{prefix}.mlp.router.weight", block.mlp.router, linear=True),
            _checked_tensor(state_dict, f"{prefix}.mlp.router.bias", block.mlp.router_bias),
            _checked_tensor(
                state_dict,
                f"{prefix}.mlp.experts.gate_proj.weight",
                block.mlp.expert_mlp.w_gate,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                f"{prefix}.mlp.experts.up_proj.weight",
                block.mlp.expert_mlp.w_up,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                f"{prefix}.mlp.experts.down_proj.weight",
                block.mlp.expert_mlp.w_down,
                linear=True,
            ),
        ]
        if block.shared is not None:
            selectors.extend(
                [
                    lambda b: b.shared.w_gate,
                    lambda b: b.shared.w_up,
                    lambda b: b.shared.w_down,
                ]
            )
            values.extend(
                [
                    _checked_tensor(
                        state_dict,
                        f"{prefix}.shared_expert.gate_proj.weight",
                        block.shared.w_gate,
                        linear=True,
                    ),
                    _checked_tensor(
                        state_dict,
                        f"{prefix}.shared_expert.up_proj.weight",
                        block.shared.w_up,
                        linear=True,
                    ),
                    _checked_tensor(
                        state_dict,
                        f"{prefix}.shared_expert.down_proj.weight",
                        block.shared.w_down,
                        linear=True,
                    ),
                ]
            )
        bound_selectors = tuple(selectors)
        imported_blocks.append(
            eqx.tree_at(
                lambda b, selectors=bound_selectors: tuple(selector(b) for selector in selectors),
                block,
                tuple(values),
            )
        )

    stacked_block = jax.tree.map(lambda *values: jnp.stack(values, axis=0), *imported_blocks)
    model = eqx.tree_at(
        lambda m: (
            m.token_embed,
            m.embed_norm.weight,
            m.embed_gated_norm.w_down,
            m.embed_gated_norm.w_up,
            m.output_proj,
            m.stacked_blocks.stacked,
            m.final_norm.weight,
            m.final_gated_norm.w_down,
            m.final_gated_norm.w_up,
        ),
        template,
        (
            _checked_tensor(state_dict, "model.embed_tokens.weight", template.token_embed),
            _checked_tensor(state_dict, "model.embed_norm.weight", template.embed_norm.weight),
            _checked_tensor(
                state_dict,
                "model.embed_gated_norm.down_proj.weight",
                template.embed_gated_norm.w_down,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                "model.embed_gated_norm.up_proj.weight",
                template.embed_gated_norm.w_up,
                linear=True,
            ),
            _checked_tensor(state_dict, "lm_head.weight", template.output_proj, linear=True),
            stacked_block,
            _checked_tensor(state_dict, "model.norm.weight", template.final_norm.weight),
            _checked_tensor(
                state_dict,
                "model.final_gated_norm.down_proj.weight",
                template.final_gated_norm.w_down,
                linear=True,
            ),
            _checked_tensor(
                state_dict,
                "model.final_gated_norm.up_proj.weight",
                template.final_gated_norm.w_up,
                linear=True,
            ),
        ),
    )
    pending_qb_betas = jnp.zeros((model.config.num_layers, model.config.num_experts), dtype=jnp.float32)
    return model, pending_qb_betas


def _hf_config(model_config: GrugModelConfig) -> HfGrugModelConfig:
    model_dict = dataclasses.asdict(model_config)
    main_fields = {field.name for field in dataclasses.fields(HfGrugModelConfig)}
    return draccus.decode(HfGrugModelConfig, {name: value for name, value in model_dict.items() if name in main_fields})


@dataclass(frozen=True)
class ImportSnowballHfConfig:
    hf_checkpoint: str
    """Pinned HF repo reference or local snapshot path."""

    output_path: str
    """Concrete native checkpoint directory, conventionally ending in ``step-0``."""

    dtype: str = "bfloat16"


def main(config: ImportSnowballHfConfig) -> None:
    start = time.monotonic()
    dtype = getattr(jnp, config.dtype)
    model_config = SNOWBALL_CHAT_MODEL_CONFIG
    if not model_config.use_array_stacked_blocks:
        raise ValueError("The Snowball SFT model config must use ArrayStacked blocks.")

    source = RepoRef.from_string(config.hf_checkpoint)
    converter = _hf_config(model_config).hf_checkpoint_converter(ref_checkpoint=str(source))
    logger.info("Loading public Snowball HF tensors from %s", source)
    with use_cpu_device():
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
        with set_mesh(mesh):
            state_dict = converter.load_state_dict(source, dtype=dtype)
            template = eqx.filter_eval_shape(Transformer.init, model_config, key=jax.random.PRNGKey(0))
            params, pending_qb_betas = snowball_from_hf_state_dict(template, state_dict)
            jax.block_until_ready(params)

            manager = GlobalAsyncCheckpointManager()
            save_checkpoint(
                {"params": params, "pending_qb_betas": pending_qb_betas},
                step=0,
                checkpoint_path=config.output_path,
                manager=manager,
                is_temporary=False,
            )
            manager.wait_until_finished()
    logger.info("Snowball native checkpoint committed to %s in %.1fs", config.output_path, time.monotonic() - start)


if __name__ == "__main__":
    levanter.config.main(main)()
