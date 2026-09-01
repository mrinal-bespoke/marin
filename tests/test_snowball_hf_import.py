# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.import_snowball_hf import (
    ImportSnowballHfConfig,
    snowball_from_hf_state_dict,
    snowball_hf_state_dict,
)
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def test_importer_cli_loads() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.june_tpu_67b_a2b.moe.import_snowball_hf", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "hf_checkpoint" in result.stdout
    assert "output_path" in result.stdout
    assert "distributed" in result.stdout
    assert "expert_axis_size" in result.stdout


def test_importer_cli_parses_distributed_geometry() -> None:
    config = draccus.parse(
        ImportSnowballHfConfig,
        args=[
            "--hf_checkpoint",
            "/input",
            "--output_path",
            "/output",
            "--distributed",
            "true",
            "--expert_axis_size",
            "8",
        ],
    )
    assert config.distributed
    assert config.expert_axis_size == 8


def _tiny_model() -> Transformer:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )
    return Transformer.init(config, key=jax.random.PRNGKey(0))


def test_hf_round_trip_restores_vendored_layout_without_pending_router_bias() -> None:
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        source = _tiny_model()
        hf_state = snowball_hf_state_dict(source)
        template = eqx.filter_eval_shape(Transformer.init, source.config, key=jax.random.PRNGKey(1))

        imported, pending_qb_betas = snowball_from_hf_state_dict(template, hf_state)

        imported_hf_state = snowball_hf_state_dict(imported)
        assert imported_hf_state.keys() == hf_state.keys()
        for name in hf_state:
            assert jnp.array_equal(imported_hf_state[name], hf_state[name]), name
        assert jnp.array_equal(
            pending_qb_betas,
            jnp.zeros((source.config.num_layers, source.config.num_experts), dtype=jnp.float32),
        )


def test_hf_import_rejects_missing_or_unexpected_tensors() -> None:
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        source = _tiny_model()
        hf_state = snowball_hf_state_dict(source)
        template = eqx.filter_eval_shape(Transformer.init, source.config, key=jax.random.PRNGKey(1))

        missing_state = dict(hf_state)
        missing_state.pop("model.embed_tokens.weight")
        with pytest.raises(ValueError, match=r"missing=.*model\.embed_tokens\.weight"):
            snowball_from_hf_state_dict(template, missing_state)

        unexpected_state = dict(hf_state)
        unexpected_state["unexpected.weight"] = jnp.zeros((1,))
        with pytest.raises(ValueError, match=r"unexpected=.*unexpected\.weight"):
            snowball_from_hf_state_dict(template, unexpected_state)
