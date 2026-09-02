# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import pytest

from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _model as ORIGINAL_MODEL_CONFIG
from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _optimizer as ORIGINAL_OPTIMIZER
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    SNOWBALL_CHAT_BATCH_SIZE,
    SNOWBALL_CHAT_DEVICES,
    SNOWBALL_CHAT_EXAMPLES,
    SNOWBALL_CHAT_EXPERT_PARALLEL,
    SNOWBALL_CHAT_MODEL_AXIS,
    SNOWBALL_CHAT_MODEL_CONFIG,
    SNOWBALL_CHAT_OPTIMIZER,
    SNOWBALL_CHAT_REPLICA_AXIS,
    SNOWBALL_CHAT_SEQUENCE_LENGTH,
    SNOWBALL_CHAT_STEPS,
    SNOWBALL_CHAT_TOKENS,
    SNOWBALL_NATIVE_PARAMETERS,
    expected_chat_steps,
    snowball_chat_format,
    validate_chat_cache_layout,
    validate_chat_epoch,
    validate_native_checkpoint_layout,
    vista_trainer_mesh_config,
)
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE


def test_vista_recipe_matches_snowball_chat_contract() -> None:
    assert SNOWBALL_CHAT_MODEL_CONFIG == ORIGINAL_MODEL_CONFIG
    assert SNOWBALL_CHAT_OPTIMIZER == ORIGINAL_OPTIMIZER
    assert SNOWBALL_CHAT_DEVICES == 64
    assert SNOWBALL_CHAT_SEQUENCE_LENGTH == 32_768
    assert SNOWBALL_CHAT_BATCH_SIZE == 64
    assert SNOWBALL_CHAT_STEPS == 257
    assert SNOWBALL_CHAT_EXPERT_PARALLEL == 8
    assert SNOWBALL_CHAT_REPLICA_AXIS == 1
    assert SNOWBALL_CHAT_MODEL_AXIS == 1
    assert SNOWBALL_CHAT_MODEL_CONFIG.use_array_stacked_blocks
    assert SNOWBALL_CHAT_MODEL_CONFIG.attention_implementation == "gpu_fa4_cute"
    assert SNOWBALL_CHAT_MODEL_CONFIG.ce_implementation == "batched_xla"
    assert SNOWBALL_CHAT_OPTIMIZER.learning_rate == 5e-5
    assert SNOWBALL_CHAT_OPTIMIZER.adam_lr == 5e-5
    assert SNOWBALL_CHAT_OPTIMIZER.beta1 == 0.9
    assert SNOWBALL_CHAT_OPTIMIZER.beta2 == 0.95
    assert SNOWBALL_CHAT_OPTIMIZER.weight_decay == 0.0
    assert SNOWBALL_CHAT_OPTIMIZER.warmup == 0.03

    fmt = snowball_chat_format()
    assert fmt.messages_field == "conversation"
    assert fmt.chat_template == DELPHI_V0_CHAT_TEMPLATE
    assert fmt.mask_user_turns
    assert fmt.pack is None


def test_epoch_step_count_uses_packed_token_count() -> None:
    total_tokens = 538_900_000
    assert expected_chat_steps(total_tokens) == math.ceil(
        total_tokens / (SNOWBALL_CHAT_SEQUENCE_LENGTH * SNOWBALL_CHAT_BATCH_SIZE)
    )
    assert expected_chat_steps(total_tokens) == 257
    assert validate_chat_epoch(total_tokens) == 257


def test_epoch_gate_rejects_cache_drift() -> None:
    with pytest.raises(ValueError, match="requires 257"):
        validate_chat_epoch(536_000_000)


def test_vista_trainer_mesh_accepts_one_device_per_node() -> None:
    mesh = vista_trainer_mesh_config()

    ici_axes, dcn_axes = mesh.axis_shapes(num_devices=64, num_slices=64)

    assert ici_axes == {"data": 1, "replica": 1, "model": 1, "expert": 1}
    assert dcn_axes == {"replica_dcn": 64}


def test_native_checkpoint_layout_requires_discoverable_step(tmp_path: Path) -> None:
    experiment_path = tmp_path / "base-native-bufferfix"
    checkpoint_path = experiment_path / "checkpoints" / "step-0"
    payload_path = checkpoint_path / "d"
    payload_path.mkdir(parents=True)
    (checkpoint_path / "metadata.json").write_text('{"step": 0, "timestamp": "2026-09-01T20:09:27"}')
    (checkpoint_path / "manifest.ocdbt").write_bytes(b"manifest")
    (checkpoint_path / "manifest.0000000000000001").write_bytes(b"manifest")
    with (payload_path / "payload").open("wb") as file:
        file.truncate(SNOWBALL_NATIVE_PARAMETERS * 2)

    with pytest.raises(FileNotFoundError, match="Could not discover checkpoint"):
        validate_native_checkpoint_layout(str(experiment_path))

    assert validate_native_checkpoint_layout(str(checkpoint_path)) == (
        str(checkpoint_path),
        1,
        SNOWBALL_NATIVE_PARAMETERS * 2,
    )


def test_chat_cache_layout_requires_exact_completed_dataset(tmp_path: Path) -> None:
    train_path = tmp_path / "cache" / "train"
    shard_name = "part-00000-of-00001"
    shard_path = train_path / shard_name
    for field in ("assistant_masks", "input_ids"):
        payload_path = shard_path / field / "data"
        payload_path.mkdir(parents=True)
        (payload_path / "chunk").write_bytes(b"payload")
    (shard_path / ".success").touch()
    (train_path / ".stats.json").write_text(
        json.dumps({"total_tokens": SNOWBALL_CHAT_TOKENS, "total_elements": SNOWBALL_CHAT_EXAMPLES})
    )
    (train_path / "shard_ledger.json").write_text(
        json.dumps(
            {
                "is_finished": True,
                "shard_rows": {shard_name: SNOWBALL_CHAT_EXAMPLES},
                "finished_shards": [shard_name],
                "field_counts": {
                    "assistant_masks": SNOWBALL_CHAT_TOKENS,
                    "input_ids": SNOWBALL_CHAT_TOKENS,
                },
            }
        )
    )

    assert validate_chat_cache_layout(str(tmp_path / "cache")) == (
        SNOWBALL_CHAT_TOKENS,
        SNOWBALL_CHAT_EXAMPLES,
        1,
    )

    (train_path / ".stats.json").write_text(
        json.dumps({"total_tokens": SNOWBALL_CHAT_TOKENS - 1, "total_elements": SNOWBALL_CHAT_EXAMPLES})
    )
    with pytest.raises(ValueError, match="expected 538877811 tokens"):
        validate_chat_cache_layout(str(tmp_path / "cache"))
