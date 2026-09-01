# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _model as ORIGINAL_MODEL_CONFIG
from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _optimizer as ORIGINAL_OPTIMIZER
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    SNOWBALL_CHAT_BATCH_SIZE,
    SNOWBALL_CHAT_DEVICES,
    SNOWBALL_CHAT_EXPERT_PARALLEL,
    SNOWBALL_CHAT_MODEL_AXIS,
    SNOWBALL_CHAT_MODEL_CONFIG,
    SNOWBALL_CHAT_OPTIMIZER,
    SNOWBALL_CHAT_REPLICA_AXIS,
    SNOWBALL_CHAT_SEQUENCE_LENGTH,
    SNOWBALL_CHAT_STEPS,
    expected_chat_steps,
    snowball_chat_format,
    validate_chat_epoch,
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
