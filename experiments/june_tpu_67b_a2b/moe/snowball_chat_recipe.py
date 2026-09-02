# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball Base-to-Chat model and optimizer contract."""

import dataclasses
import math

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig

SNOWBALL_CHAT_SEQUENCE_LENGTH = 32_768
SNOWBALL_CHAT_BATCH_SIZE = 64
SNOWBALL_CHAT_STEPS = 257
SNOWBALL_CHAT_TOKENS = 538_877_811
SNOWBALL_CHAT_EXAMPLES = 385_700
SNOWBALL_CHAT_DEVICES = 64
SNOWBALL_CHAT_EXPERT_PARALLEL = 8
SNOWBALL_CHAT_REPLICA_AXIS = 1
SNOWBALL_CHAT_MODEL_AXIS = 1
SNOWBALL_CHAT_SEED = 0
SNOWBALL_CHAT_MP = "params=float32,compute=bfloat16,output=bfloat16"
SNOWBALL_NATIVE_PARAMETERS = 67_078_882_816

_QK_MULT = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)
_MODEL_BASE = MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=65_536)

SNOWBALL_CHAT_MODEL_CONFIG = dataclasses.replace(
    _MODEL_BASE,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH,
    attention_implementation="gpu_fa4_cute",
    ce_implementation="batched_xla",
)

SNOWBALL_CHAT_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=5e-5,
    adam_lr=5e-5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)
