# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and train the Snowball chat stage on TACC Vista.

Each Slurm rank invokes the ``train`` command once. JAX discovers the Slurm
process topology and the vendored Grug loop runs directly on those ranks; no
Fray coordinator is started inside the allocation.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
from datetime import timedelta
from pathlib import Path

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem import prefix_join

from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
    SNOWBALL_CHAT_BATCH_SIZE,
    SNOWBALL_CHAT_DEVICES,
    SNOWBALL_CHAT_EXPERT_PARALLEL,
    SNOWBALL_CHAT_MODEL_AXIS,
    SNOWBALL_CHAT_MODEL_CONFIG,
    SNOWBALL_CHAT_MP,
    SNOWBALL_CHAT_OPTIMIZER,
    SNOWBALL_CHAT_REPLICA_AXIS,
    SNOWBALL_CHAT_SEED,
    SNOWBALL_CHAT_SEQUENCE_LENGTH,
    SNOWBALL_CHAT_STEPS,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugTrainerConfig, run_grug_local
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE


def snowball_chat_format() -> ChatLmDatasetFormat:
    return ChatLmDatasetFormat(
        messages_field="conversation",
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        mask_user_turns=True,
        pack=None,
    )


def prepare_chat_cache(
    *,
    parquet_glob: str,
    cache_path: str,
    tokenizer_path: str,
) -> int:
    """Tokenize and pack the pinned WildChat Parquet shards."""
    from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize  # noqa: PLC0415

    paths = sorted(glob.glob(parquet_glob))
    if not paths:
        raise FileNotFoundError(f"No WildChat Parquet shards matched {parquet_glob!r}.")
    tokenize(
        TokenizeConfig(
            train_paths=paths,
            validation_paths=[],
            cache_path=cache_path,
            tokenizer=tokenizer_path,
            tags=["wildchat_386k", "snowball_chat"],
            format=snowball_chat_format(),
            max_workers=len(paths),
            worker_resources=ResourceConfig(cpu=16, ram="64g", disk="20g"),
        )
    )
    return read_chat_cache_tokens(cache_path)


def read_chat_cache_tokens(cache_path: str) -> int:
    stats_path = Path(cache_path) / "train" / ".stats.json"
    stats = json.loads(stats_path.read_text())
    total_tokens = stats.get("total_tokens")
    if not isinstance(total_tokens, int) or total_tokens <= 0:
        raise ValueError(f"Invalid total_tokens in {stats_path}: {total_tokens!r}")
    return total_tokens


def snowball_chat_data_config(*, cache_path: str, tokenizer_path: str) -> LmDataConfig:
    fmt = snowball_chat_format()
    source = UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_path, format=fmt, tags=["wildchat_386k"])
    return LmDataConfig(
        tokenizer=tokenizer_path,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        enforce_eos=True,
        auto_build_caches=False,
        components={
            "wildchat_386k": DatasetComponent(
                source=source,
                cache_dir=cache_path,
                format=fmt,
                tags=["wildchat_386k"],
                split="train",
            )
        },
        train_weights={"wildchat_386k": 1.0},
        mixture_block_size=2048,
    )


def expected_chat_steps(total_tokens: int) -> int:
    return math.ceil(total_tokens / (SNOWBALL_CHAT_SEQUENCE_LENGTH * SNOWBALL_CHAT_BATCH_SIZE))


def run_distributed_probe(expected_devices: int) -> None:
    """Initialize JAX from Slurm and verify a cross-process collective."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    from levanter.distributed import DistributedConfig  # noqa: PLC0415

    DistributedConfig().initialize()
    if jax.device_count() != expected_devices:
        raise RuntimeError(f"Expected {expected_devices} global devices, found {jax.device_count()}.")
    if jax.local_device_count() != 1:
        raise RuntimeError(f"Expected one local device per Vista rank, found {jax.local_device_count()}.")

    local_value = jnp.asarray([jax.process_index() + 1], dtype=jnp.int32)
    reduced = jax.pmap(lambda value: lax.psum(value, "devices"), axis_name="devices")(local_value)
    expected_sum = expected_devices * (expected_devices + 1) // 2
    actual_sum = int(reduced[0])
    if actual_sum != expected_sum:
        raise RuntimeError(f"Collective returned {actual_sum}, expected {expected_sum}.")
    if jax.process_index() == 0:
        click.echo(f"global_devices={jax.device_count()}")
        click.echo(f"collective_sum={actual_sum}")
        click.echo("SNOWBALL_DISTRIBUTED_PROBE_OK")


def snowball_chat_run_config(
    *,
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
) -> GrugRunConfig:
    if devices != SNOWBALL_CHAT_DEVICES:
        raise ValueError(f"Snowball Chat requires {SNOWBALL_CHAT_DEVICES} devices, got {devices}.")
    total_tokens = read_chat_cache_tokens(data_cache_path)
    full_epoch_steps = expected_chat_steps(total_tokens)
    if steps > full_epoch_steps:
        raise ValueError(f"Requested {steps} steps, but the packed WildChat epoch has only {full_epoch_steps} steps.")

    run_resources = ResourceConfig.with_gpu("GH200", count=1, replicas=devices)
    trainer = TrainerConfig(
        id=run_id,
        seed=SNOWBALL_CHAT_SEED,
        train_batch_size=SNOWBALL_CHAT_BATCH_SIZE,
        per_device_parallelism=-1,
        num_train_steps=steps,
        mp=jmp.get_policy(SNOWBALL_CHAT_MP),
        tracker=WandbConfig(
            project="marin_moe_sft",
            name=run_id,
            group="grug-67b-a2b-sft",
            tags=["moe", "67b_a2b", "sft", "s1_chat", "seq32768", "vista-gh200"],
            mode="offline",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": SNOWBALL_CHAT_EXPERT_PARALLEL}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(output_path, "checkpoints"),
            temporary_base_path=prefix_join(output_path, "checkpoints-tmp"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": 1000}],
            timeout=timedelta(hours=2),
        ),
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=latest_checkpoint_path(init_checkpoint_path),
    )
    return GrugRunConfig(
        model=dataclasses.replace(SNOWBALL_CHAT_MODEL_CONFIG, max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH),
        data=snowball_chat_data_config(cache_path=data_cache_path, tokenizer_path=tokenizer_path),
        resources=run_resources,
        optimizer=SNOWBALL_CHAT_OPTIMIZER,
        trainer=GrugTrainerConfig(
            trainer=trainer,
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            replica_axis_size=SNOWBALL_CHAT_REPLICA_AXIS,
            model_axis_size=SNOWBALL_CHAT_MODEL_AXIS,
            expert_axis_size=SNOWBALL_CHAT_EXPERT_PARALLEL,
            sft_weights_only_init=True,
        ),
        eval=None,
    )


@click.group()
def main() -> None:
    pass


@main.command("prepare-data")
@click.option("--parquet-glob", required=True)
@click.option("--cache-path", required=True)
@click.option("--tokenizer-path", required=True)
def prepare_data_command(parquet_glob: str, cache_path: str, tokenizer_path: str) -> None:
    total_tokens = prepare_chat_cache(
        parquet_glob=parquet_glob,
        cache_path=cache_path,
        tokenizer_path=tokenizer_path,
    )
    click.echo(f"total_tokens={total_tokens}")
    click.echo(f"full_epoch_steps={expected_chat_steps(total_tokens)}")
    click.echo("SNOWBALL_CHAT_CACHE_OK")


@main.command("distributed-probe")
@click.option("--expected-devices", type=click.IntRange(min=2), required=True)
def distributed_probe_command(expected_devices: int) -> None:
    run_distributed_probe(expected_devices)


@main.command("train")
@click.option("--init-checkpoint-path", required=True)
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=SNOWBALL_CHAT_STEPS, show_default=True)
@click.option("--devices", type=click.IntRange(min=1), default=SNOWBALL_CHAT_DEVICES, show_default=True)
def train_command(
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
) -> None:
    run_config = snowball_chat_run_config(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
    )
    run_grug_local(run_config)


if __name__ == "__main__":
    main()
