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
import os
import shutil
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
from transformers import AutoTokenizer

from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
    SNOWBALL_CHAT_BATCH_SIZE,
    SNOWBALL_CHAT_DEVICES,
    SNOWBALL_CHAT_EXAMPLES,
    SNOWBALL_CHAT_EXPERT_PARALLEL,
    SNOWBALL_CHAT_MODEL_AXIS,
    SNOWBALL_CHAT_MODEL_CONFIG,
    SNOWBALL_CHAT_MP,
    SNOWBALL_CHAT_OPTIMIZER,
    SNOWBALL_CHAT_REPLICA_AXIS,
    SNOWBALL_CHAT_SEED,
    SNOWBALL_CHAT_SEQUENCE_LENGTH,
    SNOWBALL_CHAT_STEPS,
    SNOWBALL_CHAT_TOKENS,
    SNOWBALL_NATIVE_PARAMETERS,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugTrainerConfig, run_grug_local
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE

_MINIMUM_OUTPUT_FREE_BYTES = 1_000_000_000_000
_MINIMUM_OUTPUT_FREE_INODES = 10_000


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


@dataclasses.dataclass(frozen=True)
class SnowballChatPreflight:
    """Static artifacts and geometry accepted by the Vista launch gate."""

    init_checkpoint_path: str
    init_payload_files: int
    init_payload_bytes: int
    cache_tokens: int
    cache_examples: int
    cache_shards: int
    tokenizer_vocab_size: int
    data_axis_size: int
    per_device_batch_size: int
    output_free_bytes: int
    output_free_inodes: int
    output_checkpoint_path: str | None


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_native_checkpoint_layout(init_checkpoint_path: str) -> tuple[str, int, int]:
    """Resolve a complete Snowball step-0 checkpoint and measure its payload."""
    if not Path(init_checkpoint_path).is_absolute():
        raise ValueError(f"Snowball init checkpoint path must be absolute: {init_checkpoint_path!r}.")

    resolved_path = latest_checkpoint_path(init_checkpoint_path)
    checkpoint_path = Path(resolved_path)
    metadata = _json_object(checkpoint_path / "metadata.json")
    if metadata.get("step") != 0:
        raise ValueError(f"Snowball base checkpoint must be step 0, got {metadata.get('step')!r} in {resolved_path}.")

    manifest_path = checkpoint_path / "manifest.ocdbt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snowball base checkpoint is missing {manifest_path}.")
    numbered_manifests = [
        path
        for path in checkpoint_path.glob("manifest.*")
        if path.name not in {"manifest.json", "manifest.ocdbt"} and path.is_file()
    ]
    if not numbered_manifests:
        raise FileNotFoundError(f"Snowball base checkpoint has no numbered OCDBT manifests in {resolved_path}.")

    payload_path = checkpoint_path / "d"
    payload_files = [path for path in payload_path.rglob("*") if path.is_file()]
    payload_bytes = sum(path.stat().st_size for path in payload_files)
    minimum_payload_bytes = SNOWBALL_NATIVE_PARAMETERS * 2
    if not payload_files or payload_bytes < minimum_payload_bytes:
        raise ValueError(
            f"Snowball base checkpoint payload is incomplete: {len(payload_files)} files and {payload_bytes} bytes; "
            f"expected at least {minimum_payload_bytes} bytes."
        )
    return resolved_path, len(payload_files), payload_bytes


def validate_chat_cache_layout(data_cache_path: str) -> tuple[int, int, int]:
    """Validate the exact completed WildChat cache used by the Chat recipe."""
    cache_path = Path(data_cache_path)
    if not cache_path.is_absolute():
        raise ValueError(f"Snowball data cache path must be absolute: {data_cache_path!r}.")

    train_path = cache_path / "train"
    stats = _json_object(train_path / ".stats.json")
    total_tokens = stats.get("total_tokens")
    total_examples = stats.get("total_elements")
    if total_tokens != SNOWBALL_CHAT_TOKENS or total_examples != SNOWBALL_CHAT_EXAMPLES:
        raise ValueError(
            f"Packed WildChat cache has {total_tokens!r} tokens and {total_examples!r} examples; expected "
            f"{SNOWBALL_CHAT_TOKENS} tokens and {SNOWBALL_CHAT_EXAMPLES} examples."
        )

    ledger = _json_object(train_path / "shard_ledger.json")
    shard_rows = ledger.get("shard_rows")
    finished_shards = ledger.get("finished_shards")
    field_counts = ledger.get("field_counts")
    if not isinstance(shard_rows, dict) or not isinstance(finished_shards, list):
        raise ValueError(f"Packed WildChat cache has a malformed shard ledger at {train_path}.")
    if not all(isinstance(shard, str) for shard in finished_shards) or not all(
        isinstance(shard, str) and isinstance(rows, int) for shard, rows in shard_rows.items()
    ):
        raise ValueError(f"Packed WildChat cache has malformed shard names or row counts at {train_path}.")
    typed_finished_shards = [str(shard) for shard in finished_shards]
    typed_shard_rows = {str(shard): int(rows) for shard, rows in shard_rows.items()}
    if ledger.get("is_finished") is not True or set(typed_finished_shards) != set(typed_shard_rows):
        raise ValueError(f"Packed WildChat cache is not fully committed at {train_path}.")
    if sum(typed_shard_rows.values()) != SNOWBALL_CHAT_EXAMPLES:
        raise ValueError(f"Packed WildChat cache shard rows do not sum to {SNOWBALL_CHAT_EXAMPLES}.")
    if field_counts != {"assistant_masks": SNOWBALL_CHAT_TOKENS, "input_ids": SNOWBALL_CHAT_TOKENS}:
        raise ValueError(f"Packed WildChat cache field counts are invalid: {field_counts!r}.")

    for shard in typed_finished_shards:
        shard_path = train_path / shard
        if not (shard_path / ".success").is_file():
            raise FileNotFoundError(f"Packed WildChat cache shard is missing its success marker: {shard_path}.")
        for field in ("assistant_masks", "input_ids"):
            field_path = shard_path / field
            has_payload = field_path.is_dir() and any(
                path.is_file() and path.stat().st_size > 0 for path in field_path.rglob("*")
            )
            if not has_payload:
                raise FileNotFoundError(f"Packed WildChat cache shard has no {field} payload: {shard_path}.")

    return SNOWBALL_CHAT_TOKENS, SNOWBALL_CHAT_EXAMPLES, len(typed_finished_shards)


def _validate_tokenizer(tokenizer_path: str) -> int:
    path = Path(tokenizer_path)
    if not path.is_absolute():
        raise ValueError(f"Snowball tokenizer path must be absolute: {tokenizer_path!r}.")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Snowball tokenizer is missing {path / filename}.")

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    vocab_size = len(tokenizer)
    if vocab_size != SNOWBALL_CHAT_MODEL_CONFIG.vocab_size:
        raise ValueError(
            f"Snowball tokenizer has {vocab_size} tokens, but the model expects {SNOWBALL_CHAT_MODEL_CONFIG.vocab_size}."
        )
    token_ids = tokenizer.encode("Snowball launch preflight", add_special_tokens=False)
    if not token_ids or max(token_ids) >= vocab_size:
        raise ValueError(f"Snowball tokenizer produced invalid token IDs: {token_ids!r}.")
    return vocab_size


def _validate_output_path(
    output_path: str,
    *,
    input_paths: tuple[str, ...],
    expected_output_checkpoint_step: int | None,
) -> tuple[str | None, int, int]:
    path = Path(output_path)
    if not path.is_absolute():
        raise ValueError(f"Snowball output path must be absolute: {output_path!r}.")
    resolved_output = path.resolve(strict=False)
    for input_path in input_paths:
        resolved_input = Path(input_path).resolve(strict=True)
        if (
            resolved_output == resolved_input
            or resolved_output in resolved_input.parents
            or resolved_input in resolved_output.parents
        ):
            raise ValueError(f"Snowball output path {resolved_output} overlaps input path {resolved_input}.")

    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"Snowball output parent is not writable: {path.parent}.")
    free_bytes = shutil.disk_usage(path.parent).free
    stat = os.statvfs(path.parent)
    free_inodes = stat.f_favail
    if free_bytes < _MINIMUM_OUTPUT_FREE_BYTES:
        raise OSError(
            f"Snowball output filesystem has only {free_bytes} free bytes; "
            f"at least {_MINIMUM_OUTPUT_FREE_BYTES} are required."
        )
    if free_inodes < _MINIMUM_OUTPUT_FREE_INODES:
        raise OSError(
            f"Snowball output filesystem has only {free_inodes} free inodes; "
            f"at least {_MINIMUM_OUTPUT_FREE_INODES} are required."
        )

    if expected_output_checkpoint_step is None:
        if path.exists():
            raise FileExistsError(f"Fresh Snowball output path already exists: {path}.")
        return None, free_bytes, free_inodes

    output_checkpoint = latest_checkpoint_path(path / "checkpoints")
    output_checkpoint_path = Path(output_checkpoint)
    output_metadata = _json_object(output_checkpoint_path / "metadata.json")
    if output_metadata.get("step") != expected_output_checkpoint_step:
        raise ValueError(
            f"Snowball resume expected output checkpoint step {expected_output_checkpoint_step}, got "
            f"{output_metadata.get('step')!r} at {output_checkpoint}."
        )
    if not (output_checkpoint_path / "manifest.ocdbt").is_file():
        raise FileNotFoundError(f"Snowball resume checkpoint is missing manifest.ocdbt: {output_checkpoint}.")
    output_payload_files = [path for path in (output_checkpoint_path / "d").rglob("*") if path.is_file()]
    if not output_payload_files or sum(path.stat().st_size for path in output_payload_files) == 0:
        raise ValueError(f"Snowball resume checkpoint has no payload data: {output_checkpoint}.")
    return output_checkpoint, free_bytes, free_inodes


def preflight_snowball_chat(
    *,
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None = None,
) -> SnowballChatPreflight:
    """Validate every static input before requesting a 64-node training allocation."""
    if not run_id or "/" in run_id:
        raise ValueError(f"Snowball run ID must be a non-empty path-free name, got {run_id!r}.")
    if steps < 1 or steps > SNOWBALL_CHAT_STEPS:
        raise ValueError(f"Snowball steps must be between 1 and {SNOWBALL_CHAT_STEPS}, got {steps}.")
    if expected_output_checkpoint_step is not None and steps <= expected_output_checkpoint_step:
        raise ValueError(
            f"Snowball resume target {steps} must exceed checkpoint step {expected_output_checkpoint_step}."
        )

    resolved_init, payload_files, payload_bytes = validate_native_checkpoint_layout(init_checkpoint_path)
    cache_tokens, cache_examples, cache_shards = validate_chat_cache_layout(data_cache_path)
    tokenizer_vocab_size = _validate_tokenizer(tokenizer_path)

    mesh_factor = SNOWBALL_CHAT_REPLICA_AXIS * SNOWBALL_CHAT_EXPERT_PARALLEL * SNOWBALL_CHAT_MODEL_AXIS
    if devices % mesh_factor != 0:
        raise ValueError(f"Snowball device count {devices} is not divisible by mesh factor {mesh_factor}.")
    data_axis_size = devices // mesh_factor
    batch_shards = data_axis_size * SNOWBALL_CHAT_EXPERT_PARALLEL
    if SNOWBALL_CHAT_BATCH_SIZE % batch_shards != 0:
        raise ValueError(f"Snowball batch size {SNOWBALL_CHAT_BATCH_SIZE} is not divisible by {batch_shards} shards.")
    ici_axes, dcn_axes = vista_trainer_mesh_config().axis_shapes(num_devices=devices, num_slices=devices)
    if ici_axes != {"data": 1, "replica": 1, "model": 1, "expert": 1} or dcn_axes != {"replica_dcn": devices}:
        raise ValueError(f"Snowball Vista trainer mesh is invalid: ICI={ici_axes}, DCN={dcn_axes}.")

    output_checkpoint, output_free_bytes, output_free_inodes = _validate_output_path(
        output_path,
        input_paths=(resolved_init, data_cache_path, tokenizer_path),
        expected_output_checkpoint_step=expected_output_checkpoint_step,
    )
    run_config = snowball_chat_run_config(
        init_checkpoint_path=resolved_init,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
    )
    if run_config.trainer.trainer.initialize_from != resolved_init:
        raise ValueError(
            f"Snowball config resolved init checkpoint to {run_config.trainer.trainer.initialize_from}, "
            f"expected {resolved_init}."
        )

    return SnowballChatPreflight(
        init_checkpoint_path=resolved_init,
        init_payload_files=payload_files,
        init_payload_bytes=payload_bytes,
        cache_tokens=cache_tokens,
        cache_examples=cache_examples,
        cache_shards=cache_shards,
        tokenizer_vocab_size=tokenizer_vocab_size,
        data_axis_size=data_axis_size,
        per_device_batch_size=SNOWBALL_CHAT_BATCH_SIZE // batch_shards,
        output_free_bytes=output_free_bytes,
        output_free_inodes=output_free_inodes,
        output_checkpoint_path=output_checkpoint,
    )


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


def validate_chat_epoch(total_tokens: int) -> int:
    steps = expected_chat_steps(total_tokens)
    if steps != SNOWBALL_CHAT_STEPS:
        raise ValueError(
            f"Packed WildChat cache resolves to {steps} steps, but the pinned Snowball Chat contract requires "
            f"{SNOWBALL_CHAT_STEPS}."
        )
    return steps


def vista_trainer_mesh_config() -> MeshConfig:
    """Return Trainer bookkeeping for Vista's one-GPU-per-node topology.

    Grug constructs its compute mesh separately as ``(replica_dcn, data, expert,
    model) = (1, 8, 8, 1)``.  TrainerConfig only needs a compatible 64-way batch
    mesh to derive per-device parallelism before the Grug mesh exists.  On Vista,
    every Slurm rank is a one-device JAX slice, so the local ICI expert axis must
    remain size one and the 64 ranks are represented by ``replica_dcn``.
    """
    return MeshConfig(
        axes={"expert": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
    )


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


def run_gpu_kernel_probe() -> None:
    """Validate training backends and execute the production attention kernels on one GH200."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention  # noqa: PLC0415

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Snowball GPU kernel probe requires a GPU backend, got {jax.default_backend()!r}.")
    cpu_devices = jax.local_devices(backend="cpu")
    if len(cpu_devices) != 1:
        raise RuntimeError(f"Snowball data loading requires one local CPU device, found {len(cpu_devices)}.")

    key = jax.random.PRNGKey(SNOWBALL_CHAT_SEED)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (1, 128, 20, 128), dtype=jnp.bfloat16) * 0.1
    k = jax.random.normal(k_key, (1, 128, 5, 128), dtype=jnp.bfloat16) * 0.1
    v = jax.random.normal(v_key, (1, 128, 5, 128), dtype=jnp.bfloat16) * 0.1
    mask = AttentionMask.causal(sliding_window=31)

    @jax.jit
    def loss_and_output(q_value, k_value, v_value):
        output = gpu_fa4_cute_attention(q_value, k_value, v_value, mask)
        return jnp.sum(output.astype(jnp.float32)), output

    (_, output), gradients = jax.value_and_grad(loss_and_output, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    jax.block_until_ready((output, gradients))
    arrays = (output, *gradients)
    if not all(bool(jax.device_get(jnp.all(jnp.isfinite(array)))) for array in arrays):
        raise ValueError("Snowball FA4 forward or backward produced non-finite values.")

    click.echo(f"backend={jax.default_backend()}")
    click.echo(f"device={jax.devices()[0]}")
    click.echo(f"cpu_device={cpu_devices[0]}")
    click.echo("SNOWBALL_GPU_KERNEL_PROBE_OK")


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
    full_epoch_steps = validate_chat_epoch(total_tokens)
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
        mesh=vista_trainer_mesh_config(),
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
    click.echo(f"full_epoch_steps={validate_chat_epoch(total_tokens)}")
    click.echo("SNOWBALL_CHAT_CACHE_OK")


@main.command("distributed-probe")
@click.option("--expected-devices", type=click.IntRange(min=2), required=True)
def distributed_probe_command(expected_devices: int) -> None:
    run_distributed_probe(expected_devices)


@main.command("gpu-kernel-probe")
def gpu_kernel_probe_command() -> None:
    run_gpu_kernel_probe()


@main.command("preflight")
@click.option("--init-checkpoint-path", required=True)
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=SNOWBALL_CHAT_STEPS, show_default=True)
@click.option("--devices", type=click.IntRange(min=1), default=SNOWBALL_CHAT_DEVICES, show_default=True)
@click.option("--expected-output-checkpoint-step", type=click.IntRange(min=0))
def preflight_command(
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None,
) -> None:
    report = preflight_snowball_chat(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        expected_output_checkpoint_step=expected_output_checkpoint_step,
    )
    click.echo(json.dumps(dataclasses.asdict(report), sort_keys=True))
    click.echo("SNOWBALL_CHAT_PREFLIGHT_OK")


@main.command("train")
@click.option("--init-checkpoint-path", required=True)
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=SNOWBALL_CHAT_STEPS, show_default=True)
@click.option("--devices", type=click.IntRange(min=1), default=SNOWBALL_CHAT_DEVICES, show_default=True)
@click.option("--expected-output-checkpoint-step", type=click.IntRange(min=0))
def train_command(
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None,
) -> None:
    preflight_snowball_chat(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        expected_output_checkpoint_step=expected_output_checkpoint_step,
    )
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
