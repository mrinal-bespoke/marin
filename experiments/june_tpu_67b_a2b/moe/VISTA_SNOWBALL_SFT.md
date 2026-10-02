# Snowball SFT on TACC Vista

This is the portable handoff for the Snowball 67B-A2B SFT work tracked in #9026. The validated Vista geometry is 64 GH200 nodes, one GPU per node, sequence length 32K, and global batch 64.

## What is reproduced

- `chat`: base step 0 → WildChat, one packed epoch (257 steps), Delphi V0 template, LR 5e-5.
- `thinking`: Chat step 257 → Nemotron science reasoning, one packed epoch (630 steps in the measured cache), Delphi V0 template, LR 5e-5.
- `nemotron_terminal`: the historical Vista hybrid, Thinking step 630 → Nemotron Terminal, 1,888 steps, Marin template, LR 5e-5.
- `nemotron_terminal_reference`: the #8225 standalone contract, base step 0 → Nemotron Terminal, 1,888 steps, Marin template, LR 5e-6.

The exact chained third stage in #8225 used a pre-tokenized OpenCode cache that is no longer available. Do not describe the historical Vista hybrid as that exact stage.

## One-time setup

Check out `codex/snowball-tacc-handoff` on Vista and set account-independent paths:

```bash
export SNOWBALL_ROOT="$SCRATCH/snowball-sft"
export MARIN_ROOT="$SCRATCH/marin"
export SNOWBALL_ENV="$SCRATCH/envs/marin-grug-sft-nccl2307"
export SNOWBALL_TOKENIZER=/absolute/path/to/the/pinned/snowball-hf-snapshot
```

Build the environment:

```bash
bash "$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe/bootstrap_vista_snowball_env.sh"
export MARIN_PYTHON="$SNOWBALL_ENV/bin/python"
```

The bootstrap installs and verifies exactly NCCL 2.30.7. Do not use the lockfile's resolved 2.28.9 on Vista/aarch64: it silently wedges multi-host collectives. The guarded runner also verifies NCCL before initializing JAX.

Download these exact dataset revisions and the pinned Snowball model snapshot:

| Stage | Dataset | Revision | Turns column | Files |
| --- | --- | --- | --- | ---: |
| Chat | `nyu-dice-lab/wildchat50m-rewild-sft-385700` | `46a5bb56fffd8c57c3ecc812e647990b1527c001` | `conversation` | 5 |
| Thinking | `laion/llama-nemotron-science-reasoning-on-canonical-think-full` | `bae881d7227146ef6b93fe830a1f613e96ea1338` | `messages` | 15 |
| Terminal | `nvidia/Nemotron-Terminal-Corpus` | `a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9` | `conversations` | 29 |

## Import the base checkpoint

The importer converts the pinned HF weights to the native OCDBT layout and validates representative tensors in a fresh restore:

```bash
export SNOWBALL_HF_CHECKPOINT=/absolute/path/to/the/pinned/snowball-hf-snapshot
export SNOWBALL_INIT="$SNOWBALL_ROOT/experiments/snowball-chat-vista/base-native/checkpoints/step-0"
sbatch --export=ALL "$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe/vista_snowball_import.sbatch"
```

Run the same job again with `SNOWBALL_VALIDATE_ONLY=true` after the import completes.

## Build caches

Create a newline-delimited list of absolute Parquet paths for each pinned selection. Submit `vista_snowball_cache.sbatch` with `SNOWBALL_STAGE`, `SNOWBALL_SOURCE_LIST`, `SNOWBALL_CACHE`, `SNOWBALL_DATASET_ID`, `SNOWBALL_DATASET_REVISION`, and `SNOWBALL_EXPECT_FILES` set to the table above. The job tokenizes, packs, and writes a provenance sidecar; training refuses a cache whose counts, template, tokenizer, revision, or source-file count differs.

Nemotron Terminal has three single-row-group files that exceed PyArrow's nested-array conversion limit. Prepare the 29-file derived view first:

```bash
"$MARIN_PYTHON" -m experiments.june_tpu_67b_a2b.moe.rechunk_nemotron_terminal \
  --source-root /absolute/path/to/nemotron-terminal-snapshot \
  --output-root "$SNOWBALL_ROOT/data/nemotron-terminal-rechunked"
```

Use the generated `parquet-list.txt`. The tool rewrites only `code.parquet`, `math.parquet`, and `swe.parquet`; it computes a streaming logical-row digest before and after and aborts on any content or ordering change.

## Launch training

The supported launcher is `launch_vista_snowball_stage.sh`. It validates inputs, refuses an existing output directory, verifies NCCL, and submits the external-watchdog runner. It never enables the in-process faulthandler timer that killed the first healthy run.

```bash
bash "$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe/launch_vista_snowball_stage.sh" chat
bash "$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe/launch_vista_snowball_stage.sh" thinking
bash "$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe/launch_vista_snowball_stage.sh" nemotron_terminal_reference
```

Set `DRY_RUN=1` to resolve and print a stage without submitting. Set `SNOWBALL_ACCOUNT`, `SNOWBALL_OUTPUT`, `SNOWBALL_CACHE`, or the stage-output variables to override defaults. The failed r7/r8 and sweep launchers remain available in Git history, not in this handoff branch.

The runner disables XLA GPU command buffers, disables NCCL symmetric windows, keeps the JAX compilation cache, and cancels only after two external captures confirm 15 minutes without progress. Completed checkpoints must contain `metadata.json`, `manifest.ocdbt`, numbered manifests, and non-empty `d/` blobs.
