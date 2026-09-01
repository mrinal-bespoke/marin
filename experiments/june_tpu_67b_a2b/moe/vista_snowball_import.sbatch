#!/bin/bash
#SBATCH --partition=gg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --job-name=snowball-base-import-d4

# Four Grace CPU ranks give the importer a data axis of four. Each rank holds
# roughly one quarter of the 134 GB source plus one quarter of the converted
# layout, avoiding the single-node host-memory limit without consuming GH200s.

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter GPU Python executable.}"
: "${SNOWBALL_HF_CHECKPOINT:?Set SNOWBALL_HF_CHECKPOINT to the pinned local HF snapshot.}"
: "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the new native step-0 checkpoint path.}"

module purge
module load gcc/13.2.0

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"

test -x "${MARIN_PYTHON}"
cd "${MARIN_ROOT}"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.import_snowball_hf \
    --hf_checkpoint "${SNOWBALL_HF_CHECKPOINT}" \
    --output_path "${SNOWBALL_INIT}" \
    --distributed true \
    --expert_axis_size 1 \
    --replica_axis_size 1 \
    --model_axis_size 1
