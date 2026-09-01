#!/bin/bash
#SBATCH --partition=gh
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --job-name=snowball-chat
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c636-121,c635-101,c641-061,c611-051,c636-152,c608-042,c634-142,c641-012,c634-052,c621-102,c637-042,c638-101,c638-102,c638-112

# Run with SNOWBALL_MODE=distributed-probe and `sbatch --nodes=8` for the
# multi-node fabric gate. The default mode launches a 64-device Chat run.

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter GPU Python executable.}"

SNOWBALL_MODE="${SNOWBALL_MODE:-train}"

module purge
module load gcc/13.2.0 cuda/12.8

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${TACC_CUDA_DIR}"
export TRITON_CC="$(command -v gcc)"
export CC="${TRITON_CC}"
export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"

test -x "${MARIN_PYTHON}"
cd "${MARIN_ROOT}"

if [[ "${SNOWBALL_MODE}" == "distributed-probe" ]]; then
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
    "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat \
      distributed-probe --expected-devices "${SLURM_NTASKS}"
elif [[ "${SNOWBALL_MODE}" == "train" ]]; then
  : "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the native weights-only checkpoint root.}"
  : "${SNOWBALL_CACHE:?Set SNOWBALL_CACHE to the packed WildChat cache.}"
  : "${SNOWBALL_TOKENIZER:?Set SNOWBALL_TOKENIZER to the pinned tokenizer directory.}"
  : "${SNOWBALL_OUTPUT:?Set SNOWBALL_OUTPUT to a unique training output directory.}"
  : "${SNOWBALL_RUN_ID:?Set SNOWBALL_RUN_ID to a unique run name.}"
  : "${SNOWBALL_STEPS:?Set SNOWBALL_STEPS to 2 for smoke or 257 for Chat.}"
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
    "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat train \
      --init-checkpoint-path "${SNOWBALL_INIT}" \
      --data-cache-path "${SNOWBALL_CACHE}" \
      --tokenizer-path "${SNOWBALL_TOKENIZER}" \
      --output-path "${SNOWBALL_OUTPUT}" \
      --run-id "${SNOWBALL_RUN_ID}" \
      --steps "${SNOWBALL_STEPS}" \
      --devices "${SLURM_NTASKS}"
else
  echo "FATAL: SNOWBALL_MODE must be distributed-probe or train, got ${SNOWBALL_MODE}." >&2
  exit 2
fi
