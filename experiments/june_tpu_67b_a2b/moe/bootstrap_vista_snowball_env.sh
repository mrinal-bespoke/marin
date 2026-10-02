#!/bin/bash
# Build the disposable Vista environment used by the successful Snowball runs.
set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout on Vista}"
SNOWBALL_ENV="${SNOWBALL_ENV:-${SCRATCH:?}/envs/marin-grug-sft-nccl2307}"

module purge
module load gcc/13.2.0 cuda/12.8
cd "${MARIN_ROOT}"

UV_PROJECT_ENVIRONMENT="${SNOWBALL_ENV}" uv sync --all-packages --extra=gpu --frozen
uv pip install --python "${SNOWBALL_ENV}/bin/python" --no-deps nvidia-nccl-cu13==2.30.7

export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
NCCL_LIBRARY=$("${SNOWBALL_ENV}/bin/python" -c 'import glob, os, sys; print(sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))[0])')
export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
"${SNOWBALL_ENV}/bin/python" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat verify-nccl \
  --expected 2.30.7

echo "SNOWBALL_VISTA_ENV_OK ${SNOWBALL_ENV}"
