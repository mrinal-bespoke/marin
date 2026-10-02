#!/bin/bash
# Build and provenance-check one packed Snowball cache on a Vista CPU node.
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=06:00:00
#SBATCH --job-name=snowball-cache

set -euo pipefail
: "${MARIN_ROOT:?}"; : "${MARIN_PYTHON:?}"; : "${SNOWBALL_STAGE:?}"
: "${SNOWBALL_SOURCE_LIST:?}"; : "${SNOWBALL_CACHE:?}"; : "${SNOWBALL_TOKENIZER:?}"
: "${SNOWBALL_DATASET_ID:?}"; : "${SNOWBALL_DATASET_REVISION:?}"; : "${SNOWBALL_EXPECT_FILES:?}"

module purge
module load gcc/13.2.0
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
cd "${MARIN_ROOT}"

"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat prepare-data \
  --parquet-list "${SNOWBALL_SOURCE_LIST}" --expect-files "${SNOWBALL_EXPECT_FILES}" \
  --cache-path "${SNOWBALL_CACHE}" --tokenizer-path "${SNOWBALL_TOKENIZER}" \
  --stage "${SNOWBALL_STAGE}"

"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat write-provenance \
  --cache-path "${SNOWBALL_CACHE}" --stage "${SNOWBALL_STAGE}" \
  --dataset-id "${SNOWBALL_DATASET_ID}" --dataset-revision "${SNOWBALL_DATASET_REVISION}" \
  --parquet-list "${SNOWBALL_SOURCE_LIST}" --expect-files "${SNOWBALL_EXPECT_FILES}" \
  --tokenizer-path "${SNOWBALL_TOKENIZER}"

echo "SNOWBALL_CACHE_JOB_OK stage=${SNOWBALL_STAGE}"
