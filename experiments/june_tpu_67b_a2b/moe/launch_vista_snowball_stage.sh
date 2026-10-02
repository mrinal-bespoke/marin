#!/bin/bash
# Portable launcher for the validated Vista Snowball SFT path.
set -euo pipefail

STAGE="${1:-}"
case "${STAGE}" in
  chat|thinking|nemotron_terminal|nemotron_terminal_reference) ;;
  *)
    echo "usage: $0 {chat|thinking|nemotron_terminal|nemotron_terminal_reference}" >&2
    exit 2
    ;;
esac

ROOT="${SNOWBALL_ROOT:-${SCRATCH:?Set SNOWBALL_ROOT or SCRATCH}}"
MARIN_ROOT="${MARIN_ROOT:-${ROOT}/marin}"
MARIN_PYTHON="${MARIN_PYTHON:-${ROOT}/envs/marin-grug-sft-nccl2307/bin/python}"
SNOWBALL_TOKENIZER="${SNOWBALL_TOKENIZER:?Set SNOWBALL_TOKENIZER to the pinned Snowball HF snapshot}"
WORK="${SNOWBALL_WORKDIR:-${ROOT}/experiments/snowball-chat-vista}"
BASE_INIT="${SNOWBALL_BASE_INIT:-${WORK}/base-native/checkpoints/step-0}"
CHAT_OUTPUT="${SNOWBALL_CHAT_OUTPUT:-${WORK}/chat-s1}"
THINKING_OUTPUT="${SNOWBALL_THINKING_OUTPUT:-${WORK}/thinking-s2}"

case "${STAGE}" in
  chat)
    INIT="${BASE_INIT}"
    CACHE="${SNOWBALL_CACHE:-${WORK}/wildchat-cache}"
    OUTPUT="${SNOWBALL_OUTPUT:-${CHAT_OUTPUT}}"
    RUN_ID="${SNOWBALL_RUN_ID:-snowball-chat-s1}"
    STEPS=257
    WALL="${SNOWBALL_WALL:-04:00:00}"
    ;;
  thinking)
    INIT="${SNOWBALL_INIT:-${CHAT_OUTPUT}/checkpoints/step-257}"
    CACHE="${SNOWBALL_CACHE:-${WORK}/thinking-cache}"
    OUTPUT="${SNOWBALL_OUTPUT:-${THINKING_OUTPUT}}"
    RUN_ID="${SNOWBALL_RUN_ID:-snowball-thinking-s2}"
    STEPS="$(cd "${MARIN_ROOT}" && "${MARIN_PYTHON}" - <<PY
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import derive_epoch_steps, read_chat_cache_tokens
print(derive_epoch_steps(read_chat_cache_tokens("${CACHE}")))
PY
)"
    WALL="${SNOWBALL_WALL:-04:00:00}"
    ;;
  nemotron_terminal)
    # Historical Vista hybrid: Thinking -> Nemotron Terminal, Chat LR (5e-5).
    INIT="${SNOWBALL_INIT:-${THINKING_OUTPUT}/checkpoints/step-630}"
    CACHE="${SNOWBALL_CACHE:-${WORK}/nemotron-terminal-cache}"
    OUTPUT="${SNOWBALL_OUTPUT:-${WORK}/nemotron-terminal-s3-hybrid}"
    RUN_ID="${SNOWBALL_RUN_ID:-snowball-nemotron-s3-hybrid}"
    STEPS=1888
    WALL="${SNOWBALL_WALL:-06:00:00}"
    ;;
  nemotron_terminal_reference)
    # Exact #8225 standalone branch: Base -> Nemotron Terminal, 5e-6.
    INIT="${SNOWBALL_INIT:-${BASE_INIT}}"
    CACHE="${SNOWBALL_CACHE:-${WORK}/nemotron-terminal-cache}"
    OUTPUT="${SNOWBALL_OUTPUT:-${WORK}/nemotron-terminal-reference}"
    RUN_ID="${SNOWBALL_RUN_ID:-snowball-nemotron-terminal-reference}"
    STEPS=1888
    WALL="${SNOWBALL_WALL:-06:00:00}"
    ;;
esac

for path in "${MARIN_ROOT}" "${MARIN_PYTHON}" "${INIT}" "${CACHE}" "${SNOWBALL_TOKENIZER}"; do
  [ -e "${path}" ] || { echo "FATAL: required path does not exist: ${path}" >&2; exit 3; }
done
[ ! -e "${OUTPUT}" ] || { echo "FATAL: output already exists: ${OUTPUT}" >&2; exit 3; }

export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat verify-nccl --expected 2.30.7

mkdir -p "${ROOT}/logs"
cat <<PLAN
stage      = ${STAGE}
steps      = ${STEPS}
init       = ${INIT}
cache      = ${CACHE}
tokenizer  = ${SNOWBALL_TOKENIZER}
output     = ${OUTPUT}
python     = ${MARIN_PYTHON}
wall       = ${WALL}
PLAN
[ "${DRY_RUN:-0}" = 1 ] && exit 0

SBATCH_ACCOUNT=()
[ -z "${SNOWBALL_ACCOUNT:-}" ] || SBATCH_ACCOUNT=(-A "${SNOWBALL_ACCOUNT}")
sbatch "${SBATCH_ACCOUNT[@]}" \
  -J "${RUN_ID}" -t "${WALL}" -o "${ROOT}/logs/${RUN_ID}.%j.log" \
  --export=ALL,MARIN_ROOT="${MARIN_ROOT}",MARIN_PYTHON="${MARIN_PYTHON}",SNOWBALL_INIT="${INIT}",SNOWBALL_CACHE="${CACHE}",SNOWBALL_TOKENIZER="${SNOWBALL_TOKENIZER}",SNOWBALL_OUTPUT="${OUTPUT}",SNOWBALL_RUN_ID="${RUN_ID}",SNOWBALL_STEPS="${STEPS}",SNOWBALL_STAGE="${STAGE}",STALL_SECONDS=900 \
  "${MARIN_ROOT}/experiments/june_tpu_67b_a2b/moe/vista_snowball_guarded.sbatch"
