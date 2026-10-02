#!/bin/bash
# Guarded full-epoch Snowball SFT run with #7344 forensics armed up front.
# Stage-agnostic: SNOWBALL_STEPS selects the epoch length (Chat 257; Thinking is
# DERIVED from its completed cache, never hardcoded).
#
# NOT a six-step test. Issue #7344 reports the hazard tracks WALL CLOCK, so a low
# step budget can finish before encountering it. The run always continues to the
# requested stage budget in the same allocation.
#
# NCCL 2.30.7 completed both the 257-step Chat run and the 630-step Thinking run
# on Vista. The runner pins that build and records the mapped library for every
# new run rather than assuming the environment resolved it correctly.
#SBATCH --partition=gh
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --job-name=snowball-guarded

set -uo pipefail
: "${MARIN_ROOT:?}"; : "${MARIN_PYTHON:?}"
: "${SNOWBALL_INIT:?}"; : "${SNOWBALL_CACHE:?}"; : "${SNOWBALL_TOKENIZER:?}"
: "${SNOWBALL_OUTPUT:?}"; : "${SNOWBALL_RUN_ID:?}"
STALL_SECONDS="${STALL_SECONDS:-900}"
# Epoch length. Defaults to the pinned Chat contract so existing behaviour is
# unchanged; Thinking passes the value derived from its cache.
SNOWBALL_STEPS="${SNOWBALL_STEPS:-257}"
# Stage selects the turns field, dataset component, W&B tag and epoch ceiling.
# Without it BOTH the preflight and train commands silently default to "chat"
# and demand the 257-step Chat contract.
SNOWBALL_STAGE="${SNOWBALL_STAGE:-chat}"

module purge; module load gcc/13.2.0 cuda/12.8
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda,cpu
export HF_HUB_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${TACC_CUDA_DIR}"
export TRITON_CC="$(command -v gcc)"; export CC="${TRITON_CC}"; export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
S_SCRATCH="${SCRATCH:-$(dirname "${MARIN_ROOT}")}"

# Put the validated wheel ahead of Vista's system NCCL before JAX starts. Merely
# finding 2.30.7 in the venv is insufficient if the dynamic loader maps 2.28.9.
NCCL_LIBRARY=$("${MARIN_PYTHON}" -c 'import glob, os, sys; print(sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))[0])')
export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"

# Command buffers stay off: independently established (took the smoke stage from
# no-checkpoint-in-7h to a verified step-2 checkpoint).
export JAX_ENABLE_PGLE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${SNOWBALL_XLA_EXTRA:-}"
export JAX_COMPILATION_CACHE_DIR="${S_SCRATCH}/jax-cache/snowball-chat"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
# DISABLED (0). PROBABLE cause of the job 966961 crash -- not proven: without a
# native core backtrace we cannot show the faulting instruction was inside
# faulthandler. But the timing is tight and CPython has documented crashes
# inspecting other running threads this way (python/cpython#116008, #140815).
# faulthandler.dump_traceback_later walks every thread stack WITHOUT the GIL. With
# ~100 JAX/NCCL threads per rank in active native code that is the documented
# unsafe case. It fired at T+15min across all 64 ranks; five ranks then segfaulted
# and the rest aborted during distributed teardown, ending a run that was healthy
# at step 92/257 and ~19 minutes from finishing:
#   line 2244  last progress line (step 92)
#   line 2245  Timeout (0:15:00)!  <- the dump
#   line 6207  first Segmentation fault
# The watchdog already detects stalls from OUTSIDE the process via /proc, which is
# strictly safer and needs no in-process instrumentation. Do not re-enable this.
export SNOWBALL_HANG_DUMP_SECONDS=0

# ---- #7344 forensics prerequisites, set BEFORE the run ----
export NCCL_SET_THREAD_NAME=1
export NCCL_RAS_ENABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
FORENSICS="${S_SCRATCH}/logs/forensics/${SLURM_JOB_ID}"
mkdir -p "${FORENSICS}"
export NCCL_DEBUG_FILE="${FORENSICS}/nccl.%h.%p.log"
export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_WIN_ENABLE=0
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"
cd "${MARIN_ROOT}"

# The watchdog reads a log THIS script owns, so it cannot be pointed at the wrong
# file by a submission that omits or renames -o.
RUNLOG="${FORENSICS}/run.log"
: > "${RUNLOG}"
echo "GUARDED_RUN python=${MARIN_PYTHON} stage=${SNOWBALL_STAGE} steps=${SNOWBALL_STEPS} stall_seconds=${STALL_SECONDS}"
echo "GUARDED_RUN runlog=${RUNLOG} forensics=${FORENSICS}"

# Fail before JAX initializes if a fresh environment silently resolved the
# affected 2.28.9 build. The successful Vista runs used exactly 2.30.7.
"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat verify-nccl \
  --expected 2.30.7 || exit 3

# Vista sets kernel.yama.ptrace_scope=2, so gdb/eu-stack CANNOT attach even to a
# child (verified: job 966900). The symbolized C stacks in the #7344 workflow are
# therefore unavailable here. These ptrace-free substitutes reconstruct the coarse
# signature instead (verified: job 966903):
#   spinning in ncclLocalOpAppend  -> state=R, empty wchan, nonvoluntary switches climbing
#   parked in ncclProxyGetPostedOps -> state=S, wchan=futex_wait
#   polling at proxy.cc:986         -> state=S, wchan=poll/epoll
# NCCL_SET_THREAD_NAME=1 makes comm carry real NCCL thread names, so threads are
# identifiable without symbols. Sampled twice to separate spinning from idle.
REMOTE_STACKS='P=$(pgrep -u $USER -f "moe.vista_snowball_chat train" | head -1); echo "=== host $(hostname) pid $P ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null) ==="; echo "--- nccl mapped ---"; grep -i nccl /proc/$P/maps 2>/dev/null | head -5; for pass in 1 2; do echo "--- thread sample $pass ---"; for t in /proc/$P/task/*; do printf "tid=%-8s state=%-2s wchan=%-28s comm=%-18s vol=%s nonvol=%s\n" "$(basename $t)" "$(awk "{print \$3}" $t/stat 2>/dev/null)" "$(cat $t/wchan 2>/dev/null)" "$(cat $t/comm 2>/dev/null)" "$(awk "/^voluntary/{print \$2}" $t/status 2>/dev/null)" "$(awk "/^nonvoluntary/{print \$2}" $t/status 2>/dev/null)"; done; sleep 5; done; echo "--- summary ---"; grep -E "^(Threads|State)" /proc/$P/status'
REMOTE_MAPS='P=$(pgrep -u $USER -f "moe.vista_snowball_chat train" | head -1); echo "host $(hostname) pid $P"; grep -i nccl /proc/$P/maps 2>/dev/null | head -5'
REMOTE_GPU='for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv,noheader; sleep 2; done'

collect_forensics() {
  local tag="$1" out="${FORENSICS}/wedge-${tag}"
  mkdir -p "$out"
  local hosts; hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  echo "WEDGE_FORENSICS_START ${tag} -> ${out}"

  # RAS on EVERY host. It is a cheap socket query and the lagging rank can be any
  # of the 64 -- sampling the first four could miss it completely.
  for h in ${hosts}; do
    ( timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$h" \
        "echo verbose status | nc -w 8 localhost 28028" > "$out/ras.$h.txt" 2>&1 ) &
  done
  wait

  # RAS output is kept as EVIDENCE only -- it is NOT used to pick hosts. RAS
  # reports communicator-wide status, so a response collected from host A can name
  # a lagging rank on host Z; the filename only records where the query ran.
  # Selecting hosts that way would census the wrong node. Census ALL 64 instead:
  # it cannot miss the lagging rank, and it is cheap.
  echo "  RAS collected from all hosts as evidence; censusing all ${SLURM_NNODES} hosts"

  local running=0
  for h in ${hosts}; do
    ( timeout 90 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$h" "${REMOTE_STACKS}" \
        > "$out/threads.$h.txt" 2>&1
      timeout 45 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$h" "${REMOTE_GPU}" \
        > "$out/gpu.$h.txt" 2>&1 ) &
    running=$(( running + 1 ))
    if [ "${running}" -ge 24 ]; then wait -n 2>/dev/null || wait; running=$(( running - 1 )); fi
  done
  wait

  # Surface the discriminator immediately: a RUNNING thread with no wchan is the
  # userspace spin; S + futex_wait is a park. Full per-host detail is in the dir.
  echo "  --- threads in state R with empty wchan (spin candidates) ---"
  grep -h "state=R " "$out"/threads.*.txt 2>/dev/null | grep -E "wchan= +comm|wchan=0" | head -12
  echo "  --- hosts reporting any R-state thread ---"
  grep -l "state=R " "$out"/threads.*.txt 2>/dev/null | sed -E "s#.*/threads\.(.*)\.txt#\1#" | tr "\n" " "; echo
  echo "  --- GPU power spread (spin ~150-215W/0%; compute ~408W/20%) ---"
  cat "$out"/gpu.*.txt 2>/dev/null | awk -F, "{print \$4}" | sort -n | uniq -c | tail -5

  echo "WEDGE_FORENSICS_DONE ${tag}"
}

watchdog() {
  # Driven by CHANGE, not presence: "no progress line has EVER appeared" is a
  # stall like any other. The previous version only counted once a line existed,
  # so it was blind to the step-0 hang that r8 and hero-exact actually exhibited.
  local last="<none>" cur last_change now stalled captures=0
  last_change=$(date +%s)
  while true; do
    sleep 60
    now=$(date +%s)
    cur=$(grep -oE "Progress on:train [0-9.]+it/${SNOWBALL_STEPS}it" "${RUNLOG}" 2>/dev/null | tail -1)
    cur="${cur:-<none>}"
    if [ "${cur}" != "${last}" ]; then
      last="${cur}"; last_change=${now}; captures=0; continue
    fi
    stalled=$(( now - last_change ))
    if [ "${stalled}" -ge "${STALL_SECONDS}" ]; then
      captures=$(( captures + 1 ))
      echo "WEDGE_SUSPECTED capture ${captures}/2 after ${stalled}s at progress=${last}"
      collect_forensics "n${captures}"
      if [ "${captures}" -ge 2 ]; then
        cur=$(grep -oE "Progress on:train [0-9.]+it/${SNOWBALL_STEPS}it" "${RUNLOG}" 2>/dev/null | tail -1)
        if [ "${cur:-<none>}" = "${last}" ]; then
          echo "WEDGE_CONFIRMED: no progress across two captures. Terminating so 64 nodes"
          echo "                 are not held unmonitored. Forensics: ${FORENSICS}"
          scancel "${SLURM_JOB_ID}"
          return
        fi
        echo "  progress resumed during capture; standing down"
        captures=0
      fi
      last_change=${now}
    fi
  done
}

# Provenance: prove which libnccl the ranks actually mapped. The clone's package
# metadata still reads 2.28.9 (only the .so was swapped), so /proc/<pid>/maps and
# the NCCL banner are the only authoritative evidence.
( sleep 240
  for h in $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk 'NR==1 || NR==33'); do
    timeout 30 ssh -o StrictHostKeyChecking=no "$h" "${REMOTE_MAPS}" >> "${FORENSICS}/nccl_provenance.txt" 2>&1 || true
  done ) &

watchdog & WATCHDOG_PID=$!
trap 'kill ${WATCHDOG_PID} 2>/dev/null' EXIT

"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat preflight \
  --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
  --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "${SNOWBALL_OUTPUT}" \
  --run-id "${SNOWBALL_RUN_ID}" --steps "${SNOWBALL_STEPS}" --devices "${SLURM_NTASKS}" \
  --stage "${SNOWBALL_STAGE}" || exit 4

# One srun for the complete requested stage budget.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat train \
    --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
    --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "${SNOWBALL_OUTPUT}" \
    --run-id "${SNOWBALL_RUN_ID}" --steps "${SNOWBALL_STEPS}" --devices "${SLURM_NTASKS}" \
    --stage "${SNOWBALL_STAGE}" 2>&1 | tee -a "${RUNLOG}"
RC=${PIPESTATUS[0]}
kill ${WATCHDOG_PID} 2>/dev/null
echo "GUARDED_RUN_EXIT rc=${RC}"
exit "${RC}"
