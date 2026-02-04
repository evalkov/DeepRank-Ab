#!/bin/bash
#SBATCH --job-name=drab-B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=%x_%A_%a.log
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
IFS=$'\n\t'

module purge >/dev/null 2>&1 || true
module load deeprank-ab/latest >/dev/null 2>&1 || true

: "${RUN_ROOT:?set RUN_ROOT}"
: "${DEEPRANK_ROOT:?set DEEPRANK_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}"

# Normalize to absolute paths (prevents accidental duplicate trees)
RUN_ROOT="$(readlink -f "$RUN_ROOT")"
DEEPRANK_ROOT="$(readlink -f "$DEEPRANK_ROOT")"
MODEL_PATH="$(readlink -f "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")"

SCRIPT="${DEEPRANK_ROOT}/scripts/split_stageB_gpu.py"

# ============================================================
# COMPUTE METRICS SAMPLING (StageB GPU)
# ============================================================
COLLECT_COMPUTE_METRICS="${COLLECT_COMPUTE_METRICS:-1}"
COMPUTE_METRICS_INTERVAL="${COMPUTE_METRICS_INTERVAL:-2}"

COMPUTE_METRICS_DIR="${RUN_ROOT}/exchange/compute_metrics"
mkdir -p "${COMPUTE_METRICS_DIR}"

MONITOR_SCRIPT="${MONITOR_SCRIPT:-${DEEPRANK_ROOT}/scripts/collect_compute_metrics.py}"

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_TAG="${SLURM_ARRAY_TASK_ID:-0}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname || echo host)"
STAGE_TAG="${STAGE_TAG:-stageB}"
PREFIX="${STAGE_TAG}_${JOB_TAG}_${TASK_TAG}_${HOST_SHORT}"

MONITOR_PID=""

_gpu_list_from_alloc() {
  # Prefer Slurm-provided GPU counts when available
  local n=""

  # Common Slurm envs: SLURM_GPUS_ON_NODE or SLURM_GPUS_PER_NODE
  n="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
  if [[ -n "${n}" ]]; then
    # strip anything after first non-digit
    n="$(echo "${n}" | sed -E 's/[^0-9].*$//')"
  fi

  if [[ -n "${n}" && "${n}" -ge 1 ]]; then
    python3 - <<PY
n=int("${n}")
print(",".join(str(i) for i in range(n)))
PY
    return 0
  fi

  # Fallback: count via nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
    if [[ "${n}" -ge 1 ]]; then
      python3 - <<PY
n=int("${n}")
print(",".join(str(i) for i in range(n)))
PY
      return 0
    fi
  fi

  return 1
}

start_monitor() {
  if [[ "${COLLECT_COMPUTE_METRICS}" != "1" ]]; then
    echo "[metrics] disabled (COLLECT_COMPUTE_METRICS=${COLLECT_COMPUTE_METRICS})"
    return 0
  fi

  if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
    echo "[metrics] WARNING: monitor script not found: ${MONITOR_SCRIPT} (skipping)"
    return 0
  fi

  local gpu_list=""
  if ! gpu_list="$(_gpu_list_from_alloc)"; then
    echo "[metrics] WARNING: could not determine GPUs (skipping)"
    return 0
  fi

  echo "[metrics] Starting sampler -> ${COMPUTE_METRICS_DIR}"
  echo "[metrics]   prefix=${PREFIX} interval=${COMPUTE_METRICS_INTERVAL}s gpus=${gpu_list}"

  python3 "${MONITOR_SCRIPT}" \
    --outdir "${COMPUTE_METRICS_DIR}" \
    --interval "${COMPUTE_METRICS_INTERVAL}" \
    --gpus "${gpu_list}" \
    --prefix "${PREFIX}" \
    --percore \
    --net \
    > "${COMPUTE_METRICS_DIR}/${PREFIX}.monitor.log" 2>&1 &

  MONITOR_PID=$!
}

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    echo "[metrics] Stopping sampler PID=${MONITOR_PID} ..."
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${MONITOR_PID}" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

trap cleanup_monitor EXIT
start_monitor

# ============================================================
# StageB shard selection
# ============================================================
SHARDS_ROOT="${RUN_ROOT}/exchange/shards"
N_SHARDS="$(find "${SHARDS_ROOT}" -maxdepth 1 -type d -name 'shard_*' 2>/dev/null | wc -l | tr -d ' ')"
if (( N_SHARDS == 0 )); then
  echo "[StageB] No shards found in ${SHARDS_ROOT}"
  exit 1
fi

SHARDS_PER_TASK="${SHARDS_PER_TASK:-20}"

TASK="${SLURM_ARRAY_TASK_ID}"
START=$(( TASK * SHARDS_PER_TASK ))
if (( START >= N_SHARDS )); then
  echo "[StageB] Task ${TASK}: START=${START} >= N_SHARDS=${N_SHARDS}; exiting."
  exit 0
fi

COUNT="${SHARDS_PER_TASK}"

echo "[StageB] RUN_ROOT=${RUN_ROOT}"
echo "[StageB] N_SHARDS=${N_SHARDS}"
echo "[StageB] TASK=${TASK} START=${START} COUNT=${COUNT} (SHARDS_PER_TASK=${SHARDS_PER_TASK})"
echo "[StageB] MODEL_PATH=${MODEL_PATH}"
echo "[StageB] METRICS prefix=${PREFIX} interval=${COMPUTE_METRICS_INTERVAL}s dir=${COMPUTE_METRICS_DIR}"

# Tuneables
export ESM_GPUS="${ESM_GPUS:-4}"
export ESM_TOKS_PER_BATCH="${ESM_TOKS_PER_BATCH:-12288}"
export ESM_SCALAR_DTYPE="${ESM_SCALAR_DTYPE:-float16}"
export DL_WORKERS="${DL_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_CORES="${NUM_CORES:-${SLURM_CPUS_PER_TASK}}"

python3 "${SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --start-index "${START}" \
  --count "${COUNT}" \
  --model-path "${MODEL_PATH}" \
  --device cuda \
  --num-cores "${NUM_CORES}" \
  --dl-workers "${DL_WORKERS}" \
  --batch-size "${BATCH_SIZE}"

