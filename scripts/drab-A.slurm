#!/bin/bash
# All SBATCH directives are set by run_pipeline.py via CLI args.

set -euo pipefail
IFS=$'\n\t'

module purge >/dev/null 2>&1 || true
module load gcc/12.4.0 >/dev/null 2>&1 || true
module load deeprank-ab/latest >/dev/null 2>&1 || true

: "${RUN_ROOT:?set RUN_ROOT (ABSOLUTE path strongly recommended)}"
: "${PDB_ROOT:?set PDB_ROOT (absolute path)}"
: "${DEEPRANK_ROOT:?set DEEPRANK_ROOT e.g. /mnt/beegfs/valkov/soft/DeepRank-Ab}"

# Normalize to absolute paths to prevent accidental duplicate trees
RUN_ROOT="$(readlink -f "$RUN_ROOT")"
PDB_ROOT="$(readlink -f "$PDB_ROOT")"
DEEPRANK_ROOT="$(readlink -f "$DEEPRANK_ROOT")"

HEAVY="${HEAVY:-H}"
LIGHT="${LIGHT:--}"
ANTIGEN="${ANTIGEN:-T}"

# Auto-detect cores: use SLURM allocation, fall back to nproc
NUM_CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"
TARGET_SHARD_GB="${TARGET_SHARD_GB:-0.1}"
MIN_PER_SHARD="${MIN_PER_SHARD:-10}"
MAX_PER_SHARD="${MAX_PER_SHARD:-100}"
GLOB_PAT="${GLOB_PAT:-**/*.pdb}"

# Voronota parallelization settings:
# - VORO_OMP_THREADS: OpenMP threads per voronota subprocess (default: 1)
#   For voronota 1.29 compiled with -qopenmp, increase for intra-voronota parallelism.
#   Trade-off: NUM_CORES × VORO_OMP_THREADS should not exceed total cores.
#   Examples: NUM_CORES=32, VORO_OMP_THREADS=1 -> 32 threads (default)
#             NUM_CORES=16, VORO_OMP_THREADS=2 -> 32 threads
#             NUM_CORES=8,  VORO_OMP_THREADS=4 -> 32 threads
export VORO_OMP_THREADS="${VORO_OMP_THREADS:-1}"
export VORO_PROCESSORS=1
export VORO_CHAIN_PARALLEL=0
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "=== Storage check ==="
echo "TMPDIR=${TMPDIR:-not set}"
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-not set}"
df -h /tmp
df -h "${SLURM_TMPDIR:-/tmp}" 2>/dev/null || echo "SLURM_TMPDIR not available"
echo "===================="

SCRIPT="${DEEPRANK_ROOT}/scripts/split_stageA_cpu.py"

SHARD_LIST_DIR="${RUN_ROOT}/shard_lists"
SHARDS_DIR="${RUN_ROOT}/shards"
mkdir -p "${SHARD_LIST_DIR}" "${SHARDS_DIR}"

READY_FILE="${SHARD_LIST_DIR}/SHARDS_READY"
PARAMS_FILE="${SHARD_LIST_DIR}/shards.params.txt"
LOCK_FILE="${SHARD_LIST_DIR}/.shards.lock"

# ============================================================
# COMPUTE METRICS SAMPLING (StageA-safe: CPU-only is OK)
# ============================================================
COLLECT_COMPUTE_METRICS="${COLLECT_COMPUTE_METRICS:-1}"
COMPUTE_METRICS_INTERVAL="${COMPUTE_METRICS_INTERVAL:-2}"

COMPUTE_METRICS_DIR="${RUN_ROOT}/compute_metrics"
mkdir -p "${COMPUTE_METRICS_DIR}"

MONITOR_SCRIPT="${MONITOR_SCRIPT:-${DEEPRANK_ROOT}/scripts/collect_compute_metrics.py}"

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_TAG="${SLURM_ARRAY_TASK_ID:-0}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname || echo host)"
STAGE_TAG="${STAGE_TAG:-stageA}"
PREFIX="${STAGE_TAG}_${JOB_TAG}_${TASK_TAG}_${HOST_SHORT}"

MONITOR_PID=""

_gpu_list_from_alloc() {
  # If Slurm says GPUs allocated, use that.
  local n=""
  n="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
  if [[ -n "${n}" ]]; then
    n="$(echo "${n}" | sed -E 's/[^0-9].*$//')"
  fi
  if [[ -n "${n}" && "${n}" -ge 1 ]]; then
    python3 - <<PY
n=int("${n}")
print(",".join(str(i) for i in range(n)))
PY
    return 0
  fi

  # If nvidia-smi exists, count GPUs; otherwise: CPU node, return empty.
  if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi -L 2>/dev/null | wc -l | xargs || echo 0)"
    # Ensure n is a single integer
    n="${n%%[!0-9]*}"
    [[ -z "${n}" ]] && n=0
    if [[ "${n}" -ge 1 ]]; then
      python3 - <<PY
n=int("${n}")
print(",".join(str(i) for i in range(n)))
PY
      return 0
    fi
  fi

  # CPU-only node: no GPUs
  echo ""
  return 0
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
  gpu_list="$(_gpu_list_from_alloc)"

  echo "[metrics] Starting sampler -> ${COMPUTE_METRICS_DIR}"
  echo "[metrics]   prefix=${PREFIX} interval=${COMPUTE_METRICS_INTERVAL}s gpus=${gpu_list:-"(none)"}"

  local args=( "--outdir" "${COMPUTE_METRICS_DIR}"
               "--interval" "${COMPUTE_METRICS_INTERVAL}"
               "--prefix" "${PREFIX}"
               "--percore"
               "--net" )

  # Only pass --gpus if we have any
  if [[ -n "${gpu_list}" ]]; then
    args+=( "--gpus" "${gpu_list}" )
  fi

  python3 "${MONITOR_SCRIPT}" "${args[@]}" \
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
if [[ "${MAKE_SHARDS_ONLY:-0}" != "1" ]]; then
  start_monitor
fi

# ============================================================
# VORONOTA / BSA BACKEND
# ============================================================
VORONOTA_BIN="${DEEPRANK_ROOT}/src/tools/voronota/voronota"

if [[ ! -x "${VORONOTA_BIN}" ]]; then
  echo "[StageA] ERROR: voronota binary not found or not executable: ${VORONOTA_BIN}"
  exit 1
fi

# Extract version: "Voronota version 1.29.XXXX" -> "1.29.XXXX"
VORONOTA_VERSION=$("${VORONOTA_BIN}" 2>&1 | head -1 | awk '{print $3}' || true)
[[ -z "${VORONOTA_VERSION}" ]] && VORONOTA_VERSION="unknown"
VORONOTA_BACKEND="voronota v${VORONOTA_VERSION} (additive tessellation)"
BSA_BACKEND="freesasa"
echo "[StageA] ${VORONOTA_BACKEND}, BSA=${BSA_BACKEND}"

# ============================================================
# SHARD PARAMETER STAMP (uniquely defines sharding scheme)
# ============================================================
PARAM_STAMP=$(
  cat <<EOF
PDB_ROOT=${PDB_ROOT}
GLOB_PAT=${GLOB_PAT}
TARGET_SHARD_GB=${TARGET_SHARD_GB}
MIN_PER_SHARD=${MIN_PER_SHARD}
MAX_PER_SHARD=${MAX_PER_SHARD}
HEAVY=${HEAVY}
LIGHT=${LIGHT}
ANTIGEN=${ANTIGEN}
VORONOTA=${VORONOTA_BACKEND}
BSA=${BSA_BACKEND}
EOF
)

build_shards() {
  echo "[StageA] (re)building shard lists in: ${SHARD_LIST_DIR}"
  echo "[StageA] Params:"
  echo "${PARAM_STAMP}" | sed 's/^/[StageA]   /'

  # Remove old shard lists so we don't keep stale shard_*.lst around
  rm -f "${SHARD_LIST_DIR}"/shard_*.lst "${SHARD_LIST_DIR}"/shard_*.tmp "${READY_FILE}"

  python3 "${SCRIPT}" \
    --pdb-root "${PDB_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --make-shards \
    --target-shard-gb "${TARGET_SHARD_GB}" \
    --min-per-shard "${MIN_PER_SHARD}" \
    --max-per-shard "${MAX_PER_SHARD}" \
    --glob "${GLOB_PAT}" \
    --heavy "${HEAVY}" \
    --light "${LIGHT}" \
    --antigen "${ANTIGEN}"

  printf "%s\n" "${PARAM_STAMP}" > "${PARAMS_FILE}"
  echo "ok" > "${READY_FILE}"
  echo "[StageA] Shard lists ready."
}

# ------------------------------------------------------------
# One-time shard list creation (ONLY task 0), with lock + param check
# ------------------------------------------------------------
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  exec 9>"${LOCK_FILE}"
  flock 9

  NEED_BUILD=0
  if [[ ! -f "${READY_FILE}" ]]; then
    NEED_BUILD=1
  elif [[ ! -f "${PARAMS_FILE}" ]]; then
    NEED_BUILD=1
  else
    if ! diff -q <(printf "%s\n" "${PARAM_STAMP}") "${PARAMS_FILE}" >/dev/null 2>&1; then
      echo "[StageA] Detected shard parameter change; will rebuild shard lists."
      NEED_BUILD=1
    fi
  fi

  if [[ "${FORCE_RESHARD:-0}" == "1" ]]; then
    echo "[StageA] FORCE_RESHARD=1 set; will rebuild shard lists."
    NEED_BUILD=1
  fi

  if [[ "${NEED_BUILD}" == "1" ]]; then
    build_shards
  else
    echo "[StageA] Shard lists already ready and parameters match; not rebuilding."
  fi

  flock -u 9

  # If launched as sharding-only job, exit now
  if [[ "${MAKE_SHARDS_ONLY:-0}" == "1" ]]; then
    echo "[StageA] MAKE_SHARDS_ONLY=1; sharding complete, exiting."
    exit 0
  fi
else
  # Wait for shard lists to be created by task 0
  for _ in $(seq 1 900); do
    [[ -f "${READY_FILE}" ]] && break
    sleep 2
  done
  if [[ ! -f "${READY_FILE}" ]]; then
    echo "[StageA] ERROR: timed out waiting for ${READY_FILE}"
    exit 2
  fi
fi

# ------------------------------------------------------------
# Determine how many shard lists exist
# ------------------------------------------------------------
N_SHARDS="$(ls -1 "${SHARD_LIST_DIR}"/shard_*.lst 2>/dev/null | wc -l | tr -d ' ')"
if (( N_SHARDS <= 0 )); then
  echo "[StageA] ERROR: N_SHARDS=0 (no shard_*.lst found in ${SHARD_LIST_DIR})"
  exit 2
fi

TASK="${SLURM_ARRAY_TASK_ID}"
if (( TASK >= N_SHARDS )); then
  echo "[StageA] Task ${TASK} >= N_SHARDS ${N_SHARDS}; exiting."
  exit 0
fi

SHARD_ID="$(printf "%06d" "${TASK}")"

echo "[StageA] RUN_ROOT=${RUN_ROOT}"
echo "[StageA] PDB_ROOT=${PDB_ROOT}"
echo "[StageA] SHARD_LIST_DIR=${SHARD_LIST_DIR}"
echo "[StageA] SHARD_ID=${SHARD_ID} (task ${TASK} / ${N_SHARDS})"
echo "[StageA] CHAINS heavy=${HEAVY} light=${LIGHT} antigen=${ANTIGEN}"
echo "[StageA] CORES=${NUM_CORES}"
echo "[StageA] METRICS prefix=${PREFIX} interval=${COMPUTE_METRICS_INTERVAL}s dir=${COMPUTE_METRICS_DIR}"

python3 "${SCRIPT}" \
  --pdb-root "${PDB_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --shard-id "${SHARD_ID}" \
  --heavy "${HEAVY}" \
  --light "${LIGHT}" \
  --antigen "${ANTIGEN}" \
  --num-cores "${NUM_CORES}"

