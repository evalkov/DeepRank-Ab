#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${RUN_ROOT:?set RUN_ROOT}"
: "${DEEPRANK_ROOT:?set DEEPRANK_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}"

SLURM_FILE="${1:-stageB_gpu.slurm}"

SHARDS_PER_TASK="${SHARDS_PER_TASK:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

SHARDS_ROOT="${RUN_ROOT}/exchange/shards"
N_SHARDS="$(find "${SHARDS_ROOT}" -maxdepth 1 -type d -name 'shard_*' 2>/dev/null | wc -l | tr -d ' ')"
if (( N_SHARDS <= 0 )); then
  echo "No shards found in ${SHARDS_ROOT}" >&2
  exit 1
fi

# ceil(N_SHARDS / SHARDS_PER_TASK)
N_TASKS=$(( (N_SHARDS + SHARDS_PER_TASK - 1) / SHARDS_PER_TASK ))
LAST=$(( N_TASKS - 1 ))

echo "RUN_ROOT=${RUN_ROOT}"
echo "N_SHARDS=${N_SHARDS}"
echo "SHARDS_PER_TASK=${SHARDS_PER_TASK}"
echo "Submitting array: 0-${LAST}%${MAX_CONCURRENT}"

sbatch --array="0-${LAST}%${MAX_CONCURRENT}" "${SLURM_FILE}"

