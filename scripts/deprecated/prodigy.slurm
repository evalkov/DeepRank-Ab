#!/bin/bash
#SBATCH --job-name=prodigy-opt
#SBATCH --partition=norm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=prodigy_opt_%j.out
#SBATCH --error=prodigy_opt_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
IFS=$'\n\t'

# ============================================================
# OPTIMIZED PRODIGY BATCH PROCESSOR
# ============================================================
#
# Key optimizations vs original:
#   1. Uses Prodigy as a Python library (no subprocess per PDB)
#   2. No file copying (reads directly from BeeGFS)
#   3. Higher parallelism (32+ workers per node vs 16)
#   4. Batched work units to reduce IPC overhead
#
# Expected speedup: 5-15x over subprocess-based approach
#
# Usage:
#   sbatch prodigy_optimized.slurm /path/to/pdbs
#   sbatch prodigy_optimized.slurm /path/to/pdbs custom_output_dir
#
# ============================================================

# ============================================================
# USER CONFIG
# ============================================================
INPUT_DIR="${1:-pdbs1K_cured}"
OUT_DIR="${2:-${OUT_DIR:-prodigy_out}}"

SELECTION_A="${SELECTION_A:-H}"
SELECTION_B="${SELECTION_B:-T}"
TEMPERATURE="${TEMPERATURE:-25.0}"
DISTANCE_CUTOFF="${DISTANCE_CUTOFF:-5.5}"
ACC_THRESHOLD="${ACC_THRESHOLD:-0.05}"

# Workers per node (default: use all available CPUs)
# Higher is better since PRODIGY is lightweight
WORKERS_PER_NODE="${WORKERS_PER_NODE:-${SLURM_CPUS_PER_TASK:-32}}"

# Batch size: PDBs per work unit (higher = less IPC overhead)
BATCH_SIZE="${BATCH_SIZE:-100}"

# Path to the optimized worker script (default: same dir as this script)
WORKER_SCRIPT="${WORKER_SCRIPT:-${SLURM_SUBMIT_DIR}/prodigy_batch_worker.py}"

# ============================================================
# ENV / MODULES
# ============================================================
module purge 2>/dev/null || true
module load prodigy/2.4.0 2>/dev/null || true

mkdir -p "${OUT_DIR}"

# Resolve paths
if [[ "${INPUT_DIR}" != /* ]]; then
  INPUT_DIR="${SLURM_SUBMIT_DIR}/${INPUT_DIR}"
fi
INPUT_DIR="$(readlink -f "${INPUT_DIR}")"

# Ensure WORKER_SCRIPT is absolute
if [[ "${WORKER_SCRIPT}" != /* ]]; then
  WORKER_SCRIPT="${SLURM_SUBMIT_DIR}/${WORKER_SCRIPT}"
fi
WORKER_SCRIPT="$(readlink -f "${WORKER_SCRIPT}")"

echo "============================================================"
echo "PRODIGY Optimized Batch Processor"
echo "============================================================"
echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] Submit dir: ${SLURM_SUBMIT_DIR}"
echo "[INFO] Nodes=${SLURM_JOB_NUM_NODES}  Tasks=${SLURM_NTASKS}  CPUs/task=${SLURM_CPUS_PER_TASK}"
echo "[INFO] INPUT_DIR=${INPUT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] Selection: ${SELECTION_A} vs ${SELECTION_B}"
echo "[INFO] Temperature: ${TEMPERATURE}°C"
echo "[INFO] Distance cutoff: ${DISTANCE_CUTOFF}Å"
echo "[INFO] WORKERS_PER_NODE=${WORKERS_PER_NODE}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] WORKER_SCRIPT=${WORKER_SCRIPT}"
echo

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] INPUT_DIR does not exist: ${INPUT_DIR}"
  exit 2
fi

if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "[ERROR] Worker script not found: ${WORKER_SCRIPT}"
  exit 2
fi

# ============================================================
# DISCOVER STRUCTURES
# ============================================================
TMP_DIR="${OUT_DIR}/.tmp_${SLURM_JOB_ID}"
mkdir -p "${TMP_DIR}"

LIST_FILE="${TMP_DIR}/pdb_list.txt"
echo "[INFO] Scanning for structures under: ${INPUT_DIR}"

find -L "${INPUT_DIR}" -type f \
  \( -iname "*.pdb" -o -iname "*.cif" -o -iname "*.mmcif" \) \
  ! -iname "manifest*" \
  -print | sort > "${LIST_FILE}"

TOTAL="$(wc -l < "${LIST_FILE}" | tr -d ' ')"
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[ERROR] No structure files found under ${INPUT_DIR}"
  exit 2
fi
echo "[INFO] Total structures discovered: ${TOTAL}"

# ============================================================
# COMPUTE WORK DISTRIBUTION
# ============================================================
NTASKS="${SLURM_NTASKS}"
PER_TASK=$(( (TOTAL + NTASKS - 1) / NTASKS ))

echo "[INFO] Work distribution: ${TOTAL} structures / ${NTASKS} tasks = ~${PER_TASK} per task"
echo

# ============================================================
# WORKER FUNCTION
# ============================================================
worker() {
  set -euo pipefail

  local rank="${SLURM_PROCID}"
  local ntasks="${SLURM_NTASKS}"
  local cpus="${SLURM_CPUS_PER_TASK}"

  # Calculate this rank's portion
  local per=$(( (TOTAL + ntasks - 1) / ntasks ))
  local start=$(( rank * per ))
  local count="${per}"

  # Check if we have work to do
  if (( start >= TOTAL )); then
    echo "[Rank ${rank}] No work (start ${start} >= total ${TOTAL})"
    exit 0
  fi

  # Adjust count for last rank
  if (( start + count > TOTAL )); then
    count=$(( TOTAL - start ))
  fi

  echo "[Rank ${rank}] Processing ${count} structures starting at index ${start} on $(hostname)"

  local out_tsv="${TMP_DIR}/rank_$(printf "%02d" "${rank}").tsv"

  # Run the optimized Python worker
  # NO file copying - reads directly from BeeGFS
  python3 "${WORKER_SCRIPT}" \
    --input-dir "${INPUT_DIR}" \
    --output-tsv "${out_tsv}" \
    --file-list "${LIST_FILE}" \
    --selection-a "${SELECTION_A}" \
    --selection-b "${SELECTION_B}" \
    --temperature "${TEMPERATURE}" \
    --distance-cutoff "${DISTANCE_CUTOFF}" \
    --acc-threshold "${ACC_THRESHOLD}" \
    --workers "${WORKERS_PER_NODE}" \
    --batch-size "${BATCH_SIZE}" \
    --start-index "${start}" \
    --count "${count}"

  echo "[Rank ${rank}] Completed -> ${out_tsv}"
}

export -f worker
export TOTAL TMP_DIR LIST_FILE WORKER_SCRIPT
export INPUT_DIR OUT_DIR
export SELECTION_A SELECTION_B TEMPERATURE DISTANCE_CUTOFF ACC_THRESHOLD
export WORKERS_PER_NODE BATCH_SIZE

# ============================================================
# LAUNCH WORKERS
# ============================================================
echo "[INFO] Launching ${SLURM_NTASKS} workers..."
T_START=$(date +%s)

srun \
  --ntasks="${SLURM_NTASKS}" \
  --nodes="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --distribution=cyclic \
  bash --noprofile --norc -c 'worker'

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

# ============================================================
# MERGE RESULTS
# ============================================================
echo
echo "[INFO] Merging results..."

FINAL_TSV="${OUT_DIR}/prodigy_${SLURM_JOB_ID}.tsv"

# Get header from first rank file
first_rank="${TMP_DIR}/rank_00.tsv"
if [[ -f "${first_rank}" ]]; then
  head -n 1 "${first_rank}" > "${FINAL_TSV}"
else
  echo "[ERROR] No rank output files found"
  exit 1
fi

# Merge all rank files (skip headers, sort by PDB name)
tail -n +2 -q "${TMP_DIR}"/rank_*.tsv \
  | awk 'NF>0' \
  | sort -t $'\t' -k1,1 \
  >> "${FINAL_TSV}"

# Count results
ROWS=$(($(wc -l < "${FINAL_TSV}" | tr -d ' ') - 1))
N_OK=$(awk -F'\t' 'NR>1 && $17=="ok" {count++} END {print count+0}' "${FINAL_TSV}")
N_FAIL=$(awk -F'\t' 'NR>1 && $17=="fail" {count++} END {print count+0}' "${FINAL_TSV}")

# Analyze failures if any exist
if (( N_FAIL > 0 )); then
  echo
  echo "[INFO] Analyzing failures (showing first 20)..."
  echo "PDB_NAME                ERROR"
  echo "----------------------------------------------------"
  awk -F'\t' 'NR>1 && $17=="fail" {printf "%-23s %s\n", $1, substr($18, 1, 80)}' "${FINAL_TSV}" | head -20
fi

# ============================================================
# SUMMARY
# ============================================================
echo
echo "============================================================"
echo "PRODIGY Batch Processing Complete"
echo "============================================================"
echo "[INFO] Final TSV: ${FINAL_TSV}"
echo "[INFO] Total rows: ${ROWS}"
echo "[INFO] Succeeded: ${N_OK}"
echo "[INFO] Failed: ${N_FAIL}"

# Calculate failure percentage
if (( ROWS > 0 )); then
  FAIL_PCT=$(echo "scale=2; 100 * ${N_FAIL} / ${ROWS}" | bc)
  echo "[INFO] Failure rate: ${FAIL_PCT}%"
fi

echo "[INFO] Elapsed time: ${ELAPSED}s"
if (( ROWS > 0 && ELAPSED > 0 )); then
  RATE=$(echo "scale=1; ${ROWS} / ${ELAPSED}" | bc)
  echo "[INFO] Throughput: ${RATE} structures/sec"
fi
echo

# Cleanup temp directory
rm -rf "${TMP_DIR}"

# Exit with error only if failure rate exceeds 5%
if (( ROWS > 0 )); then
  FAIL_THRESHOLD=5
  FAIL_EXCEEDS=$(echo "${FAIL_PCT} > ${FAIL_THRESHOLD}" | bc -l)
  
  if (( FAIL_EXCEEDS )); then
    echo "[ERROR] Failure rate ${FAIL_PCT}% exceeds threshold of ${FAIL_THRESHOLD}%"
    exit 1
  elif (( N_FAIL > 0 )); then
    echo "[WARN] ${N_FAIL} structures failed (${FAIL_PCT}%), but under ${FAIL_THRESHOLD}% threshold - treating as success"
  fi
fi

exit 0
