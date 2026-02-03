#!/usr/bin/env bash
# cure_pdbs_submit.sh
#
# Usage (submit):
#   ./cure_pdbs_submit.sh /path/to/input_dir /path/to/output_dir
#
# Optional env vars (submit-time):
#   PARTITION=norm
#   CPUS=32
#   TIME=08:00:00
#   FILES_PER_TASK=2000
#   JOBS_PER_TASK=0     # 0 => parallel uses CPUS, else set explicit -j
#   SKIP_EXISTING=1
#
# Notes:
# - Uses a Slurm array; each task uses GNU parallel internally.
# - Requires GNU parallel on compute nodes.

set -euo pipefail
IFS=$'\n\t'

MODE="${1:-}"
if [[ "${MODE}" == "--worker" ]]; then
  # ----------------------------
  # ARRAY WORKER MODE
  # ----------------------------
  shift

  unset DISPLAY XAUTHORITY

  : "${MANIFEST:?must set MANIFEST}"
  : "${IN_ROOT:?must set IN_ROOT}"
  : "${OUT_ROOT:?must set OUT_ROOT}"

  CPUS="${SLURM_CPUS_PER_TASK:-${CPUS:-1}}"
  JOBS_PER_TASK="${JOBS_PER_TASK:-0}"   # 0 => use CPUS
  SKIP_EXISTING="${SKIP_EXISTING:-1}"

  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

  [[ -f "$MANIFEST" ]] || { echo "[ERROR] MANIFEST not found: $MANIFEST" >&2; exit 2; }
  [[ -d "$IN_ROOT"  ]] || { echo "[ERROR] IN_ROOT not dir: $IN_ROOT" >&2; exit 2; }
  mkdir -p "$OUT_ROOT"

  N_TOTAL="$(wc -l < "$MANIFEST" | awk '{print $1}')"
  if [[ "$N_TOTAL" -le 0 ]]; then
    echo "[ERROR] Manifest empty: $MANIFEST" >&2
    exit 3
  fi

  # Balanced partitioning
  START="$(( (TASK_ID * N_TOTAL) / TASK_COUNT + 1 ))"
  END="$(( ((TASK_ID + 1) * N_TOTAL) / TASK_COUNT ))"

  if [[ "$END" -lt "$START" ]]; then
    echo "[INFO] Task ${TASK_ID}: empty shard (N_TOTAL=${N_TOTAL}, TASK_COUNT=${TASK_COUNT})"
    exit 0
  fi

  echo "[INFO] Worker task ${TASK_ID}/${TASK_COUNT} lines ${START}-${END} (N_TOTAL=${N_TOTAL})"
  echo "[INFO] IN_ROOT=$IN_ROOT"
  echo "[INFO] OUT_ROOT=$OUT_ROOT"
  echo "[INFO] CPUS=$CPUS  JOBS_PER_TASK=$JOBS_PER_TASK  SKIP_EXISTING=$SKIP_EXISTING"

  AWK_FILTER='BEGIN{OFS=""}
    ($1=="ATOM" || $1=="HETATM"){
      an=substr($0,13,4)
      if(an==" OXT") next
      elem=substr($0,77,2); gsub(/ /,"",elem)
      if(toupper(elem)=="H") next
    }
    {print}
  '

  export IN_ROOT OUT_ROOT AWK_FILTER SKIP_EXISTING

  process_one() {
    set -euo pipefail
    local in="$1"
    [[ -f "$in" ]] || { echo "[WARN] missing file: $in" >&2; return 0; }

    # preserve relative path if under IN_ROOT
    local rel="${in#${IN_ROOT}/}"
    if [[ "$rel" == "$in" ]]; then
      rel="$(basename "$in")"
    fi

    local base="$(basename "$rel")"
    local dir="$(dirname "$rel")"
    local stem="${base%.*}"

    local outdir="${OUT_ROOT}/${dir}"
    mkdir -p "$outdir"

    local out="${outdir}/${stem}.pdb"
    local tmp="${out}.tmp.$$"

    if [[ "${SKIP_EXISTING}" == "1" && -s "$out" ]]; then
      return 0
    fi

    awk "${AWK_FILTER}" "$in" > "$tmp"
    mv -f "$tmp" "$out"
  }
  export -f process_one

  if ! command -v parallel >/dev/null 2>&1; then
    echo "[ERROR] GNU parallel not found on node. Load it (module load parallel) or install." >&2
    exit 20
  fi

  # choose -j for parallel
  if [[ "$JOBS_PER_TASK" == "0" ]]; then
    JPAR="$CPUS"
  else
    JPAR="$JOBS_PER_TASK"
  fi

  JOBLOG="$(mktemp "${OUT_ROOT}/cure_joblog.${SLURM_JOB_ID:-nojob}.${TASK_ID}.XXXXXX")"

  sed -n "${START},${END}p" "$MANIFEST" \
    | parallel -j "$JPAR" --line-buffer --halt soon,fail=1 --joblog "$JOBLOG" \
        'process_one "{}"'

  n_ok="$(awk 'NR>1 && $7==0 {c++} END{print c+0}' "$JOBLOG")"
  n_fail="$(awk 'NR>1 && $7!=0 {c++} END{print c+0}' "$JOBLOG")"
  rm -f "$JOBLOG"

  echo "[INFO] Completed task ${TASK_ID}: ok=${n_ok} fail=${n_fail}"
  if [[ "$n_fail" -ne 0 ]]; then
    echo "[ERROR] Task ${TASK_ID}: failures detected (fail=${n_fail})" >&2
    exit 10
  fi
  if [[ "$n_ok" -eq 0 ]]; then
    echo "[ERROR] Task ${TASK_ID}: 0 successful outputs for a non-empty shard." >&2
    exit 11
  fi

  exit 0
fi

# ----------------------------
# SUBMIT MODE (default)
# ----------------------------
IN_DIR="${1:?usage: $0 /path/to/input_dir /path/to/output_dir}"
OUT_DIR="${2:?usage: $0 /path/to/input_dir /path/to/output_dir}"

PARTITION="${PARTITION:-norm}"
CPUS="${CPUS:-32}"
TIME="${TIME:-08:00:00}"
FILES_PER_TASK="${FILES_PER_TASK:-2000}"
JOBS_PER_TASK="${JOBS_PER_TASK:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

IN_ROOT="$(readlink -f "$IN_DIR")"
OUT_ROOT="$(readlink -f "$OUT_DIR")"
mkdir -p "$OUT_ROOT" slurm

MANIFEST="${MANIFEST:-${OUT_ROOT}/manifest.pdbs.txt}"

echo "[INFO] IN_DIR  = $IN_DIR"
echo "[INFO] IN_ROOT = $IN_ROOT"
echo "[INFO] OUT_DIR = $OUT_DIR"
echo "[INFO] OUT_ROOT= $OUT_ROOT"
echo "[INFO] Building manifest at $MANIFEST ..."

if [[ ! -d "$IN_ROOT" ]]; then
  echo "[ERROR] Input is not a directory: $IN_DIR -> $IN_ROOT" >&2
  exit 2
fi

# Follow symlinks; include .pdb and .ent
find -L "$IN_ROOT" -type f \( -iname "*.pdb" -o -iname "*.ent" \) -print \
  | LC_ALL=C sort > "$MANIFEST"

N_TOTAL="$(wc -l < "$MANIFEST" | awk '{print $1}')"
if [[ "$N_TOTAL" -le 0 ]]; then
  echo "[ERROR] Found 0 inputs under: $IN_ROOT" >&2
  exit 3
fi

NTASKS="$(( (N_TOTAL + FILES_PER_TASK - 1) / FILES_PER_TASK ))"
if [[ "$NTASKS" -lt 1 ]]; then NTASKS=1; fi

echo "[INFO] Inputs        : $N_TOTAL"
echo "[INFO] Files/task    : $FILES_PER_TASK"
echo "[INFO] Array tasks   : $NTASKS"
echo "[INFO] Partition     : $PARTITION"
echo "[INFO] CPUS/task     : $CPUS"
echo "[INFO] Time          : $TIME"
echo "[INFO] JOBS_PER_TASK : $JOBS_PER_TASK"
echo "[INFO] SKIP_EXISTING : $SKIP_EXISTING"

# Submit THIS script as the worker (--worker mode)
sbatch \
  --job-name=cure_pdbs \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem=0 \
  --time="$TIME" \
  --array="0-$((NTASKS-1))" \
  --output="slurm/cure_pdbs_%A_%a.out" \
  --error="slurm/cure_pdbs_%A_%a.err" \
  --export=ALL,MANIFEST="$MANIFEST",IN_ROOT="$IN_ROOT",OUT_ROOT="$OUT_ROOT",JOBS_PER_TASK="$JOBS_PER_TASK",SKIP_EXISTING="$SKIP_EXISTING" \
  "$0" --worker

echo "[INFO] Submitted."

