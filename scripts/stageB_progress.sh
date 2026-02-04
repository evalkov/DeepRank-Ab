#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<EOF
Usage:
  $0 --run-root RUN_ROOT [--job JOBID] [--interval SECONDS] [--stage A|B]
  $0 --run-root RUN_ROOT [--job JOBID] [--interval SECONDS] --script /path/to/stageX_progress.sh

Options:
  --run-root     Run root containing exchange/
  --job          Slurm array master job ID (optional)
  --interval     Refresh interval seconds (default: 5)
  --stage        A or B (default: A). Auto-select stageA_progress.sh or stageB_progress.sh.
  --script       Explicit progress script path (overrides --stage).
  --top          Pass-through (default: 10)
  --rate-n       Pass-through (default: 20)
  --no-slurm     Pass-through (disable slurm queries)
EOF
}

RUN_ROOT=""
JOBID=""
INTERVAL="${INTERVAL:-5}"
STAGE="${STAGE:-A}"
SCRIPT=""
TOP_N="${TOP_N:-10}"
RATE_N="${RATE_N:-20}"
NO_SLURM_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --job) JOBID="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-A}"; shift 2 ;;
    --script) SCRIPT="${2:-}"; shift 2 ;;
    --top) TOP_N="${2:-10}"; shift 2 ;;
    --rate-n) RATE_N="${2:-20}"; shift 2 ;;
    --no-slurm) NO_SLURM_FLAG="--no-slurm"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  echo "ERROR: --run-root is required" >&2
  usage
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT does not exist: ${RUN_ROOT}" >&2
  exit 2
fi

RUN_ROOT="$(readlink -f "${RUN_ROOT}")"

if [[ -z "${SCRIPT}" ]]; then
  STAGE="$(echo "${STAGE}" | tr '[:lower:]' '[:upper:]')"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  case "${STAGE}" in
    A) SCRIPT="${SCRIPT_DIR}/stageA_progress.sh" ;;
    B) SCRIPT="${SCRIPT_DIR}/stageB_progress.sh" ;;
    *) echo "ERROR: --stage must be A or B (got: ${STAGE})" >&2; exit 2 ;;
  esac
fi

if [[ ! -x "${SCRIPT}" ]]; then
  echo "ERROR: progress script not found or not executable: ${SCRIPT}" >&2
  echo "Fix: chmod +x ${SCRIPT}" >&2
  exit 2
fi

clear_screen() { printf '\033[H\033[2J'; }  # do NOT clear scrollback
hide_cursor() { printf '\033[?25l'; }
show_cursor() { printf '\033[?25h'; }

trap 'printf "\nStopped.\n"; show_cursor; exit 0' INT TERM
hide_cursor

while true; do
  tmp_out="$(mktemp -t watch_progress_out.XXXXXX)"
  tmp_err="$(mktemp -t watch_progress_err.XXXXXX)"
  rc=0

  # ---- Run the progress script FIRST (so we don't show a blank screen while it works) ----
  t0="$(date +%s)"
  set +e
  if [[ -n "${JOBID}" ]]; then
    "${SCRIPT}" --run-root "${RUN_ROOT}" --job "${JOBID}" --top "${TOP_N}" --rate-n "${RATE_N}" ${NO_SLURM_FLAG} >"${tmp_out}" 2>"${tmp_err}"
    rc=$?
  else
    "${SCRIPT}" --run-root "${RUN_ROOT}" --top "${TOP_N}" --rate-n "${RATE_N}" ${NO_SLURM_FLAG} >"${tmp_out}" 2>"${tmp_err}"
    rc=$?
  fi
  set -e
  t1="$(date +%s)"
  dt="$(( t1 - t0 ))"

  # ---- Now paint the screen in one shot ----
  clear_screen
  printf "Stage%s progress (live)\n" "${STAGE}"
  printf "RUN_ROOT: %s\n" "${RUN_ROOT}"
  printf "JOBID:    %s\n" "${JOBID:-"(none)"}"
  printf "SCRIPT:   %s\n" "${SCRIPT}"
  printf "Time:     %(%F %T)T\n" -1
  printf "Refresh:  every %ss   (Ctrl+C to stop)\n" "${INTERVAL}"
  printf "Update:   took %ss\n" "${dt}"
  printf -- "------------------------------------------------------------\n"

  cat "${tmp_out}"

  if (( rc != 0 )); then
    echo
    echo "[watch_progress] progress script exited with rc=${rc}"
    echo "[watch_progress] --- stderr (last 80 lines) ---"
    tail -n 80 "${tmp_err}" || true
    echo "[watch_progress] --------------------------------"
    echo
    echo "Paused so you can read the error. Press Enter to retry."
    read -r _
  fi

  rm -f "${tmp_out}" "${tmp_err}" || true
  sleep "${INTERVAL}"
done

