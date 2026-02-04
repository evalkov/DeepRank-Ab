#!/usr/bin/env bash
# stageA_progress.sh
# Minimal progress + ETA for StageA shards. Optional compact Slurm summary.
# Also reports PDBs processed/remaining from shard_lists + per-shard meta_stageA.json.

set -euo pipefail
IFS=$'\n\t'

# Avoid literal globs when nothing matches
shopt -s nullglob

RUN_ROOT=""
JOB_ID=""
WATCH=""
TOP_N=10
RATE_N="${RATE_N:-20}"
SHOW_SLURM=1

usage() {
  cat <<EOF
Usage:
  $0 --run-root RUN_ROOT [--job JOB_ID] [--watch SECONDS] [--top N] [--rate-n N] [--no-slurm]
  $0 RUN_ROOT [--job JOB_ID] [--watch SECONDS] [--top N] [--rate-n N] [--no-slurm]

Options:
  --run-root   Path to run root containing exchange/
  --job        Slurm job ID (array master) for compact state summary
  --watch      Refresh every N seconds (runs continuously)
  --top        Show top N most recent completions (default: ${TOP_N})
  --rate-n     Use last N completions for ETA (default: ${RATE_N})
  --no-slurm   Don't query squeue/sacct at all

Environment:
  RATE_N       Same as --rate-n (overridden by --rate-n)

Examples:
  $0 --run-root /mnt/beegfs/.../test_09 --job 50878735
  $0 /mnt/beegfs/.../test_09 --job 50878735 --watch 5
EOF
}

# -------------------- args (supports positional RUN_ROOT) --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --job) JOB_ID="${2:-}"; shift 2 ;;
    --watch) WATCH="${2:-}"; shift 2 ;;
    --top) TOP_N="${2:-10}"; shift 2 ;;
    --rate-n) RATE_N="${2:-20}"; shift 2 ;;
    --no-slurm) SHOW_SLURM=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ -z "${RUN_ROOT}" && "${1}" != -* ]]; then
        RUN_ROOT="$1"; shift 1
      else
        echo "ERROR: Unknown arg: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  echo "ERROR: --run-root (or positional RUN_ROOT) is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT does not exist or is not a directory: ${RUN_ROOT}" >&2
  exit 2
fi

# Resolve RUN_ROOT robustly (readlink -f can fail on some systems; fall back)
if RUN_ROOT_ABS="$(readlink -f "${RUN_ROOT}" 2>/dev/null)"; then
  RUN_ROOT="${RUN_ROOT_ABS}"
else
  # fallback: use python to resolve if available; otherwise keep as-is
  if command -v python3 >/dev/null 2>&1; then
    RUN_ROOT="$(python3 - <<'PY' "${RUN_ROOT}"
import sys
from pathlib import Path
print(str(Path(sys.argv[1]).resolve()))
PY
)"
  fi
fi

SHARD_LIST_DIR="${RUN_ROOT}/exchange/shard_lists"
SHARDS_DIR="${RUN_ROOT}/exchange/shards"

# -------------------- helpers --------------------
pct() {
  local a="$1" b="$2"
  if [[ "$b" -le 0 ]]; then
    echo "0.0"; return
  fi
  awk -v a="$a" -v b="$b" 'BEGIN{ printf "%.1f", (100.0*a)/b }'
}

count_total_shards() {
  local n=0
  if [[ -d "${SHARD_LIST_DIR}" ]]; then
    local lsts=( "${SHARD_LIST_DIR}"/shard_*.lst )
    n="${#lsts[@]}"
  fi
  if [[ "${n}" -le 0 && -d "${SHARDS_DIR}" ]]; then
    local dirs=( "${SHARDS_DIR}"/shard_* )
    n="${#dirs[@]}"
  fi
  echo "${n}"
}

count_done_shards() {
  if [[ ! -d "${SHARDS_DIR}" ]]; then
    echo 0; return
  fi
  local done=( "${SHARDS_DIR}"/shard_*/STAGEA_DONE )
  echo "${#done[@]}"
}

recent_done() {
  if [[ ! -d "${SHARDS_DIR}" ]]; then
    echo "(none)"; return 0
  fi

  # Use ls -t for mtime ordering; guard against transient failures
  local files
  files="$(ls -t "${SHARDS_DIR}"/shard_*/STAGEA_DONE 2>/dev/null | head -n "${TOP_N}" || true)"
  if [[ -z "${files}" ]]; then
    echo "(none)"; return 0
  fi

  while read -r f; do
    [[ -z "$f" ]] && continue
    local sid ts
    sid="$(basename "$(dirname "$f")" | sed 's/^shard_//')"
    ts="$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1 || echo "?")"
    echo "${ts}  ${sid}"
  done <<< "${files}"
}

list_incomplete() {
  if [[ -d "${SHARD_LIST_DIR}" ]]; then
    local lst
    for lst in "${SHARD_LIST_DIR}"/shard_*.lst; do
      [[ -e "$lst" ]] || continue
      local sid
      sid="$(basename "$lst" .lst | sed 's/^shard_//')"
      if [[ ! -f "${SHARDS_DIR}/shard_${sid}/STAGEA_DONE" ]]; then
        echo "${sid}"
      fi
    done
  elif [[ -d "${SHARDS_DIR}" ]]; then
    local d
    for d in "${SHARDS_DIR}"/shard_*; do
      [[ -d "$d" ]] || continue
      local sid
      sid="$(basename "$d" | sed 's/^shard_//')"
      if [[ ! -f "$d/STAGEA_DONE" ]]; then
        echo "${sid}"
      fi
    done
  fi
}

# -------------------- PDB progress block --------------------
pdb_progress_block() {
  if [[ ! -d "${SHARD_LIST_DIR}" ]]; then
    echo "PDBs processed:  (shard_lists missing)"
    echo "PDBs remaining:  (unknown)"
    return 0
  fi

  command -v python3 >/dev/null 2>&1 || {
    echo "PDBs processed:  (python3 not found)"
    echo "PDBs remaining:  (unknown)"
    return 0
  }

  python3 - <<'PY' "${SHARD_LIST_DIR}" "${SHARDS_DIR}"
import json, sys
from pathlib import Path

shard_list_dir = Path(sys.argv[1])
shards_dir = Path(sys.argv[2])

total_pdbs = 0
done_pdbs = 0
done_models = 0
ok_models = 0
fail_models = 0
bad_meta = 0
done_shards = 0

for lst in sorted(shard_list_dir.glob("shard_*.lst")):
    sid = lst.stem.replace("shard_", "")
    try:
        n_lines = sum(1 for line in lst.open("r") if line.strip())
    except Exception:
        n_lines = 0
    total_pdbs += n_lines

    done_flag = shards_dir / f"shard_{sid}" / "STAGEA_DONE"
    if not done_flag.is_file():
        continue

    done_shards += 1
    meta = shards_dir / f"shard_{sid}" / "meta_stageA.json"
    if meta.is_file():
        try:
            j = json.loads(meta.read_text())
            r = j.get("result", {})
            done_pdbs += int(r.get("n_inputs", n_lines))
            done_models += int(r.get("n_models", 0))
            ok_models += int(r.get("n_ok", 0))
            fail_models += int(r.get("n_fail", 0))
        except Exception:
            bad_meta += 1
            done_pdbs += n_lines
    else:
        bad_meta += 1
        done_pdbs += n_lines

remaining = max(0, total_pdbs - done_pdbs)

print(f"PDBs processed:  {done_pdbs} / {total_pdbs}")
print(f"PDBs remaining:  {remaining}")
if done_shards > 0:
    print(f"Prep ok/fail:    {ok_models} ok, {fail_models} fail (models={done_models})")
if bad_meta > 0:
    print(f"Note:            {bad_meta} shards missing/bad meta_stageA.json (used fallbacks)")
PY
}

# -------------------- Slurm compact summary --------------------
SLURM_RUNNING_COUNT=0

slurm_compact_summary() {
  local job="$1" done_shards="$2" total_shards="$3"
  SLURM_RUNNING_COUNT=0

  [[ -z "$job" ]] && return 0
  [[ "$SHOW_SLURM" -eq 0 ]] && return 0
  command -v sacct >/dev/null 2>&1 || return 0

  local sacct_out
  sacct_out="$(sacct -j "${job}" --format=JobIDRaw,State,Elapsed -n -P 2>/dev/null || true)"
  [[ -z "$sacct_out" ]] && return 0

  echo "------------------------------------------------------------"
  echo "Slurm (compact for job ${job}):"

  echo "$sacct_out" | awk -F'|' '
    $1 ~ /\.batch$/ {next}
    $1 ~ /\.extern$/ {next}
    NF>=3 && $1!="" && $2!="" { c[$2]++; tot++ }
    END {
      if (tot==0) { print "  (no records)"; exit }
      split("RUNNING PENDING COMPLETED FAILED CANCELLED TIMEOUT OUT_OF_MEMORY", ord, " ")
      for(i in ord){ s=ord[i]; if(c[s]>0) printf("  %s: %d\n", s, c[s]) }
      for (s in c){
        found=0; for(i in ord){ if (s==ord[i]) found=1 }
        if(!found) printf("  %s: %d\n", s, c[s])
      }
    }'

  echo "$sacct_out" | awk -F'|' '
    function tosec(t,   a,n,d,h,m,s) {
      d=0; if (t ~ /-/) { split(t,a,"-"); d=a[1]; t=a[2] }
      n=split(t,a,":")
      if (n==3){ h=a[1]; m=a[2]; s=a[3] }
      else if(n==2){ h=0; m=a[1]; s=a[2] }
      else { return 0 }
      return (d*86400 + h*3600 + m*60 + s)
    }
    $1 ~ /\.batch$/ {next}
    $1 ~ /\.extern$/ {next}
    $2=="COMPLETED" { sec=tosec($3); if (sec>0){ arr[++k]=sec; sum+=sec } }
    END {
      if (k<1) { print "  COMPLETED runtime: (no data)"; exit }
      for(i=2;i<=k;i++){ v=arr[i]; j=i-1; while(j>=1 && arr[j]>v){ arr[j+1]=arr[j]; j-- } arr[j+1]=v }
      avg=sum/k; p50=arr[int((k+1)/2)]
      printf("  COMPLETED runtime: avg %.1f min, median %.1f min (n=%d)\n", avg/60.0, p50/60.0, k)
    }'

  if command -v squeue >/dev/null 2>&1; then
    SLURM_RUNNING_COUNT="$(squeue -j "${job}" -h -o "%T" 2>/dev/null | awk '$1=="RUNNING"{c++} END{print c+0}')"
    local sq
    sq="$(squeue -j "${job}" -h -o "%.18i %.12T %.10M %.6D %R" 2>/dev/null | head -n 6 || true)"
    if [[ -n "$sq" ]]; then
      echo "  squeue (first few):"
      echo "$sq" | sed 's/^/    /'
    fi
  fi

  if [[ "${done_shards}" -ge "${total_shards}" && "${SLURM_RUNNING_COUNT}" -gt 0 ]]; then
    echo "  Note: filesystem shows all shards DONE for this RUN_ROOT, but ${SLURM_RUNNING_COUNT} Slurm tasks still RUNNING."
    echo "        Common causes: extra array indices, cleanup time, or tasks running against a different RUN_ROOT."
  fi
}

# -------------------- watch UI helpers --------------------
clear_screen() { printf '\033[H\033[2J'; }  # no scrollback clear
hide_cursor() { printf '\033[?25l'; }
show_cursor() { printf '\033[?25h'; }

one_shot() {
  local total done percent
  total="$(count_total_shards)"
  done="$(count_done_shards)"
  percent="$(pct "$done" "$total")"

  # ETA from most recent completions (filesystem mtimes)
  local eta_line="ETA:            (insufficient data)"
  if [[ "$total" -gt 0 && "$done" -gt 1 && "$done" -lt "$total" && -d "${SHARDS_DIR}" ]]; then
    mapfile -t mtimes < <(
      { ls -t "${SHARDS_DIR}"/shard_*/STAGEA_DONE 2>/dev/null || true; } \
        | head -n "${RATE_N}" \
        | xargs -r -n 1 stat -c %Y 2>/dev/null || true
    )

    local k="${#mtimes[@]}"
    if [[ "${k}" -ge 2 ]]; then
      local newest="${mtimes[0]}"
      local oldest="${mtimes[$((k-1))]}"
      local dt=$(( newest - oldest ))

      if [[ "$dt" -gt 0 ]]; then
        local remaining=$(( total - done ))
        local eta_sec
        eta_sec="$(awk -v rem="$remaining" -v k="$k" -v dt="$dt" 'BEGIN{
          rate=(k-1)/dt;
          if(rate<=0) print -1;
          else printf "%.0f", rem/rate
        }')"

        if [[ "${eta_sec}" -gt 0 ]]; then
          local eta_hms finish_epoch finish_str rate_hr
          eta_hms="$(awk -v s="$eta_sec" 'BEGIN{
            h=int(s/3600); m=int((s%3600)/60); ss=int(s%60);
            printf "%02dh:%02dm:%02ds", h,m,ss
          }')"
          finish_epoch=$(( $(date +%s) + eta_sec ))
          finish_str="$(date -d "@${finish_epoch}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "(unknown)")"
          rate_hr="$(awk -v k="$k" -v dt="$dt" 'BEGIN{ printf "%.2f", ((k-1)/dt)*3600 }')"
          eta_line="ETA:            ~${eta_hms} (finish ~${finish_str}) | recent rate ~${rate_hr} shards/hour over last ${k} completions"
        fi
      fi
    fi
  fi

  echo "============================================================"
  if [[ -n "${WATCH}" ]]; then
    echo "StageA progress (live)"
    echo "RUN_ROOT: ${RUN_ROOT}"
    echo "JOBID:    ${JOB_ID:-"(none)"}"
    echo "Time:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refresh:  every ${WATCH}s   (Ctrl+C to stop)"
  else
    echo "StageA progress"
    echo "RUN_ROOT:        ${RUN_ROOT}"
    echo "Time:            $(date '+%Y-%m-%d %H:%M:%S')"
  fi
  echo "------------------------------------------------------------"
  echo "DONE:            ${done} / ${total} (${percent}%)"
  echo "${eta_line}"
  echo "------------------------------------------------------------"
  pdb_progress_block
  echo "------------------------------------------------------------"
  echo "Most recent STAGEA_DONE (top ${TOP_N}):"
  recent_done | sed 's/^/  /'
  echo "------------------------------------------------------------"
  echo "First incomplete shard IDs:"
  list_incomplete | head -n 20 | sed 's/^/  /' || true

  if [[ -n "${JOB_ID}" ]]; then
    slurm_compact_summary "${JOB_ID}" "${done}" "${total}"
  fi
  echo "============================================================"
}

if [[ -n "${WATCH}" ]]; then
  trap 'show_cursor; echo; echo "Stopped."; exit 0' INT TERM
  hide_cursor
  while true; do
    clear_screen
    one_shot
    sleep "${WATCH}"
  done
else
  one_shot
fi

