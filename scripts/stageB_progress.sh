#!/usr/bin/env bash
# stageB_progress.sh
# Fast StageB progress + ETA for GPU inference.
#
# Key improvements vs prior versions:
#   - NO HDF5 reads (no h5py needed)
#   - Caches shard list line counts -> instant refresh under watch
#   - ETA uses nanosecond mtimes (avoids dt=0 when multiple DONE files land in same second)
#   - Guards module usage (watch/non-login shells)

set -euo pipefail
IFS=$'\n\t'
shopt -s nullglob

# Optional module load (guarded; won't hard-fail under watch/non-login shells)
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load deeprank-ab/latest >/dev/null 2>&1 || true
fi

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
  --run-root   Path to run root
  --job        Slurm job ID (array master) for compact state summary
  --watch      Refresh every N seconds (runs continuously)
  --top        Show top N most recent DONE shards (default: ${TOP_N})
  --rate-n     Use last N DONE mtimes for ETA (default: ${RATE_N})
  --no-slurm   Don't query squeue/sacct at all
EOF
}

# -------------------- args (RUN_ROOT can be positional) --------------------
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

# Resolve RUN_ROOT robustly
if RUN_ROOT_ABS="$(readlink -f "${RUN_ROOT}" 2>/dev/null)"; then
  RUN_ROOT="${RUN_ROOT_ABS}"
else
  if command -v python3 >/dev/null 2>&1; then
    RUN_ROOT="$(python3 - <<'PY' "${RUN_ROOT}"
import sys
from pathlib import Path
print(str(Path(sys.argv[1]).resolve()))
PY
)"
  fi
fi

SHARDS_DIR="${RUN_ROOT}/shards"
PREDS_DIR="${RUN_ROOT}/preds"
SHARD_LIST_DIR="${RUN_ROOT}/shard_lists"

CACHE_COUNTS="${SHARD_LIST_DIR}/shard_line_counts.tsv"   # sid \t n_lines
CACHE_META="${SHARD_LIST_DIR}/shard_line_counts.meta"    # tiny stamp file

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

count_stageA_ready() {
  if [[ ! -d "${SHARDS_DIR}" ]]; then
    echo 0; return
  fi
  local done=( "${SHARDS_DIR}"/shard_*/STAGEA_DONE )
  echo "${#done[@]}"
}

count_stageB_done() {
  if [[ ! -d "${PREDS_DIR}" ]]; then
    echo 0; return
  fi
  local done=( "${PREDS_DIR}"/DONE_shard_*.ok )
  echo "${#done[@]}"
}

recent_done() {
  if [[ ! -d "${PREDS_DIR}" ]]; then
    echo "(none)"; return 0
  fi
  local files
  files="$(ls -t "${PREDS_DIR}"/DONE_shard_*.ok 2>/dev/null | head -n "${TOP_N}" || true)"
  if [[ -z "${files}" ]]; then
    echo "(none)"; return 0
  fi
  while read -r f; do
    [[ -z "$f" ]] && continue
    local sid ts
    sid="$(basename "$f" | sed -E 's/^DONE_shard_([0-9]+)\.ok$/\1/')"
    ts="$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1 || echo "?")"
    echo "${ts}  ${sid}"
  done <<< "${files}"
}

list_incomplete() {
  [[ -d "${SHARDS_DIR}" ]] || return 0
  [[ -d "${PREDS_DIR}" ]] || return 0

  if [[ -d "${SHARD_LIST_DIR}" ]]; then
    local lst
    for lst in "${SHARD_LIST_DIR}"/shard_*.lst; do
      [[ -e "$lst" ]] || continue
      local sid
      sid="$(basename "$lst" .lst | sed 's/^shard_//')"
      [[ -f "${SHARDS_DIR}/shard_${sid}/STAGEA_DONE" ]] || continue
      [[ -f "${PREDS_DIR}/DONE_shard_${sid}.ok" ]] && continue
      echo "${sid}"
    done
  else
    local d
    for d in "${SHARDS_DIR}"/shard_*; do
      [[ -d "$d" ]] || continue
      local sid
      sid="$(basename "$d" | sed 's/^shard_//')"
      [[ -f "$d/STAGEA_DONE" ]] || continue
      [[ -f "${PREDS_DIR}/DONE_shard_${sid}.ok" ]] && continue
      echo "${sid}"
    done
  fi
}

# ------------------------------------------------------------
# Build/refresh cached shard line counts (fast subsequent reads)
# Cache is considered valid if it is newer than all shard_*.lst files.
# ------------------------------------------------------------
ensure_cache() {
  [[ -d "${SHARD_LIST_DIR}" ]] || return 0
  command -v python3 >/dev/null 2>&1 || return 0

  python3 - <<'PY' "${SHARD_LIST_DIR}" "${CACHE_COUNTS}" "${CACHE_META}"
import sys, os, time
from pathlib import Path

lists = Path(sys.argv[1])
cache = Path(sys.argv[2])
meta  = Path(sys.argv[3])

lst_files = sorted(lists.glob("shard_*.lst"))
if not lst_files:
    # nothing to cache
    raise SystemExit(0)

def newest_mtime_ns(paths):
    m = 0
    for p in paths:
        try:
            m = max(m, p.stat().st_mtime_ns)
        except Exception:
            pass
    return m

need = True
if cache.is_file():
    try:
        cache_m = cache.stat().st_mtime_ns
        lists_m = newest_mtime_ns(lst_files)
        # If cache is newer than all lists, keep it
        if cache_m >= lists_m:
            need = False
    except Exception:
        need = True

if not need:
    raise SystemExit(0)

tmp = cache.with_suffix(".tmp")
counts = {}
total = 0
for lf in lst_files:
    sid = lf.stem.replace("shard_", "")
    n = 0
    try:
        with lf.open("r") as fh:
            for ln in fh:
                if ln.strip():
                    n += 1
    except Exception:
        n = 0
    counts[sid] = n
    total += n

with tmp.open("w") as out:
    out.write("# sid\tpdb_count\n")
    for sid in sorted(counts.keys()):
        out.write(f"{sid}\t{counts[sid]}\n")

tmp.replace(cache)
meta.write_text(f"generated_at={time.strftime('%F %T')}\nshards={len(counts)}\nexpected_pdbs={total}\n")
PY
}

# ------------------------------------------------------------
# PDB progress block (NO HDF5)
# Uses cached shard line counts + DONE_shard_*.ok IDs
# ------------------------------------------------------------
pdb_progress_block() {
  if [[ ! -d "${SHARD_LIST_DIR}" ]]; then
    echo "PDBs completed:  (shard_lists missing; cannot estimate)"
    echo "PDBs remaining:  (unknown)"
    return 0
  fi
  if [[ ! -d "${PREDS_DIR}" ]]; then
    echo "PDBs completed:  0 / (unknown)"
    echo "PDBs remaining:  (unknown)"
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || {
    echo "PDBs completed:  (python3 not found)"
    echo "PDBs remaining:  (unknown)"
    return 0
  }

  ensure_cache || true

  python3 - <<'PY' "${SHARD_LIST_DIR}" "${PREDS_DIR}" "${CACHE_COUNTS}" "${CACHE_META}"
import re, sys
from pathlib import Path

lists = Path(sys.argv[1])
preds = Path(sys.argv[2])
cache = Path(sys.argv[3])
meta  = Path(sys.argv[4])

# Load cached counts if present; otherwise fall back to reading lists (slower)
counts = {}
expected = 0

if cache.is_file():
    try:
        for ln in cache.read_text().splitlines():
            if not ln or ln.startswith("#"):
                continue
            sid, n = ln.split("\t", 1)
            n = int(n)
            counts[sid] = n
            expected += n
    except Exception:
        counts = {}
        expected = 0

if not counts:
    for lst in lists.glob("shard_*.lst"):
        sid = lst.stem.replace("shard_", "")
        n = 0
        try:
            with lst.open("r") as fh:
                n = sum(1 for ln in fh if ln.strip())
        except Exception:
            n = 0
        counts[sid] = n
        expected += n

done_sids = []
for f in preds.glob("DONE_shard_*.ok"):
    m = re.search(r"DONE_shard_(\d+)\.ok$", f.name)
    if m:
        done_sids.append(m.group(1))

completed = 0
missing = 0
for sid in done_sids:
    if sid in counts:
        completed += counts[sid]
    else:
        missing += 1

remaining = max(0, expected - completed)

print(f"PDBs completed:  {completed} / {expected}")
print(f"PDBs remaining:  {remaining}")
print(f"DONE shards:     {len(done_sids)}")
if meta.is_file():
    try:
        for ln in meta.read_text().splitlines():
            if ln.startswith("generated_at="):
                print(f"Counts cache:     {ln.split('=',1)[1]}")
                break
    except Exception:
        pass
if missing:
    print(f"Note:            {missing} DONE shards missing in shard count map (unexpected)")
PY
}

# ------------------------------------------------------------
# ETA using DONE file mtimes with nanosecond resolution
# ------------------------------------------------------------
eta_line() {
  local total="$1" done="$2"
  if [[ "$total" -le 0 || "$done" -le 1 || "$done" -ge "$total" || ! -d "${PREDS_DIR}" ]]; then
    echo "ETA:            (insufficient data)"
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || {
    echo "ETA:            (insufficient data)"
    return 0
  }

  python3 - <<'PY' "${PREDS_DIR}" "${RATE_N}" "${total}" "${done}"
import sys
from pathlib import Path

preds = Path(sys.argv[1])
rate_n = int(sys.argv[2])
total = int(sys.argv[3])
done = int(sys.argv[4])

files = sorted(preds.glob("DONE_shard_*.ok"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
files = files[:max(2, rate_n)]
if len(files) < 2:
    print("ETA:            (insufficient data)")
    raise SystemExit(0)

newest = files[0].stat().st_mtime_ns
oldest = files[-1].stat().st_mtime_ns
dt_s = (newest - oldest) / 1e9
k = len(files)

if dt_s <= 0:
    print("ETA:            (insufficient data)")
    raise SystemExit(0)

rate = (k - 1) / dt_s  # shards/sec
if rate <= 0:
    print("ETA:            (insufficient data)")
    raise SystemExit(0)

remaining = total - done
eta_s = remaining / rate

# format eta
h = int(eta_s // 3600)
m = int((eta_s % 3600) // 60)
s = int(eta_s % 60)

import time
finish_epoch = int(time.time() + eta_s)
finish_str = time.strftime("%F %T", time.localtime(finish_epoch))
rate_hr = rate * 3600.0

print(f"ETA:            ~{h:02d}h:{m:02d}m:{s:02d}s (finish ~{finish_str}) | recent rate ~{rate_hr:.2f} shards/hour over last {k} completions")
PY
}

# ------------------------------------------------------------
# Slurm compact summary (unchanged)
# ------------------------------------------------------------
SLURM_RUNNING_COUNT=0

slurm_compact_summary() {
  local job="$1" done_b="$2" total_b="$3"
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
      d=0
      if (t ~ /-/) { split(t,a,"-"); d=a[1]; t=a[2] }
      n=split(t,a,":")
      if (n==3){ h=a[1]; m=a[2]; s=a[3] }
      else if(n==2){ h=0; m=a[1]; s=a[2] }
      else { return 0 }
      return (d*86400 + h*3600 + m*60 + s)
    }
    $1 ~ /\.batch$/ {next}
    $1 ~ /\.extern$/ {next}
    $2=="COMPLETED" { sec=tosec($3); if(sec>0){ arr[++k]=sec; sum+=sec } }
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

  if [[ "${done_b}" -ge "${total_b}" && "${SLURM_RUNNING_COUNT}" -gt 0 ]]; then
    echo "  Note: filesystem shows all StageB DONE for this RUN_ROOT, but ${SLURM_RUNNING_COUNT} Slurm tasks still RUNNING."
    echo "        Common causes: extra array indices, cleanup time, or tasks running against a different RUN_ROOT."
  fi
}

# ------------------------------------------------------------
# Watch UI helpers
# ------------------------------------------------------------
clear_screen() { printf '\033[H\033[2J'; }  # do NOT clear scrollback
hide_cursor() { printf '\033[?25l'; }
show_cursor() { printf '\033[?25h'; }

one_shot() {
  local total stageA_ready done percent
  total="$(count_total_shards)"
  stageA_ready="$(count_stageA_ready)"
  done="$(count_stageB_done)"
  percent="$(pct "$done" "$total")"

  echo "============================================================"
  if [[ -n "${WATCH}" ]]; then
    echo "StageB progress (live)"
    echo "RUN_ROOT: ${RUN_ROOT}"
    echo "JOBID:    ${JOB_ID:-"(none)"}"
    echo "Time:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refresh:  every ${WATCH}s   (Ctrl+C to stop)"
  else
    echo "StageB progress"
    echo "RUN_ROOT:        ${RUN_ROOT}"
    echo "Time:            $(date '+%Y-%m-%d %H:%M:%S')"
  fi
  echo "------------------------------------------------------------"
  echo "StageA-ready:    ${stageA_ready} / ${total}"
  echo "StageB DONE:     ${done} / ${total} (${percent}%)"
  eta_line "${total}" "${done}"
  echo "------------------------------------------------------------"
  pdb_progress_block
  echo "------------------------------------------------------------"
  echo "Most recent DONE_shard (top ${TOP_N}):"
  recent_done | sed 's/^/  /'
  echo "------------------------------------------------------------"
  echo "First incomplete shard IDs (StageA-ready but StageB not DONE):"
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

