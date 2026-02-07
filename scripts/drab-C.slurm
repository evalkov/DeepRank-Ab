#!/bin/bash
# All SBATCH directives are set by run_pipeline.py via CLI args.

set -euo pipefail
IFS=$'\n\t'

module purge >/dev/null 2>&1 || true
module load gcc/12.4.0 >/dev/null 2>&1 || true
module load deeprank-ab/latest >/dev/null 2>&1 || true

: "${RUN_ROOT:?set RUN_ROOT to your run dir}"
: "${DEEPRANK_ROOT:?set DEEPRANK_ROOT (DeepRank-Ab repo root)}"

RUN_ROOT="$(readlink -f "${RUN_ROOT}")"
PRED_DIR="${RUN_ROOT}/preds"
OUT_DIR="${RUN_ROOT}/summary"
mkdir -p "${OUT_DIR}"

MERGE_SCRIPT="${DEEPRANK_ROOT}/scripts/merge_pred_hdf5.py"
MERGED_H5="${OUT_DIR}/predictions_merged.h5"
OUT_TSV="${OUT_DIR}/all_predictions.tsv.gz"
OUT_STATS="${OUT_DIR}/stats.json"

echo "[StageC] RUN_ROOT=${RUN_ROOT}"
echo "[StageC] PRED_DIR=${PRED_DIR}"
echo "[StageC] OUT_DIR=${OUT_DIR}"

# ------------------------------------------------------------
# 0) Wait for all Stage B shards to finish (done or failed)
# ------------------------------------------------------------
SHARD_LIST_DIR="${RUN_ROOT}/shard_lists"
POLL_TIMEOUT="${STAGEC_POLL_TIMEOUT:-86400}"
POLL_INTERVAL=2
MAX_ITER=$(( POLL_TIMEOUT / POLL_INTERVAL ))

TOTAL="$(ls -1 "${SHARD_LIST_DIR}"/shard_*.lst 2>/dev/null | wc -l | tr -d ' ')"
if (( TOTAL <= 0 )); then
  echo "[StageC] ERROR: no shard lists in ${SHARD_LIST_DIR}" >&2; exit 2
fi

echo "[StageC] Waiting for ${TOTAL} shards (timeout ${POLL_TIMEOUT}s)..."
PREV_MSG=""
for _ in $(seq 1 "${MAX_ITER}"); do
  N_DONE="$(ls -1 "${PRED_DIR}"/DONE_shard_*.ok 2>/dev/null | wc -l | tr -d ' ')"
  N_FAIL="$(ls -1 "${PRED_DIR}"/FAILED_shard_*.txt 2>/dev/null | wc -l | tr -d ' ')"
  GOT=$(( N_DONE + N_FAIL ))
  MSG="${N_DONE} done, ${N_FAIL} failed, $(( TOTAL - GOT )) pending"
  if [[ "${MSG}" != "${PREV_MSG}" ]]; then
    echo "[StageC] ${MSG}"; PREV_MSG="${MSG}"
  fi
  (( GOT >= TOTAL )) && break
  sleep "${POLL_INTERVAL}"
done

if (( GOT < TOTAL )); then
  echo "[StageC] ERROR: timeout — ${GOT}/${TOTAL} accounted for" >&2; exit 2
fi

# Require ALL shards to have succeeded — no partial merges
if (( N_FAIL > 0 )); then
  echo "[StageC] ERROR: ${N_FAIL} of ${TOTAL} shard(s) failed:" >&2
  for ff in "${PRED_DIR}"/FAILED_shard_*.txt; do
    sid="$(basename "$ff" .txt)"
    sid="${sid#FAILED_shard_}"
    echo "[StageC]   shard_${sid}" >&2
  done
  echo "[StageC] Re-run failed shards, then re-submit Stage C." >&2
  exit 2
fi

echo "[StageC] All ${TOTAL} shards succeeded"

# ------------------------------------------------------------
# 1) Gather inputs ONLY for shards that are DONE
# ------------------------------------------------------------
mapfile -t DONE_FILES < <(ls -1 "${PRED_DIR}"/DONE_shard_*.ok 2>/dev/null | sort || true)
if (( ${#DONE_FILES[@]} == 0 )); then
  echo "ERROR: no DONE_shard_*.ok found in ${PRED_DIR}" >&2
  exit 2
fi

# Build list of pred files in the same order
PRED_FILES=()
MISSING=0
for d in "${DONE_FILES[@]}"; do
  sid="$(basename "$d" | sed -E 's/^DONE_shard_([0-9]+)\.ok$/\1/')"
  pf="${PRED_DIR}/pred_shard_${sid}.h5"
  if [[ -f "${pf}" ]]; then
    PRED_FILES+=( "${pf}" )
  else
    echo "ERROR: DONE exists but missing pred file: ${pf}" >&2
    MISSING=$((MISSING+1))
  fi
done
if (( MISSING > 0 )); then
  echo "ERROR: ${MISSING} pred files missing for DONE shards" >&2
  exit 2
fi

echo "[StageC] DONE shards: ${#PRED_FILES[@]}"
echo "[StageC] Merging -> ${MERGED_H5}"

# ------------------------------------------------------------
# 2) Merge to a single HDF5 (streaming)
# ------------------------------------------------------------
python3 "${MERGE_SCRIPT}" --out "${MERGED_H5}" "${PRED_FILES[@]}"

# ------------------------------------------------------------
# 3) Export TSV + stats from merged HDF5
# ------------------------------------------------------------
python3 - <<'PY' "${MERGED_H5}" "${OUT_TSV}" "${OUT_STATS}"
import gzip, json, re, sys
from pathlib import Path
import h5py

merged_h5 = Path(sys.argv[1])
out_tsv   = Path(sys.argv[2])
out_stats = Path(sys.argv[3])

_model_suffix_re = re.compile(r"_model_\d+$")
def clean_mol_id(m: str) -> str:
    return _model_suffix_re.sub("", m)

# P² quantile estimator (O(1) memory)
class P2Quantile:
    def __init__(self, p: float):
        if not (0.0 < p < 1.0): raise ValueError("p must be (0,1)")
        self.p=p; self.n=0; self.initial=[]
        self.q=[0.0]*5; self.np=[0.0]*5; self.ni=[0]*5
        self.dn=[0.0, p/2.0, p, (1.0+p)/2.0, 1.0]
    def add(self,x: float):
        self.n += 1
        if self.n <= 5:
            self.initial.append(x)
            if self.n == 5:
                self.initial.sort()
                self.q = self.initial[:]
                self.ni = [1,2,3,4,5]
                self.np = [1.0, 1.0 + 2.0*self.p, 1.0 + 4.0*self.p, 3.0 + 2.0*self.p, 5.0]
            return
        if x < self.q[0]: self.q[0]=x; k=0
        elif x >= self.q[4]: self.q[4]=x; k=3
        else:
            k=0
            while k < 4 and not (self.q[k] <= x < self.q[k+1]): k += 1
            if k == 4: k=3
        for i in range(5):
            if i >= k+1: self.ni[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]
        for i in (1,2,3):
            d = self.np[i] - self.ni[i]
            if (d >= 1 and self.ni[i+1]-self.ni[i] > 1) or (d <= -1 and self.ni[i-1]-self.ni[i] < -1):
                di = 1 if d >= 1 else -1
                qip = self.q[i] + di * (
                    (self.ni[i] - self.ni[i-1] + di) * (self.q[i+1] - self.q[i]) / (self.ni[i+1] - self.ni[i]) +
                    (self.ni[i+1] - self.ni[i] - di) * (self.q[i] - self.q[i-1]) / (self.ni[i] - self.ni[i-1])
                ) / (self.ni[i+1] - self.ni[i-1])
                if self.q[i-1] < qip < self.q[i+1]:
                    self.q[i] = qip
                else:
                    self.q[i] = self.q[i] + di * (self.q[i+di] - self.q[i]) / (self.ni[i+di] - self.ni[i])
                self.ni[i] += di
    def value(self):
        if self.n == 0: return float("nan")
        if self.n <= 5:
            s=sorted(self.initial)
            idx=int(round((len(s)-1)*self.p))
            return float(s[idx])
        return float(self.q[2])

q10,q25,q50,q75,q90 = P2Quantile(0.10),P2Quantile(0.25),P2Quantile(0.50),P2Quantile(0.75),P2Quantile(0.90)

count=0
sum_x=0.0
min_x=float("inf")
max_x=float("-inf")
high=med=low=0

tmp = out_tsv.with_suffix(out_tsv.suffix + ".tmp")
with h5py.File(merged_h5, "r") as h, gzip.open(tmp, "wt") as out:
    g = h["epoch_0000"]["pred"]
    mol = g["mol"]
    outs = g["outputs"]
    n = len(outs)

    out.write("pdb_id\tdockq\n")

    # stream chunks
    step = 200_000
    for i in range(0, n, step):
        j = min(n, i + step)
        mol_chunk = mol[i:j]
        out_chunk = outs[i:j]
        for m, o in zip(mol_chunk, out_chunk):
            m = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
            m = clean_mol_id(m)
            x = float(o)
            out.write(f"{m}\t{x}\n")

            count += 1
            sum_x += x
            if x < min_x: min_x = x
            if x > max_x: max_x = x

            if x >= 0.49: high += 1
            elif x >= 0.23: med += 1
            else: low += 1

            q10.add(x); q25.add(x); q50.add(x); q75.add(x); q90.add(x)

tmp.replace(out_tsv)

if count == 0:
    raise SystemExit("ERROR: wrote 0 rows")

stats = dict(
    merged_h5=str(merged_h5),
    rows=count,
    mean=(sum_x / count),
    median=q50.value(),
    min=min_x,
    max=max_x,
    p10=q10.value(),
    p25=q25.value(),
    p75=q75.value(),
    p90=q90.value(),
    high_ge_0_49=high,
    med_0_23_0_49=med,
    low_lt_0_23=low,
)
out_stats.write_text(json.dumps(stats, indent=2) + "\n")

print(f"✓ Wrote: {out_tsv}")
print(f"✓ Stats: {out_stats}")
PY

# Copy predictions TSV and stats to run root
cp "${OUT_TSV}" "${RUN_ROOT}/predictions.tsv.gz"
cp "${OUT_STATS}" "${RUN_ROOT}/stats.json"
echo "[StageC] ✓ predictions.tsv.gz -> ${RUN_ROOT}/"
echo "[StageC] ✓ stats.json -> ${RUN_ROOT}/"

# ------------------------------------------------------------
# 4) Compute metrics: summarize, plot, copy to run root
# ------------------------------------------------------------
METRICS_DIR="${RUN_ROOT}/compute_metrics"
SUMMARIZE="${DEEPRANK_ROOT}/scripts/summarize_compute_metrics.py"
PLOT_TS="${DEEPRANK_ROOT}/scripts/plot_compute_metrics_timeseries.py"
PLOT_SUM="${DEEPRANK_ROOT}/scripts/plot_compute_metrics_summary.py"

if [[ -d "${METRICS_DIR}" ]]; then
  echo "[StageC] Running compute-metrics analysis..."

  # Step 1: summarize (produces summary_*_ALL.tsv/json in METRICS_DIR)
  if python3 "${SUMMARIZE}" "${METRICS_DIR}"; then
    echo "[StageC] ✓ Metrics summarized"

    # Find the aggregate TSV for the summary plot
    AGG_TSV=$(ls -1t "${METRICS_DIR}"/summary_*_ALL.tsv 2>/dev/null | head -1 || true)

    # Step 2: summary bar-chart PDF
    if [[ -n "${AGG_TSV}" ]]; then
      SUMMARY_PDF="${RUN_ROOT}/compute_metrics_summary.pdf"
      if python3 "${PLOT_SUM}" "${AGG_TSV}" --out "${SUMMARY_PDF}"; then
        echo "[StageC] ✓ Summary plot: ${SUMMARY_PDF}"
      else
        echo "[StageC] WARNING: summary plot failed (non-fatal)"
      fi
    fi

    # Step 3: time-series PDF
    TIMESERIES_PDF="${RUN_ROOT}/compute_metrics_timeseries.pdf"
    if python3 "${PLOT_TS}" "${METRICS_DIR}" --out "${TIMESERIES_PDF}"; then
      echo "[StageC] ✓ Time-series plot: ${TIMESERIES_PDF}"
    else
      echo "[StageC] WARNING: time-series plot failed (non-fatal)"
    fi

    # Copy aggregate tables to run root with clean names
    AGG_JSON=$(ls -1t "${METRICS_DIR}"/summary_*_ALL.json 2>/dev/null | head -1 || true)
    if [[ -n "${AGG_TSV}" ]]; then
      cp "${AGG_TSV}" "${RUN_ROOT}/compute_metrics.tsv"
      echo "[StageC] ✓ compute_metrics.tsv"
    fi
    if [[ -n "${AGG_JSON}" ]]; then
      cp "${AGG_JSON}" "${RUN_ROOT}/compute_metrics.json"
      echo "[StageC] ✓ compute_metrics.json"
    fi
  else
    echo "[StageC] WARNING: metrics summarization failed (non-fatal)"
  fi
else
  echo "[StageC] No compute_metrics directory found; skipping metrics analysis"
fi

echo "[StageC] Done."

