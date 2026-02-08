#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-${RUN_ROOT:-}}"
if [[ -z "${RUN_ROOT}" ]]; then
  echo "ERROR: RUN_ROOT not set. Usage: $0 /path/to/run_root  (or export RUN_ROOT=...)" >&2
  exit 1
fi

shopt -s nullglob
files=( "${RUN_ROOT}"/shards/*/meta_stageA.json )

if (( ${#files[@]} == 0 )); then
  echo "ERROR: No meta_stageA.json files found under: ${RUN_ROOT}/shards/*/" >&2
  exit 1
fi

# Aggregate timing breakdown across all shards
# Expects meta_stageA.json fields: .result.prep_s .result.annotate_s .result.graphs_s .result.cluster_s
jq -r '
  .result
  | [
      (.prep_s // 0),
      (.annotate_s // 0),
      (.graphs_s // 0),
      (.cluster_s // 0)
    ]
  | @tsv
' "${files[@]}" \
| awk -F'\t' '
  {
    prep  += $1
    anno  += $2
    graph += $3
    clust += $4
    n++
  }
  END {
    total = prep + anno + graph + clust
    if (n == 0) {
      print "ERROR: no rows parsed" > "/dev/stderr"
      exit 2
    }
    if (total <= 0) {
      printf "shards: %d\n", n
      printf "total_s: %8.1f\n", total
      printf "prep_s: %8.1f\nannotate_s: %8.1f\ngraphs_s: %8.1f\ncluster_s: %8.1f\n", prep, anno, graph, clust
      exit 0
    }
    printf "shards:      %d\n", n
    printf "total_s:     %8.1f\n", total
    printf "prep_s:      %8.1f (%5.1f%%)\n", prep,  prep/total*100
    printf "annotate_s:  %8.1f (%5.1f%%)\n", anno,  anno/total*100
    printf "graphs_s:    %8.1f (%5.1f%%)\n", graph, graph/total*100
    printf "cluster_s:   %8.1f (%5.1f%%)\n", clust, clust/total*100
  }
'

