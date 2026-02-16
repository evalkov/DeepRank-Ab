# DeepRank-Ab SLURM Pipeline

Three-stage pipeline for large-scale antibody-antigen binding affinity prediction.

## Script Map

```
                              config.yaml
                                  |
                          run_pipeline.py
                         /       |       \
                   sbatch      sbatch     sbatch
                  (array)     (array)    (single)
                   /             |            \
  .-----------.  .-----------.  .-----------.  .-----------.
  | STAGE A   |  | STAGE A   |  | STAGE B   |  | STAGE C   |
  | drab-A    |  | drab-A    |  | drab-B    |  | drab-C    |
  | (shard)   |  | (process) |  |           |  |           |
  '-----------'  '-----------'  '-----------'  '-----------'
       |              |              |              |
       v              v              v              |
 split_stageA   split_stageA   split_stageB        |
  _cpu.py        _cpu.py        _gpu.py            |
  --make-        --shard-id     (ESM2 +            |
    shards                       inference)        |
       |              |              |              v
       |         collect_       collect_       merge_pred
       |         compute_       compute_        _hdf5.py
       |         metrics.py     metrics.py         |
       |         (background)   (background)       v
       |              |              |         inline Python
       |              v              v         (TSV + stats)
       |         compute_       compute_           |
       |         metrics/       metrics/           v
       |              \             /       summarize_compute
       |               \           /          _metrics.py
       |                \         /                |
       |                 v       v           .-----+------.
       |                 RUN_ROOT/           |            |
       v                compute_metrics/     v            v
  shard_lists/               |          plot_compute  plot_compute
  shards/                    |          _metrics_     _metrics_
  preds/                     |          summary.py    timeseries.py
                             |               |            |
                             v               v            v
                       compute_       *_summary    *_timeseries
                       metrics          .pdf          .pdf
                       .tsv/.json

  MONITORING (user-facing, run in parallel)
  ------------------------------------------
  progress_live.py RUN_ROOT
  (legacy monitors are in scripts/deprecated/)
```

## Overview

| Stage | SLURM script | Resources | Python workhorse | Purpose |
|-------|--------------|-----------|------------------|---------|
| A (shard) | `drab-A.slurm` | CPU | `split_stageA_cpu.py --make-shards` | Partition PDBs into shards |
| A (process) | `drab-A.slurm` | CPU (32 cores, 64G) | `split_stageA_cpu.py --shard-id` | Voronota contacts, graph construction, clustering |
| B | `drab-B.slurm` | GPU (4x, 128G) | `split_stageB_gpu.py` | ESM2 embeddings + model inference |
| C | `drab-C.slurm` | CPU (4 cores, 16G) | `merge_pred_hdf5.py` + inline | Merge predictions, export TSV/stats, metrics analysis |

All SBATCH directives are set by `run_pipeline.py` via CLI args. The `.slurm` scripts
contain no `#SBATCH` headers.

## Quick Start

```bash
# Recommended: use the pipeline launcher with a YAML config
python scripts/run_pipeline.py pipeline.yaml

# Dry run (shows sbatch commands without submitting)
python scripts/run_pipeline.py pipeline.yaml --dry-run

# Run a single stage
python scripts/run_pipeline.py pipeline.yaml --stage a

# Analyze input data and show resource estimates
python scripts/run_pipeline.py pipeline.yaml --analyze
```

## Input Hygiene (Optional)

`run_pipeline.py` can perform a pre-Stage-A input hygiene phase:

1. Run `scripts/check_pdb_preflight.py` on a sample of input PDBs
2. Write reports into `RUN_ROOT`:
   - `preflight_report_pre.json`
   - `preflight_report_pre.tsv`
   - `input_hygiene_summary.json`
3. Optionally submit `scripts/cure_pdbs.sh` and chain Stage A after cure

Enable it in YAML:

```yaml
input_hygiene:
  mode: recommend   # off | recommend | required | auto
```

Mode behavior:
- `off`: skip checker/cure.
- `recommend`: run checker, continue pipeline, record recommendations.
- `required`: run checker and abort if sampled FAIL findings exist.
- `auto`: run checker; if OXT/H prevalence is high, submit cure and run Stage A on cured outputs.

## Directory Structure

After running, `RUN_ROOT` contains:

```
RUN_ROOT/
├── shard_lists/              # Stage A: PDB file lists per shard
│   ├── shard_000000.lst
│   └── ...
├── shards/                   # Stage A: processed graph data
│   ├── shard_000000/
│   │   ├── graphs.h5
│   │   ├── meta_stageA.json
│   │   └── STAGEA_DONE
│   └── ...
├── preds/                    # Stage B: per-shard predictions
│   ├── pred_shard_000000.h5
│   ├── DONE_shard_000000.ok
│   └── ...
├── summary/                  # Stage C: final merged outputs
│   ├── predictions_merged.h5
│   ├── all_predictions.tsv.gz
│   └── stats.json
├── compute_metrics/          # Raw performance metrics (CSV)
├── logs/                     # SLURM job logs (.log / .err)
├── compute_metrics_report.html
├── compute_metrics.tsv
├── compute_metrics.json
└── pipeline_jobs.json        # Job IDs from run_pipeline.py
```

## Stage A: Feature Extraction

CPU-bound stage that processes PDBs into graph representations.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAVY` | `H` | Heavy chain ID |
| `LIGHT` | `-` | Light chain ID (`-` = none/nanobody) |
| `ANTIGEN` | `T` | Antigen chain ID |
| `NUM_CORES` | `$SLURM_CPUS_PER_TASK` | Parallel workers |
| `TARGET_SHARD_GB` | `0.1` | Target shard size in GB |
| `MIN_PER_SHARD` | `10` | Minimum PDBs per shard |
| `MAX_PER_SHARD` | `100` | Maximum PDBs per shard |
| `GLOB_PAT` | `**/*.pdb` | PDB file glob pattern |
| `FORCE_RESHARD` | `0` | Set to `1` to rebuild shard lists |
| `STAGEA_PHASE` | `full` | `full`, `prep_graphs`, or `cluster_only` |
| `STAGEA_STRICT_AFFINITY` | `0` | Fail if effective cpuset is smaller than requested cores |
| `STAGEA_USE_SRUN_BIND` | `1` | Launch payload via `srun --cpu-bind` when available |
| `STAGEA_SRUN_CPU_BIND` | `cores` | Value passed to `srun --cpu-bind` |
| `GRAPH_PIPELINE_TELEMETRY` | `0` | Enable graph writer/enqueue throughput logs |
| `GRAPH_RESULT_QUEUE_MAXSIZE` | `100` | Max buffered graph results between workers/writer |
| `ANARCI_PARALLEL_BATCHES` | `1` | Number of parallel ANARCI batches in annotation phase |

### Array Sizing

`run_pipeline.py` sets `--array=0-N%M` automatically:
- `N`: Number of shards minus 1 (excess tasks exit cleanly)
- `M`: `max_concurrent_a` from YAML config (default: 20)

Task 0 builds the shard lists; other tasks wait for completion before processing.

When `stage_a.split_mode: true`, Stage A is submitted as two dependent arrays:
1. `prep_graphs` phase (copy/prep/annotate/graphs; writes `STAGEA_GRAPHS_DONE`)
2. `cluster_only` phase (MCL + finalize; writes `STAGEA_DONE`)

## Stage B: GPU Inference

GPU-bound stage that computes ESM embeddings and runs model inference.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | (required) | Path to trained model `.pt` file |
| `SHARDS_PER_TASK` | `20` | Shards processed per array task |
| `ESM_GPUS` | `4` | GPUs for ESM embedding |
| `ESM_TOKS_PER_BATCH` | `12288` | Tokens per ESM batch |
| `BATCH_SIZE` | `64` | Inference batch size |
| `DL_WORKERS` | `8` | DataLoader workers |

## Stage C: Consolidation

Merges per-shard predictions and runs metrics analysis.

### Outputs

| File | Description |
|------|-------------|
| `summary/predictions_merged.h5` | Combined HDF5 with all predictions |
| `summary/all_predictions.tsv.gz` | Tab-separated export (pdb_id, dockq) |
| `summary/stats.json` | Summary statistics (count, mean, median, percentiles, quality bins) |
| `compute_metrics_report.html` | Human-readable metrics dashboard (CPU/GPU/disk/network + findings) |
| `compute_metrics.tsv` | Aggregate metrics table |
| `compute_metrics.json` | Aggregate metrics (JSON) |

Set `METRICS_GENERATE_PDFS=1` to additionally produce legacy PDF plots.

### Quality Bins

| Category | DockQ Range |
|----------|-------------|
| High | >= 0.49 |
| Medium | 0.23 - 0.49 |
| Low | < 0.23 |

---

# Progress Monitoring

Use the unified live tracker:

```bash
python3 scripts/progress_live.py /path/to/run
```

Behavior:
- Refreshes every 5 seconds
- Shows Stage A + Stage B as job-level rows
- Shows Stage C phase/status as a concise line
- Shows final pipeline start/end/walltime when complete

Legacy monitors are preserved under `scripts/deprecated/`:
- `scripts/deprecated/watch_progress.sh`
- `scripts/deprecated/stageA_progress.sh`
- `scripts/deprecated/stageB_progress.sh`
- `scripts/deprecated/stageA_timing_breakdown.sh`

---

# Compute Metrics

Both Stage A and Stage B collect optional performance metrics via
`collect_compute_metrics.py` running as a background daemon.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECT_COMPUTE_METRICS` | `1` | Enable/disable collection |
| `COMPUTE_METRICS_INTERVAL` | `2` | Sampling interval (seconds) |

Metrics are saved to `RUN_ROOT/compute_metrics/`.

Stage C automatically runs the analysis scripts and places an HTML report and
aggregate tables in `RUN_ROOT/` (optional legacy PDFs via
`METRICS_GENERATE_PDFS=1`).

### Manual Analysis

```bash
# Summarize raw CSVs into aggregate tables
python scripts/summarize_compute_metrics.py /path/to/run/compute_metrics

# Plot time-series (multi-page PDF, one page per task)
python scripts/plot_compute_metrics_timeseries.py /path/to/run/compute_metrics

# Plot summary bar-charts (single-page PDF)
python scripts/plot_compute_metrics_summary.py /path/to/run/compute_metrics/summary_*_ALL.tsv
```
