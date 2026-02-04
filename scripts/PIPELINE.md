# DeepRank-Ab SLURM Pipeline

Three-stage pipeline for large-scale antibody-antigen binding affinity prediction.

## Overview

| Stage | Script | Resources | Purpose |
|-------|--------|-----------|---------|
| A | `drab-A.slurm` | CPU (32 cores, 64G) | Feature extraction, graph generation |
| B | `drab-B.slurm` | GPU (4×L40s, 128G) | ESM embeddings + model inference |
| C | `drab-C.slurm` | CPU (4 cores, 16G) | Merge predictions, export results |

## Quick Start

```bash
# Set required environment variables
export RUN_ROOT=/path/to/run_dir          # Output directory (will contain exchange/)
export PDB_ROOT=/path/to/pdbs             # Input PDB directory
export DEEPRANK_ROOT=/path/to/DeepRank-Ab # Repository root
export MODEL_PATH=/path/to/model.pt       # Trained model (for StageB)

# Submit StageA
sbatch scripts/drab-A.slurm

# When StageA completes, submit StageB
scripts/submit_stageB.sh scripts/drab-B.slurm

# When StageB completes, submit StageC
sbatch scripts/drab-C.slurm
```

## Directory Structure

After running, `RUN_ROOT` contains:

```
RUN_ROOT/
├── exchange/
│   ├── shard_lists/          # StageA: PDB lists per shard
│   │   ├── shard_000000.lst
│   │   ├── shard_000001.lst
│   │   └── ...
│   ├── shards/               # StageA: processed shards
│   │   ├── shard_000000/
│   │   │   ├── graphs.h5
│   │   │   ├── meta_stageA.json
│   │   │   └── STAGEA_DONE
│   │   └── ...
│   ├── preds/                # StageB: predictions
│   │   ├── pred_shard_000000.h5
│   │   ├── DONE_shard_000000.ok
│   │   └── ...
│   ├── summary/              # StageC: final outputs
│   │   ├── predictions_merged.h5
│   │   ├── all_predictions.tsv.gz
│   │   └── stats.json
│   └── compute_metrics/      # Optional performance metrics
└── ...
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

### Array Sizing

Adjust `--array=0-N%M` in the SLURM header:
- `N`: Number of shards minus 1 (or use a large number; excess tasks exit cleanly)
- `M`: Maximum concurrent tasks

Task 0 builds the shard lists; other tasks wait for completion before processing.

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

### Submitting

Use `submit_stageB.sh` to auto-calculate array size:

```bash
export RUN_ROOT=/path/to/run
export DEEPRANK_ROOT=/path/to/DeepRank-Ab
export MODEL_PATH=/path/to/model.pt

# Optional: tune shards per task
export SHARDS_PER_TASK=20
export MAX_CONCURRENT=8

scripts/submit_stageB.sh scripts/drab-B.slurm
```

## Stage C: Consolidation

Merges per-shard predictions into final outputs.

### Outputs

- `predictions_merged.h5`: Combined HDF5 with all predictions
- `all_predictions.tsv.gz`: Tab-separated export (pdb_id, dockq)
- `stats.json`: Summary statistics (count, mean, median, percentiles, quality bins)

### Quality Bins

| Category | DockQ Range |
|----------|-------------|
| High | >= 0.49 |
| Medium | 0.23 - 0.49 |
| Low | < 0.23 |

---

# Progress Monitoring

## stageA_progress.sh

Monitor StageA shard processing.

```bash
# One-shot status
scripts/stageA_progress.sh --run-root /path/to/run

# With Slurm job info
scripts/stageA_progress.sh --run-root /path/to/run --job 12345678

# Live watch mode
scripts/stageA_progress.sh --run-root /path/to/run --job 12345678 --watch 5
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-root` | (required) | Path to RUN_ROOT |
| `--job` | | Slurm array job ID for status summary |
| `--watch N` | | Refresh every N seconds |
| `--top N` | `10` | Show N most recent completions |
| `--rate-n N` | `20` | Use last N completions for ETA |
| `--no-slurm` | | Disable Slurm queries |

### Output

- Shard completion count and percentage
- ETA based on recent completion rate
- PDB processed/remaining counts
- Prep ok/fail statistics
- Recent STAGEA_DONE timestamps
- Incomplete shard IDs
- Slurm job state summary (if `--job` provided)

## stageB_progress.sh

Monitor StageB GPU inference.

```bash
# One-shot status
scripts/stageB_progress.sh --run-root /path/to/run

# Live watch mode
scripts/stageB_progress.sh --run-root /path/to/run --job 12345678 --watch 5
```

Same options as `stageA_progress.sh`.

### Features

- No HDF5 reads (pure filesystem, no h5py dependency)
- Cached shard line counts for fast refresh
- Nanosecond mtime precision for accurate ETA
- Shows StageA-ready vs StageB-done counts

## watch_progress.sh

Generic wrapper for live monitoring.

```bash
# Watch StageA
scripts/watch_progress.sh --run-root /path/to/run --stage A --job 12345678

# Watch StageB
scripts/watch_progress.sh --run-root /path/to/run --stage B --job 12345678

# Custom interval
scripts/watch_progress.sh --run-root /path/to/run --stage A --interval 10
```

## stageA_timing_breakdown.sh

Aggregate timing statistics from completed shards.

```bash
scripts/stageA_timing_breakdown.sh /path/to/run
```

Output:
```
shards:      150
total_s:     12345.6
prep_s:       1234.5 ( 10.0%)
annotate_s:   4567.8 ( 37.0%)
graphs_s:     5432.1 ( 44.0%)
cluster_s:    1111.2 (  9.0%)
```

---

# Compute Metrics

Both StageA and StageB collect optional performance metrics.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECT_COMPUTE_METRICS` | `1` | Enable/disable collection |
| `COMPUTE_METRICS_INTERVAL` | `2` | Sampling interval (seconds) |

Metrics are saved to `RUN_ROOT/exchange/compute_metrics/`.

### Analysis Scripts

```bash
# Summarize metrics
python scripts/summarize_compute_metrics.py /path/to/run/exchange/compute_metrics

# Plot time series
python scripts/plot_compute_metrics_timeseries.py /path/to/run/exchange/compute_metrics

# Plot summary
python scripts/plot_compute_metrics_summary.py /path/to/run/exchange/compute_metrics
```
