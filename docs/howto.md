# DeepRank-Ab Pipeline How-To

## Quick Start

```bash
# 1. Copy the example config
cp scripts/pipeline.yaml.example my_run.yaml

# 2. Edit the config with your paths
vim my_run.yaml

# 3. Analyze input and see resource estimates
python scripts/run_pipeline.py my_run.yaml --analyze

# 4. Dry run to preview job submissions
python scripts/run_pipeline.py my_run.yaml --dry-run

# 5. Run the full pipeline
python scripts/run_pipeline.py my_run.yaml
```

## Dynamic Resource Allocation

The pipeline **automatically sizes job arrays** based on your input data:

1. **Analyzes input PDBs** - counts files, calculates sizes
2. **Estimates optimal shards** - based on `target_shard_gb` setting
3. **Sizes arrays dynamically** - no manual `--array` specification needed
4. **Respects concurrency limits** - `max_concurrent_a` and `max_concurrent_b`

```bash
# See what the pipeline will do before running
python scripts/run_pipeline.py my_run.yaml --analyze

# Output:
# PDBs: 5,000 files (2.50 GB)
# Estimated shards: 25
# Recommended cores: 32
# Stage A array: 0-24%20
# Stage B array: 0-24%10
```

---

## Configuration

### Required Settings

```yaml
# Paths (required)
run_root: /path/to/run_001           # Output directory (created automatically)
pdb_root: /path/to/input_pdbs        # Directory containing input PDB files
deeprank_root: /path/to/DeepRank-Ab  # DeepRank-Ab repository root
model_path: /path/to/model.pth       # Trained model weights

# Chain identifiers (required)
chains:
  heavy: H          # Heavy chain ID in your PDBs
  light: "-"        # Light chain ID, use "-" for VHH/nanobodies
  antigen: T        # Antigen chain ID
```

### Stage A Settings (CPU - Graph Generation)

```yaml
stage_a:
  partition: norm           # SLURM partition
  cores: 32                 # CPUs per task
  mem_gb: 64                # Memory in GB
  time: "01:00:00"          # Wall time
  array: "0-9%10"           # SLURM array (adjust based on number of shards)

  # Sharding (controls how PDBs are grouped)
  target_shard_gb: 0.1      # Target shard size in GB
  min_per_shard: 10         # Minimum PDBs per shard
  max_per_shard: 100        # Maximum PDBs per shard
  glob: "**/*.pdb"          # Glob pattern to find PDBs

  # Voronota settings
  voronota_binary: voronota_129   # voronota | voronota_129 | voronota-lt
  use_freesasa: 1                 # 1 = freesasa, 0 = voronota-lt for BSA
  voro_omp_threads: 1             # OpenMP threads per voronota call
```

### Stage B Settings (GPU - ESM Embeddings + Inference)

```yaml
stage_b:
  partition: gpu            # SLURM partition (must have GPUs)
  gpus: 4                   # GPUs per task
  cores: 32                 # CPUs per task
  mem_gb: 128               # Memory in GB
  time: "04:00:00"          # Wall time
  array: "0-9%5"            # SLURM array

  # ESM settings
  esm_model: esm2_t33_650M_UR50D
  esm_toks_per_batch: 12288
  esm_scalar_dtype: float16

  # Inference settings
  batch_size: 64
  dl_workers: 8
  prefetch_factor: 4
```

### Stage C Settings (CPU - Merge Results)

```yaml
stage_c:
  partition: norm
  cores: 4
  mem_gb: 16
  time: "02:00:00"
```

### Optional Settings

```yaml
collect_metrics: true       # Enable compute metrics sampling
metrics_interval: 2         # Sampling interval in seconds
```

---

## Running the Pipeline

### Full Pipeline (Recommended)

```bash
python scripts/run_pipeline.py my_run.yaml
```

This submits all three stages with SLURM dependencies:
- Stage A runs first
- Stage B starts after A completes successfully
- Stage C starts after B completes successfully

### Dry Run (Preview)

```bash
python scripts/run_pipeline.py my_run.yaml --dry-run
```

Shows what would be submitted without actually submitting jobs.

### Individual Stages

```bash
# Run only Stage A
python scripts/run_pipeline.py my_run.yaml --stage a

# Run only Stage B (assumes A is complete)
python scripts/run_pipeline.py my_run.yaml --stage b

# Run only Stage C (assumes B is complete)
python scripts/run_pipeline.py my_run.yaml --stage c
```

---

## Monitoring Progress

### Check Job Status

```bash
squeue -u $USER
```

### Watch Pipeline Progress

```bash
# Stage A progress
./scripts/stageA_progress.sh /path/to/run_root

# Stage B progress
./scripts/stageB_progress.sh /path/to/run_root

# Combined watcher
./scripts/watch_progress.sh /path/to/run_root
```

### View Logs

```bash
# SLURM logs are in the submission directory
ls -la drab-A_*.log drab-B_*.log drab-C_*.log

# Per-shard metadata
cat /path/to/run_root/shards/shard_000000/meta_stageA.json
```

---

## Output Structure

After the pipeline completes:

```
run_root/
├── shard_lists/              # Stage A shard definitions
│   ├── shard_000000.lst
│   └── ...
├── shards/                   # Stage A output (per-shard)
│   ├── shard_000000/
│   │   ├── graphs.h5
│   │   ├── manifest.tsv.gz
│   │   ├── meta_stageA.json
│   │   └── STAGEA_DONE
│   └── ...
├── preds/                    # Stage B output (predictions)
│   ├── pred_shard_000000.h5
│   ├── DONE_shard_000000.ok
│   └── ...
├── summary/                  # Stage C output (merged)
│   ├── predictions_merged.h5
│   ├── all_predictions.tsv.gz
│   └── stats.json
├── compute_metrics/          # Resource usage metrics (if enabled)
├── logs/                     # SLURM job logs
├── compute_metrics_summary.pdf
├── compute_metrics_timeseries.pdf
├── compute_metrics.tsv
├── compute_metrics.json
└── pipeline_config.yaml      # Copy of config used
```

---

## Common Issues

### "No PDBs found"

Check your `glob` pattern and `pdb_root`:
```bash
# Test the glob pattern
ls /path/to/pdb_root/**/*.pdb | head
```

### Stage B fails with "missing STAGEA_DONE"

Stage A didn't complete successfully. Check:
```bash
ls /path/to/run_root/shards/*/STAGEA_DONE | wc -l
```

### Out of memory

Reduce parallelism or increase memory:
```yaml
stage_a:
  cores: 16        # Fewer workers = less memory
  mem_gb: 128      # More memory per task
```

### Voronota slow

Try enabling OpenMP (for voronota 1.29):
```yaml
stage_a:
  cores: 16              # Reduce Python workers
  voro_omp_threads: 2    # Enable voronota OpenMP
```

---

## Performance Tuning

### Stage A (CPU)

| Workload | cores | voro_omp_threads | Notes |
|----------|-------|------------------|-------|
| Many small PDBs | 32 | 1 | More parallelism |
| Few large PDBs | 8 | 4 | More threads per voronota |
| Memory constrained | 16 | 1 | Reduce workers |

### Stage B (GPU)

| Setting | Effect |
|---------|--------|
| `gpus: 4` | More GPUs = faster ESM embeddings |
| `batch_size: 128` | Larger batches = better GPU utilization |
| `prefetch_factor: 4` | More prefetching = less GPU idle time |

---

## Example Configs

### Small test run (< 100 PDBs)

```yaml
stage_a:
  array: "0-0"          # Single task
  min_per_shard: 1
  max_per_shard: 100

stage_b:
  array: "0-0"          # Single task
  gpus: 1
```

### Large production run (10,000+ PDBs)

```yaml
stage_a:
  array: "0-99%20"      # 100 tasks, 20 concurrent
  target_shard_gb: 1.0
  min_per_shard: 50
  max_per_shard: 200

stage_b:
  array: "0-99%10"      # 100 tasks, 10 concurrent
  gpus: 4
  time: "08:00:00"
```
