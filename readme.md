# DeepRank-Ab

DeepRank-Ab is a scoring function for ranking antibody-antigen docking
models based on geometric deep learning.

📄 **Publication**\
https://www.biorxiv.org/content/10.64898/2025.12.03.691974v1

------------------------------------------------------------------------

## Overview

DeepRank-Ab processes antibody-antigen PDB structures through three stages:

1. **Stage A** (CPU) — Graph generation: split ensemble models, annotate
   CDRs via ANARCI, compute Voronota tessellation features, build
   atom-level graphs, cluster with MCL
2. **Stage B** (GPU) — Inference: compute ESM-2 embeddings, run EGNN
   model, predict DockQ scores per structure
3. **Stage C** (CPU) — Merge: concatenate per-shard HDF5 predictions,
   export TSV + stats

## Repository Structure

```
DeepRank-Ab/
├── scripts/
│   ├── split_stageA_cpu.py    # Stage A worker (graph generation)
│   ├── split_stageB_gpu.py    # Stage B worker (ESM + EGNN inference)
│   └── merge_pred_hdf5.py     # Stage C merge (concatenate pred HDF5s)
├── src/                       # core library
│   ├── GraphGenMP.py          # parallel graph generation
│   ├── DataSet.py             # HDF5 dataset + PreCluster
│   ├── NeuralNet_focal_EMA.py # neural net with EMA
│   ├── EGNN.py                # equivariant GNN model
│   ├── tools/
│   │   ├── annotate.py        # ANARCI CDR annotation
│   │   └── voronota/          # voronota binary (tessellation)
│   └── weights/               # pretrained model weights
├── environment-gpu.yml        # conda environment
└── scripts/deprecated/        # legacy pipeline scripts
```

------------------------------------------------------------------------

## Scripts

### split_stageA_cpu.py — Stage A (CPU graph generation)

Processes a shard of PDB files: splits multi-model ensembles, annotates
CDR loops via ANARCI, computes Voronota tessellation features, builds
atom-level HDF5 graphs, and runs MCL clustering.

**Imports from `src/`:** `GraphGenMP.GraphHDF5`, `DataSet.HDF5DataSet`,
`DataSet.PreCluster`, `tools.annotate.annotate_folder_one_by_one_mp`

**Key arguments:**
```
--pdb-root DIR          Input PDB directory
--run-root DIR          Run output directory
--shard-id ID           Shard to process (e.g. 000000)
--heavy CHAIN           Heavy chain ID (e.g. H)
--light CHAIN           Light chain ID (- for nanobody)
--antigen CHAIN         Antigen chain ID (e.g. T)
--num-cores N           CPU cores (default: $SLURM_CPUS_PER_TASK)
--make-shards           Create shard lists and exit
--prep-graphs-only      Run graph generation only (skip clustering)
--cluster-only          Run clustering only (requires existing graphs)
```

**Sharding mode** (`--make-shards`): scans `--pdb-root`, creates
`shard_lists/shard_NNNNNN.lst` files based on `--target-shard-gb`,
`--min-per-shard`, `--max-per-shard`.

**Processing mode** (`--shard-id`): processes one shard's PDB list
through the full Stage A pipeline (or a subset with `--prep-graphs-only`
/ `--cluster-only` for split-mode execution).

### split_stageB_gpu.py — Stage B (GPU inference)

Loads ESM-2 embeddings and runs EGNN inference on graph HDF5 files
produced by Stage A. Processes a range of shards per task.

**Imports from `src/`:** `NeuralNet_focal_EMA.NeuralNet`, `EGNN.egnn`

**Key arguments:**
```
--run-root DIR          Run directory (contains shards/)
--start-index N         First shard index to process (0-based)
--count N               Number of shards to process
--model-path PATH       Pretrained model weights (.pth.tar)
--device DEVICE         cuda or cpu (default: cuda)
--num-cores N           CPU cores (default: $NUM_CORES)
--batch-size N          Inference batch size (default: 64)
--dl-workers N          DataLoader workers (default: 8)
--esm-gpus N            GPUs for ESM embedding (default: 4)
--esm-toks-per-batch N  ESM tokens per batch (default: 12288)
```

Outputs per-shard prediction HDF5 files in `preds/` and writes
`DONE_shard_NNNNNN.ok` sentinels.

### merge_pred_hdf5.py — Stage C merge

Concatenates per-shard prediction HDF5 files into a single merged file.
Pure h5py — no `src/` imports.

```
python3 scripts/merge_pred_hdf5.py --out merged.h5 pred_shard_*.h5
```

------------------------------------------------------------------------

## Integration with NanobodyDesigner

DeepRank-Ab is designed to be called from
[NanobodyDesigner](https://github.com/evalkov/NanobodyDesigner), which
owns all SLURM orchestration scripts. NanobodyDesigner's pipeline:

1. Sets `DEEPRANK_ROOT` to this repo's path
2. Generates wrapper scripts that source env vars and exec the SLURM
   scripts in `NanobodyDesigner/slurm/deeprank_{a,b,c}.slurm`
3. Those SLURM scripts call `${DEEPRANK_ROOT}/scripts/split_stageA_cpu.py`,
   `split_stageB_gpu.py`, and `merge_pred_hdf5.py`

The `deeprank_root` config key in `pipeline_config.yaml` points to this
repo. See NanobodyDesigner's README for full pipeline configuration.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/evalkov/DeepRank-Ab
cd DeepRank-Ab
```

### 2. Create and activate the environment

```bash
mamba env create -f environment-gpu.yml
mamba activate deeprank-ab
```

### 3. Install ANARCI

ANARCI is required for CDR annotation.

Installation instructions:\
https://github.com/oxpig/ANARCI

*Note:* `hmmscan` is already included in the environment.\
If you encounter issues, follow the workaround here:\
https://github.com/oxpig/ANARCI/issues/102

------------------------------------------------------------------------

## Input Requirements

-   **PDB directory** — folder with antibody-antigen PDBs
-   **Chain IDs** — heavy, light (or `-` for nanobody), and antigen
-   **Model weights** — pretrained `.pth.tar` file (in `src/weights/`)
-   **Voronota binary** — included in `src/tools/voronota/`

------------------------------------------------------------------------

## License

See [LICENSE](LICENSE) for details.

## Support

For issues or questions, please open a GitHub issue.
