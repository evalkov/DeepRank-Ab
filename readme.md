# DeepRank-Ab

DeepRank-Ab is a scoring function for ranking antibody-antigen docking
models based on geometric deep learning. It takes antibody-antigen
complex structures as input and predicts DockQ scores using an
equivariant graph neural network (EGNN) augmented with ESM-2 protein
language model embeddings.

**Publication:**
https://www.biorxiv.org/content/10.64898/2025.12.03.691974v1

------------------------------------------------------------------------

## Repository Structure

```
DeepRank-Ab/
├── scripts/
│   ├── split_stageA_cpu.py     # Stage A: PDB/quiver -> graphs (CPU)
│   ├── split_stageB_gpu.py     # Stage B: ESM embeddings + EGNN inference (GPU)
│   ├── merge_pred_hdf5.py      # Stage C: merge per-shard predictions
│   ├── inference.py            # single-structure end-to-end inference
│   └── deprecated/             # legacy orchestration & monitoring scripts
├── src/
│   ├── EGNN.py                 # equivariant GNN model (7-block, dual-branch)
│   ├── NeuralNet_focal_EMA.py  # neural net wrapper with EMA
│   ├── GraphGenMP.py           # parallel graph generation (producer-consumer)
│   ├── AtomGraph.py            # atom-level graph builder
│   ├── ResidueGraph.py         # residue-level graph builder
│   ├── Graph.py                # base class: NetworkX <-> HDF5 conversion
│   ├── DataSet.py              # HDF5 dataset loader + MCL pre-clustering
│   ├── Metrics.py              # evaluation metrics
│   ├── tools/
│   │   ├── annotate.py         # ANARCI CDR region annotation
│   │   ├── quiver.py           # .qv quiver archive reader
│   │   ├── BSA.py              # buried surface area (freesasa)
│   │   ├── VoroArea.py         # Voronota tessellation contact areas
│   │   ├── edge_orientation.py # local-frame edge orientation features
│   │   ├── BioWrappers.py      # Biopython utility functions
│   │   └── voronota/           # voronota binary
│   └── weights/                # pretrained model weights
├── docs/                       # additional documentation
├── environment-gpu.yml         # conda environment specification
└── LICENSE                     # Apache-2.0
```

------------------------------------------------------------------------

## Pipeline Overview

DeepRank-Ab processes structures through three stages, designed for
distributed execution on HPC clusters via SLURM:

```
Input (PDB files or .qv quiver archive)
  |
  v
Stage A (CPU) -- graph generation
  |  split ensembles, merge chains, annotate CDRs (ANARCI),
  |  compute features (Voronota, BSA, orientation, contacts),
  |  build atom-level graphs, MCL clustering
  |  -> graphs.h5, manifest.tsv.gz per shard
  v
Stage B (GPU) -- embedding + inference
  |  batch ESM-2 embeddings across all shards,
  |  inject into graphs, run EGNN model
  |  -> pred_shard_NNNNNN.h5 per shard
  v
Stage C (CPU) -- merge
  |  concatenate per-shard prediction HDF5 files
  |  -> predictions_merged.h5
  v
Output: predicted DockQ score per structure
```

------------------------------------------------------------------------

## Input Formats

### PDB directory

A folder of individual `.pdb` files, each containing an antibody-antigen
complex. The user specifies which chain IDs correspond to heavy chain,
light chain (or `-` for nanobodies/VHH), and antigen.

```bash
python scripts/split_stageA_cpu.py \
  --pdb-root /path/to/pdbs \
  --run-root /scratch/run_001 \
  --make-shards \
  --heavy H --light L --antigen A
```

### Quiver archive (.qv)

A single concatenated text file containing multiple PDB structures
produced by RFdiffusion/Rosetta pipelines, with a companion `.qv.idx`
byte-offset index for random access. Each entry is preceded by `QV_TAG`
and `QV_SCORE` header lines and may include a Rosetta energy table.

During extraction, hydrogens, OXT atoms, Rosetta energy tables, and
quiver metadata lines are stripped automatically.

```bash
python scripts/split_stageA_cpu.py \
  --quiver /path/to/designs.qv \
  --run-root /scratch/run_001 \
  --make-shards \
  --heavy H --antigen T
```

`--pdb-root` and `--quiver` are mutually exclusive. The index file is
inferred as `<quiver_path>.idx`.

### Single-structure inference

For one-off predictions without sharding:

```bash
python scripts/inference.py \
  --pdb complex.pdb \
  --heavy H --light - --antigen T
```

------------------------------------------------------------------------

## Output

### Per-shard artifacts (Stage A)

```
run_root/shards/shard_NNNNNN/
  graphs.h5           # HDF5 atom-level graphs with features
  manifest.tsv.gz     # sequences per chain (for ESM)
  meta_stageA.json    # timing, counts, chain config
  STAGEA_DONE         # completion sentinel
```

### Graph HDF5 schema (`graphs.h5`)

Each molecule is a top-level group:

```
/<mol_name>/
  nodes                     # node keys (string array)
  edge_index                # interface edge indices
  internal_edge_index       # intra-chain edge indices
  node_data/
    chain                   # one-hot chain assignment
    pos                     # 3D coordinates
    res_type                # one-hot amino acid (20 classes)
    charge                  # scalar
    polarity                # one-hot (apolar/polar/neg/pos)
    bsa                     # buried surface area
    atom_type               # one-hot (~36 atom types)
    region                  # one-hot CDR/FR/CONST/AG
    embedding               # ESM-2 embeddings (filled in Stage B)
  edge_data/
    dist                    # scalar distance
    voro_area               # Voronota tessellation contact area
    covalent                # covalent bond feature
    vdw                     # van der Waals feature
    orientation             # local-frame orientation vector
    type                    # edge type (interface/internal)
  internal_edge_data/       # same features for intra-chain edges
  clustering/mcl/           # hierarchical MCL cluster assignments
```

### Predictions (Stage B)

```
run_root/preds/pred_shard_NNNNNN.h5
  /epoch_0000/pred/
    mol                     # molecule names (string array)
    outputs                 # predicted DockQ scores (float32)
```

### Merged output (Stage C)

```
run_root/summary/
  predictions_merged.h5     # all shards concatenated
  all_predictions.tsv.gz    # tabular export
  stats.json                # summary statistics
```

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/evalkov/DeepRank-Ab
cd DeepRank-Ab
```

### 2. Create the conda environment

```bash
mamba env create -f environment-gpu.yml
mamba activate deeprank-ab
```

Key dependencies: Python 3.9, PyTorch 2.0 (CUDA 11.8), PyTorch Geometric
2.3, Biopython 1.85, fair-esm 2.0, pdb2sql 0.5, freesasa 2.2, h5py 3.7.

### 3. Install ANARCI

ANARCI is required for CDR annotation. `hmmscan` is included in the
conda environment.

Installation: https://github.com/oxpig/ANARCI

If you encounter issues: https://github.com/oxpig/ANARCI/issues/102

### 4. Verify voronota

The voronota binary is included at `src/tools/voronota/voronota`. On
Linux x86_64, it should work out of the box. On other platforms you may
need to rebuild it from https://github.com/kliment-olechnovic/voronota.

------------------------------------------------------------------------

## Requirements Summary

| Component | Purpose | Source |
|-----------|---------|--------|
| Python 3.9 | Runtime | conda |
| PyTorch 2.0 + CUDA | Model inference | conda |
| PyTorch Geometric | Graph data handling | conda |
| Biopython | PDB parsing | conda |
| fair-esm | ESM-2 embeddings | pip |
| pdb2sql | PDB structure queries | pip |
| freesasa | Buried surface area | pip |
| h5py | HDF5 I/O | conda |
| ANARCI | CDR numbering | manual |
| voronota | Tessellation features | bundled |

------------------------------------------------------------------------

## Scripts Reference

### split_stageA_cpu.py

```
--pdb-root DIR | --quiver PATH   Input source (mutually exclusive, one required)
--run-root DIR                   Output directory
--make-shards                    Create shard lists and exit
--shard-id ID                    Process a specific shard (e.g. 000000)
--heavy CHAIN                    Heavy chain ID (required)
--light CHAIN                    Light chain ID (default: -, meaning nanobody)
--antigen CHAIN                  Antigen chain ID (required)
--num-cores N                    CPU cores (default: 32)
--target-shard-gb N              Target shard size in GB (default: 5.0)
--min-per-shard N                Minimum structures per shard (default: 200)
--max-per-shard N                Maximum structures per shard (default: 1200)
--glob PATTERN                   PDB glob pattern (default: *.pdb, pdb-root only)
--prep-graphs-only               Run through graph generation, skip clustering
--cluster-only                   Run clustering only (requires existing graphs.h5)
--no-cluster                     Skip MCL clustering entirely
```

### split_stageB_gpu.py

```
--run-root DIR                   Run directory (contains shards/)
--start-index N                  First shard index (0-based)
--count N                        Number of shards to process
--model-path PATH                Pretrained model weights (.pth.tar)
--device DEVICE                  cuda or cpu (default: cuda)
--batch-size N                   Inference batch size (default: 64)
--esm-gpus N                     GPUs for ESM embedding (default: 4)
--esm-toks-per-batch N           ESM tokens per batch (default: 12288)
```

### merge_pred_hdf5.py

```bash
python scripts/merge_pred_hdf5.py --out merged.h5 pred_shard_*.h5
```

### inference.py

Single-structure end-to-end inference (no sharding needed):

```bash
python scripts/inference.py --pdb complex.pdb --heavy H --light - --antigen T
```

------------------------------------------------------------------------

## Integration with NanobodyDesigner

DeepRank-Ab is designed for integration with
[NanobodyDesigner](https://github.com/evalkov/NanobodyDesigner) to score
100K-1M nanobody-antigen complexes. NanobodyDesigner owns SLURM
orchestration:

1. Sets `DEEPRANK_ROOT` to this repo
2. Calls `scripts/split_stageA_cpu.py`, `split_stageB_gpu.py`, and
   `merge_pred_hdf5.py` from its SLURM scripts
3. The `deeprank_root` key in `pipeline_config.yaml` points here

See the NanobodyDesigner README for full pipeline configuration.

------------------------------------------------------------------------

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Bundled dependencies:
- **voronota** — MIT License (Kliment Olechnovic)
- **ANARCI** — BSD-3-Clause (Charlotte Deane, James Dunbar, et al.)
