# DeepRank-Ab Inference Pipeline

DeepRank-Ab is a scoring function for ranking
antibody-antigen docking models based on geometric deep learning.


📄 **Publication**\
https://www.biorxiv.org/content/10.64898/2025.12.03.691974v1

This repository provides the **full inference pipeline** for the model
described in the paper.

------------------------------------------------------------------------

## 🚀 Features

-   **PDB Processing**
    -   Split ensemble models 
    -   Extract chain sequences 
    -   Merge chains for downstream analysis 
-   **FASTA Conversion**
    -   Generate FASTA files for CDR annotation and ESM embeddings 
-   **ESM Embeddings**
    -   Compute embeddings using `esm2_t33_650M_UR50D` 
-   **Graph Construction**
    -   Build atom-level graphs with precomputed node and edge features 
-   **Prediction**
    -   Inference with pretrained EGNN models and output predicted DockQ

------------------------------------------------------------------------

## 📦 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/haddocking/DeepRank-Ab
cd DeepRank-Ab
```

### 2. Create and activate the environment

``` bash
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

## 🔧 Usage

The production SLURM pipeline is executed through:

    DeepRank-Ab/scripts/run_pipeline.py

### **Run the pipeline**

``` bash
cp scripts/pipeline.yaml.example my_run.yaml
python3 scripts/run_pipeline.py my_run.yaml --analyze
python3 scripts/run_pipeline.py my_run.yaml
```

### **Example**

``` bash
python3 scripts/progress_live.py /path/to/run_root
```

This will:

-   Run Stage A (graph generation + clustering)
-   Run Stage B (ESM + inference)
-   Run Stage C (merge/export/metrics report)
-   Save outputs under your configured `run_root`

------------------------------------------------------------------------

## 🧬 Input Requirements

-   **PDB directory**\
    Folder with antibody-antigen PDBs (`glob` configurable in YAML).

-   **Chain IDs**\
    `heavy`, `light` (or `-`), and `antigen` set in YAML.

-   **Model weights path**\
    `model_path` in YAML.

-   **SLURM resources**\
    Defined per stage in YAML.

------------------------------------------------------------------------


## ⚙️ Large-Scale Inference

Use `scripts/run_pipeline.py` for large-scale jobs.

Legacy single-file and older batch helpers are now in:
- `scripts/deprecated/inference.py`
- `scripts/deprecated/large_scale_infer_vhh.py`
- `scripts/deprecated/large_scale_infer_vhh_esm_opt.py`

------------------------------------------------------------------------

## 📫 Support

For issues or questions, please open a GitHub issue.
