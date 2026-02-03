"""
Example: Large-Scale Inference Pipeline for DeepRank-Ab
------------------------------------------------------

This script demonstrates how to run DeepRank-Ab inference
of antibody-antigen models using batching, annotation reuse, and 
parallelized graph generation.

It is designed for HPC clusters, large shared-memory servers, or 
distributed environments. Adjust the configuration variables below 
for your system.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

import h5py
import torch

from DataSet import HDF5DataSet, PreCluster
from GraphGenMP_v3 import GraphHDF5
from annotate import annotate_folder_one_by_one_mp
from NeuralNet_focal_EMA import NeuralNet
from EGNN import egnn

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


# ============================================================
# USER CONFIGURATION — MODIFY FOR YOUR SYSTEM
# ============================================================

# Path to pretrained DeepRank-Ab model
MODEL_PATH = "top_mse_ep37_mse2.3192e-02_treg_ydockq_b128_e100_lr2e-03.pth.tar"

# Where intermediate files and graphs should be saved
PROCESSED_DIR = Path("processed_data")

# Number of CPU cores for annotation, graph generation, etc.
NUM_CORES = 32

# Graph batch size (number of PDBs processed per batch)
GRAPH_BATCH_SIZE = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]


# ============================================================
# CLUSTERING
# ============================================================

def cluster(hdf5_path: str) -> None:
    """Run MCL node pre-clustering on a graph database."""
    dataset = HDF5DataSet(name="EvalSet", root="./", database=hdf5_path)
    PreCluster(dataset, method="mcl")
    print(f"[CLUSTER] Done: {hdf5_path}")


# ============================================================
# GRAPH GENERATION
# ============================================================

def gen_graphs(
    pdb_dir: str,
    outfile_name: str,
    region_json: str,
    antigen_chainid: str,
    round_id: str,
    n_cores: int
) -> str:
    """Generate atom-level graphs for a batch of PDB files."""
    tmpdir = f"./tmp_{round_id}"
    os.makedirs(tmpdir, exist_ok=True)

    GraphHDF5(
        pdb_path=pdb_dir,
        outfile=outfile_name,
        graph_type="atom",
        nproc=n_cores,
        tmpdir=tmpdir,
        use_regions=True,
        region_json=region_json,
        antigen_chainid=antigen_chainid,
        add_orientation=True,
        use_voro=True,
        contact_features=True,
        embedding_path=None,
    )

    return outfile_name


# ============================================================
# EMBEDDING INJECTION
# ============================================================

def add_embedding(outfile: str, pdbid: str, embedding_root: Path):
    """
    Add precomputed embeddings to each graph.
    Expected format: {embedding_root}/{pdbid}.{chain}.pt
    """
    with h5py.File(outfile, "r+") as f:
        for mol in f.keys():
            residues = f[mol]["nodes"][()]
            emb_tensor = torch.zeros(len(residues), 1)

            for i, residue in enumerate(residues):
                chain_id = residue[0].decode()
                res_id = int(residue[1].decode())

                pt_file = embedding_root / f"{pdbid}.{chain_id}.pt"
                if not pt_file.is_file():
                    raise FileNotFoundError(f"Missing embedding file: {pt_file}")

                data = torch.load(pt_file, map_location="cpu")
                vec = data["representations"][33][res_id - 1]
                emb_tensor[i] = vec.mean()

            grp = f[mol].require_group("node_data")
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("embedding", data=emb_tensor.numpy())

    print(f"[EMBED] Added embedding → {outfile}")


# ============================================================
# MODEL EVALUATION
# ============================================================

def eval_model(
    hdf5_graph: str,
    model: str,
    out_hdf5: str
) -> Path:
    """Run EGNN inference."""
    net = NeuralNet(
        database=hdf5_graph,
        Net=egnn,
        node_feature=NODE_FEATURES,
        edge_feature=EDGE_FEATURES,
        target=None,
        task="reg",
        batch_size=64,
        num_workers=NUM_CORES,
        device_name=DEVICE,
        shuffle=False,
        pretrained_model=model,
        cluster_nodes="mcl",
    )

    net.predict(database_test=hdf5_graph, hdf5=out_hdf5)
    print(f"[PRED] Predictions saved → {out_hdf5}")
    return Path(out_hdf5)


# ============================================================
# BATCHED PIPELINE
# ============================================================

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def large_scale_inference(
    pdb_folder: str,
    pdbid: str,
    embedding_dir: str,
    antigen_chainid="A",
):
    """
    High-throughput DeepRank-Ab pipeline:
      1. Annotate CDRs once
      2. Split into batches
      3. For each batch:
         - Graph generation
         - Clustering
         - Embedding injection
         - inference
    """
    pdb_folder = Path(pdb_folder)
    embedding_dir = Path(embedding_dir)
    PROCESSED_DIR.mkdir(exist_ok=True)

    # 1. Annotate all models
    annotations_dir = pdb_folder / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    annotate_folder_one_by_one_mp(
        pdb_folder,
        output_dir=str(annotations_dir),
        n_cores=NUM_CORES,
        antigen_chainid=antigen_chainid,
    )

    region_json = annotations_dir / "annotations_cdrs.json"

    # 2. Collect PDB files
    pdbs = sorted(pdb_folder.glob("*.pdb"))
    batches = list(chunk(pdbs, GRAPH_BATCH_SIZE))

    out_h5_files = []

    for i, batch in enumerate(batches):
        batch_tag = f"{pdbid}_batch{i:04d}"

        # Prepare batch directory
        batch_dir = pdb_folder / f"_tmp_batch{i:04d}"
        batch_dir.mkdir(exist_ok=True)

        # Use hardlinks or copies
        for pdb in batch:
            os.link(pdb, batch_dir / pdb.name)

        # Graph HDF5
        graph_file = PROCESSED_DIR / f"{batch_tag}.hdf5"

        gen_graphs(
            pdb_dir=str(batch_dir),
            outfile_name=str(graph_file),
            region_json=str(region_json),
            antigen_chainid=antigen_chainid,
            round_id=batch_tag,
            n_cores=NUM_CORES,
        )

        # Clustering
        cluster(str(graph_file))

        # Embedding injection
        add_embedding(
            outfile=str(graph_file),
            pdbid=pdbid,
            embedding_root=embedding_dir,
        )

        # Inference
        pred_file = PROCESSED_DIR / f"{batch_tag}_predictions.hdf5"
        out_h5_files.append(
            eval_model(str(graph_file), MODEL_PATH, str(pred_file))
        )

        # Clean batch directory
        for f in batch_dir.iterdir():
            f.unlink()
        batch_dir.rmdir()

    return out_h5_files


# ============================================================
# MAIN
# ============================================================

def main():
    pdbid = sys.argv[1]

    large_scale_inference(
        pdb_folder=f"input_models/{pdbid}",
        pdbid=pdbid,
        embedding_dir="precomputed_embeddings/",
        antigen_chainid="A",
    )


if __name__ == "__main__":
    main()
