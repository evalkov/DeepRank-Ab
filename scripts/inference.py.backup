#!/usr/bin/env python3
"""
DeepRank-Ab Inference Pipeline
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import h5py
import pandas as pd
import requests
import torch
from Bio.PDB import PDBParser, PDBIO
from esm import FastaBatchedDataset, pretrained

# Add project root
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local modules
from src.DataSet import HDF5DataSet, PreCluster
from src.GraphGenMP import GraphHDF5
from src.tools.annotate import annotate_folder_one_by_one_mp
from src.NeuralNet_focal_EMA import NeuralNet
from src.EGNN import egnn


# ===============================================================
# CONFIGURATION
# ===============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

MODEL_PATH = (
    ROOT_DIR
    / "src"
    / "weights"
    / "af"
    / "top_mse_ep37_mse2.3192e-02_treg_ydockq_b128_e100_lr2e-03.pth.tar"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500
INCLUDE = ["mean", "per_tok"]

MAX_cores = 50
BATCH_SIZE = 64
NUM_WORKERS = 96

TARGET = "dockq"
TASK = "reg"
THRESHOLD = 0.23

NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]

ESM_MODEL = "esm2_t33_650M_UR50D"
EXPECTED_CHECKSUMS = [
    "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
]

# Logging
log = logging.getLogger("DeepRankAB")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(" [%(levelname)s] %(message)s"))
log.addHandler(handler)


# ===============================================================
# WORKSPACE SETUP
# ===============================================================


def setup_workspace(identificator: str) -> Path:
    workspace = Path.cwd() / identificator
    workspace.mkdir(parents=True, exist_ok=True)
    log.info(f"Workspace → {workspace}")
    return workspace


# ===============================================================
# PDB PROCESSING
# ===============================================================


def split_input_pdb(pdb_file: Path) -> List[Path]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    output_dir = pdb_file.parent / f"{pdb_file.stem}_split"
    output_dir.mkdir(exist_ok=True)

    io = PDBIO()
    saved = []

    is_ensemble = len(list(structure)) > 1

    for model in structure:
        model_id = model.id if is_ensemble else 0
        out = output_dir / f"{pdb_file.stem}_model_{model_id}.pdb"
        io.set_structure(model)
        io.save(str(out))
        saved.append(out)

    log.info(f"Split PDB into {len(saved)} model(s)")
    return saved


def three_to_one() -> dict:
    return {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }


def get_chain_sequence(structure, chain_id) -> str:
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return ""

    mapping = three_to_one()
    return "".join(mapping.get(res.get_resname(), "X") for res in chain)


def create_merged_pdb(
    pdb_file, heavy_chain_id, light_chain_id, antigen_chain_id, output_pdb
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)
    model = structure[0]

    chainA_res = []
    chainB_res = []

    for chain in model:
        cid = chain.id
        if cid in {heavy_chain_id, light_chain_id}:
            chainA_res.extend(res for res in chain)
        elif cid == antigen_chain_id:
            chainB_res.extend(res for res in chain)

    # Renumber
    for i, r in enumerate(chainA_res, start=1):
        r.id = (" ", i, " ")
        r.parent.id = "A"

    for i, r in enumerate(chainB_res, start=1):
        r.id = (" ", i, " ")
        r.parent.id = "B"

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


def convert_pdb_to_fastas(
    pdb_file: Path,
    fasta_outdir: Path,
    heavy_chain_id="H",
    light_chain_id="L",
    antigen_chain_id="A",
):
    fasta_outdir.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)

    root = pdb_file.stem

    seq_H = get_chain_sequence(structure, heavy_chain_id)
    seq_L = get_chain_sequence(structure, light_chain_id)
    seq_Ag = get_chain_sequence(structure, antigen_chain_id)

    # FASTA: annotation
    fasta_annot = fasta_outdir / f"{root}_HL.fasta"
    with open(fasta_annot, "w") as f:
        if seq_H:
            f.write(f">{root}.H\n{seq_H}\n")
        if seq_L:
            f.write(f">{root}.L\n{seq_L}\n")

    # FASTA: ESM
    fasta_esm = fasta_outdir / f"{root}_merged_A_B.fasta"
    with open(fasta_esm, "w") as f:
        f.write(f">{root}.A\n{seq_H + seq_L}\n")
        f.write(f">{root}.B\n{seq_Ag}\n")

    log.info(f"FASTA files created for {pdb_file.name}")
    return fasta_annot, fasta_esm


def preprocess_input_pdb(
    work_dir: Path,
    pdb_file: Path,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
):
    fasta_out = work_dir / "fastas"
    fasta_annot, fasta_esm = convert_pdb_to_fastas(pdb_file, fasta_out)

    merged_pdb = work_dir / "processed" / f"{pdb_file.stem}.pdb"
    create_merged_pdb(
        pdb_file, heavy_chain_id, light_chain_id, antigen_chain_id, merged_pdb
    )

    return merged_pdb, (fasta_annot, fasta_esm)


# ===============================================================
# WEIGHTS AND EMBEDDINGS
# ===============================================================


def calculate_checksum(path: str, algo="sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(url: str, dest: str) -> str:
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return dest


def fetch_weights() -> str:
    models = [
        (
            "WEIGHT_PATH",
            f"{ESM_MODEL}.pt",
            f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM_MODEL}.pt",
            EXPECTED_CHECKSUMS[0],
        ),
        (
            "REG_WEIGHT_PATH",
            f"{ESM_MODEL}-contact-regression.pt",
            f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM_MODEL}-contact-regression.pt",
            EXPECTED_CHECKSUMS[1],
        ),
    ]

    for env_var, fname, url, checksum in models:
        path = os.getenv(env_var) or fname
        if not os.path.exists(path):
            download_weights(url, path)

        if calculate_checksum(path) != checksum:
            log.warning(f"Checksum mismatch for {path}, re-downloading.")
            download_weights(url, path)

    return f"{ESM_MODEL}.pt"


def get_model_output(toks, model, layers):
    out = model(toks, repr_layers=layers, return_contacts="contacts" in INCLUDE)
    return {layer: t.cpu() for layer, t in out["representations"].items()}


def process_batch(labels, strs, outdir, representations):
    paths = []
    for i, label in enumerate(labels):
        data = {}
        trunc = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))

        if "per_tok" in INCLUDE:
            data["representations"] = {
                l: t[i, 1 : trunc + 1].clone() for l, t in representations.items()
            }

        if "mean" in INCLUDE:
            data["mean_representations"] = {
                l: t[i, 1 : trunc + 1].mean(0).clone()
                for l, t in representations.items()
            }

        outpath = outdir / f"{label}.pt"
        torch.save(data, outpath)
        paths.append(outpath)

    return paths


def get_embedding(fasta_file: Path, output_dir: Path) -> List[Path]:
    log.info("Generating ESM embeddings...")

    esm_path = fetch_weights()
    model, alphabet = pretrained.load_model_and_alphabet(esm_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS
    ]

    emb_paths = []
    with torch.no_grad():
        for labels, strs, toks in loader:
            if torch.cuda.is_available():
                toks = toks.cuda(non_blocking=True)

            reps = get_model_output(toks, model, repr_layers)
            emb_paths.extend(process_batch(labels, strs, output_dir, reps))

    return emb_paths


# ===============================================================
# CLUSTERING AND GRAPH GEN
# ===============================================================


def cluster(hdf5_path: str):
    dataset = HDF5DataSet(name="Train", root="./", database=hdf5_path)
    PreCluster(dataset, method="mcl")
    log.info("Clustering completed.")


def gen_graph_cdrs_orientation_contacts_one_by_one(
    pdb_dir: str,
    n_cores: int,
    outfile_name: str,
    use_regions=True,
    region_json=None,
    antigen_chainid=None,
    graph_type="atom",
    use_voro=False,
    embedding_path=None,
    contact_features=True,
):
    base = Path(pdb_dir).name
    tmpdir = tempfile.mkdtemp(prefix=f"{base}_")

    GraphHDF5(
        pdb_path=pdb_dir,
        graph_type=graph_type,
        outfile=outfile_name,
        nproc=n_cores,
        tmpdir=tmpdir,
        use_regions=use_regions,
        region_json=region_json,
        add_orientation=True,
        contact_features=True,
        antigen_chainid=antigen_chainid,
        use_voro=use_voro,
        embedding_path=embedding_path,
    )
    return outfile_name


def correct_json(work_dir: str) -> Path:
    region_json = Path(work_dir) / "annotations" / "annotations_cdrs.json"
    if not region_json.is_file():
        raise FileNotFoundError(f"Region JSON not found: {region_json}")

    with open(region_json) as f:
        data = json.load(f)

    new = {k.replace(".pdb", ""): v for k, v in data.items()}
    with open(region_json, "w") as f:
        json.dump(new, f, indent=4)

    return region_json


def add_embedding(work_dir: Path, hdf5_path: str, pdbid: str):
    base = work_dir / "processed" / "embeddings"

    with h5py.File(hdf5_path, "a") as f:
        for mol in f.keys():
            residues = f[mol]["nodes"][()]
            emb = torch.zeros(len(residues), 1)

            for i, res in enumerate(residues):
                chain, idx = res[0].decode(), int(res[1].decode())
                pt = base / f"{pdbid}.{chain}.pt"

                if not pt.is_file():
                    raise FileNotFoundError(f"Missing embedding: {pt}")

                data = torch.load(pt, map_location="cpu")
                vecs = data["representations"][33]
                emb[i] = vecs[idx - 1].mean()

            grp = f[mol].require_group("node_data")
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("embedding", data=emb.numpy())

    log.info(f"Added embeddings → {hdf5_path}")


# ===============================================================
# MODEL EVALUATION
# ===============================================================


def deeprank_evaluate_model(
    target_name: str,
    hdf5_test: str,
    model_path: str = MODEL_PATH,
    save_name: str = "eval_predictions.hdf5",
) -> Path:

    net = NeuralNet(
        database=hdf5_test,
        Net=egnn,
        node_feature=NODE_FEATURES,
        edge_feature=EDGE_FEATURES,
        target=None,
        task=TASK,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device_name=DEVICE,
        shuffle=False,
        pretrained_model=model_path,
        cluster_nodes="mcl",
    )

    out = Path(hdf5_test).parent / save_name
    net.predict(database_test=hdf5_test, hdf5=str(out))
    log.info(f"Predictions saved → {out}")

    return out


def hdf5_to_csv(hdf5_path: str) -> str:
    """Convert DeepRank-Ab prediction HDF5 to CSV with mol and dockq values."""

    hdf5_path = Path(hdf5_path)
    out_csv = hdf5_path.with_suffix(".csv")

    with h5py.File(hdf5_path, "r") as f:
        group = f["epoch_0000"]["pred"]

        # read data
        mol = group["mol"][()]
        dockq = group["outputs"][()]

        # decode mol names
        mol = [m.decode("utf-8") for m in mol]

    # create dataframe
    df = pd.DataFrame({"mol": mol, "dockq": dockq})

    # save csv
    df.to_csv(out_csv, index=False)
    return str(out_csv)


def parse_output(csv_output: str) -> None:
    """
    Read CSV created from HDF5 (mol, dockq) and rewrite it in clean format:
    pdb_id,predicted_dockq
    """
    df = pd.read_csv(csv_output)

    # Rename columns to consistent names
    df = df.rename(columns={"mol": "pdb_id", "dockq": "predicted_dockq"})

    # Log predictions
    for _, row in df.iterrows():
        log.info(
            f"Predicted dockq is {row['predicted_dockq']:.3f} for PDB ID {row['pdb_id']}"
        )

    # Save cleaned CSV
    df.to_csv(csv_output, index=False)
    log.info(f"Output written to {csv_output}")


# ===============================================================
# MAIN PIPELINE
# ===============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file")
    parser.add_argument("heavy_chain_id")
    parser.add_argument("light_chain_id")
    parser.add_argument("antigen_chain_id")
    args = parser.parse_args()

    pdb_file = Path(args.pdb_file)
    HID, LID, AID = args.heavy_chain_id, args.light_chain_id, args.antigen_chain_id

    identificator = f"{pdb_file.stem}-deeprank_ab_pred_{HID}{LID}_{AID}"
    work = setup_workspace(identificator)

    # Copy PDB
    copied = work / pdb_file.name
    shutil.copy(pdb_file, copied)

    # Split ensemble
    pdb_models = split_input_pdb(copied)
    cores = min(MAX_cores, len(pdb_models))

    for pdb in pdb_models:
        temp = work / "processed"
        _, (fasta_annot, fasta_esm) = preprocess_input_pdb(
            work, pdb, HID, LID, AID
        )

        embed_dir = temp / "embeddings"
        get_embedding(fasta_esm, embed_dir)

    # Annotate
    anno_dir = work / "annotations"
    anno_dir.mkdir(exist_ok=True)

    annotate_folder_one_by_one_mp(
        temp,
        Path(fasta_annot.parent),
        output_dir=str(anno_dir),
        n_cores=cores,
        antigen_chainid="B",
    )

    region_json = correct_json(work)
    log.info("Region JSON corrected.")

    # Graph generation
    graph_out = work / f"{identificator}_graph.hdf5"

    gen_graph_cdrs_orientation_contacts_one_by_one(
        pdb_dir=work / "processed",
        n_cores=cores,
        outfile_name=str(graph_out),
        use_regions=True,
        region_json=str(region_json),
        antigen_chainid=AID,
        graph_type="atom",
        use_voro=True,
        embedding_path=str(embed_dir),
        contact_features=True,
    )

    # Add embeddings
    for pdb in pdb_models:
        pdbid = pdb.stem
    add_embedding(work, str(graph_out), pdbid)

    # Cluster + predict
    cluster(str(graph_out))
    deeprank_evaluate_model(
        "dockq",
        str(graph_out),
        model_path=MODEL_PATH,
        save_name=f"{identificator}_predictions.hdf5",
    )

    # parse output
    csv_output = hdf5_to_csv(str(work / f"{identificator}_predictions.hdf5"))
    parse_output(csv_output)


if __name__ == "__main__":
    main()
