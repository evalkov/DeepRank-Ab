#!/usr/bin/env python3
"""
Large-scale (batched) DeepRank-Ab inference with VHH (H-only) support.

Key conventions:
- For each input PDB (and each MODEL in an ensemble), we create a merged PDB with:
    Chain A = antibody (H + optional L concatenated, renumbered from 1)
    Chain B = antigen  (renumbered from 1)
- FASTAs:
    {stem}_HL.fasta             : 1 or 2 records, ids end with .H and optionally .L
    {stem}_merged_A_B.fasta     : 2 records, ids {stem}.A and {stem}.B
- ESM embeddings written as:
    embeddings/{stem}.A.pt and embeddings/{stem}.B.pt
- Annotation uses corrected annotate.py:
    annotate_folder_one_by_one_mp(processed_pdb_dir, fasta_dir, output_dir, n_cores, antigen_chainid="B")
- Graph generation runs on processed PDBs. Embeddings are injected afterward into the graph HDF5.

Usage example:
  python3 large_scale_infer_vhh.py \
    --pdb-folder input_models/ \
    --out processed_data/ \
    --heavy H --light - --antigen T \
    --model-path top_mse_ep37_mse2.3192e-02_treg_ydockq_b128_e100_lr2e-03.pth.tar \
    --graph-batch-size 500 --num-cores 32 --dl-workers 8

Notes:
- This script expects to be run inside (or with PYTHONPATH pointing to) the DeepRank-Ab repo
  so that DataSet, GraphGenMP, NeuralNet_focal_EMA, EGNN, and annotate imports resolve.
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent          # .../DeepRank-Ab/scripts
ROOT_DIR   = SCRIPT_DIR.parent                       # .../DeepRank-Ab
SRC_DIR    = ROOT_DIR / "src"                        # .../DeepRank-Ab/src

# Ensure imports work no matter where you run from
sys.path.insert(0, str(ROOT_DIR))

import h5py
import torch
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import PPBuilder

from src.DataSet import HDF5DataSet, PreCluster
from src.GraphGenMP import GraphHDF5
from src.tools.annotate import annotate_folder_one_by_one_mp
from src.NeuralNet_focal_EMA import NeuralNet
from src.EGNN import egnn

# ESM
from esm import FastaBatchedDataset, pretrained


# ----------------------------
# Logging
# ----------------------------
log = logging.getLogger("drab-batch")
log.setLevel(logging.INFO)
_hdl = logging.StreamHandler()
_hdl.setFormatter(logging.Formatter(" [%(levelname)s] %(message)s"))
log.addHandler(_hdl)


# ----------------------------
# Config defaults
# ----------------------------
NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]

TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500
INCLUDE = ["mean", "per_tok"]

# If you already set ESM weights via env vars, this will work fine
ESM_MODEL = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")


# ----------------------------
# Helpers
# ----------------------------
def is_missing_chain(x: str) -> bool:
    return (x is None) or (str(x).strip() == "") or (str(x).strip() == "-") or (str(x).strip().lower() == "none")


def chunk_list(items: List[Path], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_models(pdb_path: Path, out_dir: Path) -> List[Path]:
    """Split ensemble PDB into per-model PDBs. If only one model, returns single file."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))

    models = list(struct)
    is_ensemble = len(models) > 1

    io = PDBIO()
    saved: List[Path] = []
    safe_mkdir(out_dir)

    for m in models:
        mid = m.id if is_ensemble else 0
        out = out_dir / f"{pdb_path.stem}_model_{mid}.pdb"
        io.set_structure(m)
        io.save(str(out))
        saved.append(out)

    return saved


def chain_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    """Extract polypeptide sequence for a chain using PPBuilder (more robust than residue-name mapping)."""
    if is_missing_chain(chain_id):
        return ""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = struct[0]
    if chain_id not in model:
        return ""

    chain = model[chain_id]
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        return ""
    # If multiple peptides (chain breaks), concatenate (matches typical DeepRank-Ab behavior)
    return "".join(str(pp.get_sequence()) for pp in peptides)


def _copy_residue(res):
    """Biopython Residue copy."""
    return res.copy()


def build_merged_structure(
    pdb_path: Path,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    out_pdb: Path,
) -> Tuple[str, str, str]:
    """
    Create a fresh structure with two chains:
      A: Ab (H + optional L) renumbered from 1
      B: Ag (antigen) renumbered from 1

    Returns: (seqH, seqL, seqAg) from the *input* PDB chains.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model_in = struct[0]

    # sequences from original chains for FASTA creation
    seqH = chain_sequence_from_pdb(pdb_path, heavy_chain_id)
    seqL = "" if is_missing_chain(light_chain_id) else chain_sequence_from_pdb(pdb_path, light_chain_id)
    seqAg = chain_sequence_from_pdb(pdb_path, antigen_chain_id)

    if heavy_chain_id not in model_in:
        raise ValueError(f"{pdb_path.name}: heavy chain '{heavy_chain_id}' not found")
    if antigen_chain_id not in model_in:
        raise ValueError(f"{pdb_path.name}: antigen chain '{antigen_chain_id}' not found")
    if (not is_missing_chain(light_chain_id)) and (light_chain_id not in model_in):
        # Allow missing L in PDB if user passed '-', but if they passed a real L id, enforce it.
        raise ValueError(f"{pdb_path.name}: light chain '{light_chain_id}' not found")

    # Build new structure
    s = Structure(pdb_path.stem)
    m = Model(0)
    s.add(m)

    chainA = Chain("A")
    chainB = Chain("B")
    m.add(chainA)
    m.add(chainB)

    # Helper: append residues renumbered
    def append_chain(dst_chain: Chain, src_chain_ids: List[str]):
        idx = 1
        for cid in src_chain_ids:
            if is_missing_chain(cid):
                continue
            src_chain = model_in[cid]
            for res in src_chain:
                new_res = _copy_residue(res)
                new_res.id = (" ", idx, " ")
                dst_chain.add(new_res)
                idx += 1

    # Ab chain: H then (optional) L
    append_chain(chainA, [heavy_chain_id, light_chain_id])
    # Ag chain
    append_chain(chainB, [antigen_chain_id])

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(out_pdb))

    return seqH, seqL, seqAg


def write_fastas(
    stem: str,
    fasta_dir: Path,
    seqH: str,
    seqL: str,
    seqAg: str,
) -> Tuple[Path, Path]:
    """
    Writes:
      {stem}_HL.fasta           (H and optionally L)
      {stem}_merged_A_B.fasta   (A=H+L, B=Ag)
    Returns: (fasta_hl, fasta_ab)
    """
    safe_mkdir(fasta_dir)

    fasta_hl = fasta_dir / f"{stem}_HL.fasta"
    with open(fasta_hl, "w") as f:
        if seqH:
            f.write(f">{stem}.H\n{seqH}\n")
        if seqL:
            f.write(f">{stem}.L\n{seqL}\n")

    fasta_ab = fasta_dir / f"{stem}_merged_A_B.fasta"
    with open(fasta_ab, "w") as f:
        f.write(f">{stem}.A\n{seqH + seqL}\n")
        f.write(f">{stem}.B\n{seqAg}\n")

    return fasta_hl, fasta_ab


# ----------------------------
# ESM embedding
# ----------------------------
def _get_model_output(toks, model, layers):
    out = model(toks, repr_layers=layers, return_contacts=("contacts" in INCLUDE))
    return {layer: t.cpu() for layer, t in out["representations"].items()}


def _process_batch(labels, strs, outdir: Path, representations) -> List[Path]:
    paths: List[Path] = []
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


def generate_esm_embeddings(fasta_file: Path, out_dir: Path, device: str) -> List[Path]:
    """
    Generates embeddings for records in fasta_file.
    Writes <label>.pt where label is FASTA header id (without '>').
    """
    safe_mkdir(out_dir)
    log.info(f"[ESM] Embeddings from {fasta_file.name}")

    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
    model.eval()

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(str(fasta_file))
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]

    out_paths: List[Path] = []
    with torch.no_grad():
        for labels, strs, toks in loader:
            if device.startswith("cuda") and torch.cuda.is_available():
                toks = toks.cuda(non_blocking=True)
            reps = _get_model_output(toks, model, repr_layers)
            out_paths.extend(_process_batch(labels, strs, out_dir, reps))

    return out_paths


# ----------------------------
# Graph + embeddings injection
# ----------------------------
def correct_region_json(region_json: Path) -> None:
    """
    DeepRank-Ab annotate writes keys like "<stem>.pdb" sometimes.
    Graph generator often uses "<stem>".
    Normalize keys by stripping ".pdb" suffix.
    """
    if not region_json.is_file():
        raise FileNotFoundError(f"Missing region JSON: {region_json}")

    with open(region_json, "r") as f:
        data = json.load(f)
    new = {k.replace(".pdb", ""): v for k, v in data.items()}

    with open(region_json, "w") as f:
        json.dump(new, f, indent=2)


def gen_graphs(
    pdb_dir: Path,
    outfile: Path,
    region_json: Path,
    n_cores: int,
    antigen_chainid: str = "B",
    tmp_base: Optional[Path] = None,
):
    tmpdir = Path(tempfile.mkdtemp(prefix="drab_graph_", dir=str(tmp_base) if tmp_base else None))
    try:
        GraphHDF5(
            pdb_path=str(pdb_dir),
            outfile=str(outfile),
            graph_type="atom",
            nproc=n_cores,
            tmpdir=str(tmpdir),
            use_regions=True,
            region_json=str(region_json),
            antigen_chainid=antigen_chainid,
            add_orientation=True,
            use_voro=True,
            contact_features=True,
            embedding_path=None,  # we inject ourselves below
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def inject_embeddings(graph_hdf5: Path, embeddings_dir: Path) -> None:
    """
    Inject per-residue embedding feature into each mol in graph HDF5.

    Expected embedding files:
      embeddings_dir/<mol>.A.pt and embeddings_dir/<mol>.B.pt

    Assumes residues in chain A/B are renumbered from 1..N so res_id-1 indexes ESM tokens.
    """
    with h5py.File(graph_hdf5, "r+") as f:
        for mol in f.keys():
            residues = f[mol]["nodes"][()]  # typically array of [chain, resid, ...]
            emb = torch.zeros((len(residues), 1), dtype=torch.float32)

            # cache per-chain pt loads
            cache: Dict[str, Dict] = {}

            for i, res in enumerate(residues):
                chain = res[0].decode() if isinstance(res[0], (bytes, bytearray)) else str(res[0])
                resid = int(res[1].decode()) if isinstance(res[1], (bytes, bytearray)) else int(res[1])

                pt_path = embeddings_dir / f"{mol}.{chain}.pt"
                if not pt_path.is_file():
                    # leave zero and continue
                    continue

                if chain not in cache:
                    cache[chain] = torch.load(pt_path, map_location="cpu")

                data = cache[chain]
                vecs = data["representations"][33]
                j = resid - 1
                if 0 <= j < vecs.shape[0]:
                    emb[i, 0] = vecs[j].mean().item()

            grp = f[mol].require_group("node_data")
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("embedding", data=emb.numpy())


# ----------------------------
# Clustering + prediction
# ----------------------------
def cluster_mcl(graph_hdf5: Path) -> None:
    dataset = HDF5DataSet(name="EvalSet", root="./", database=str(graph_hdf5))
    PreCluster(dataset, method="mcl")


def predict(
    graph_hdf5: Path,
    model_path: Path,
    out_pred_hdf5: Path,
    device: str,
    dl_workers: int,
    batch_size: int,
    num_cores: int,
) -> None:
    # Clamp dl workers to something sane on head nodes / small systems
    if dl_workers < 0:
        dl_workers = 0
    if dl_workers > num_cores:
        dl_workers = num_cores

    net = NeuralNet(
        database=str(graph_hdf5),
        Net=egnn,
        node_feature=NODE_FEATURES,
        edge_feature=EDGE_FEATURES,
        target=None,
        task="reg",
        batch_size=batch_size,
        num_workers=dl_workers,
        device_name=device,
        shuffle=False,
        pretrained_model=str(model_path),
        cluster_nodes="mcl",
    )

    net.predict(database_test=str(graph_hdf5), hdf5=str(out_pred_hdf5))


# ----------------------------
# Main pipeline
# ----------------------------
def process_one_pdb(
    pdb_path: Path,
    heavy: str,
    light: str,
    antigen: str,
    processed_dir: Path,
    fasta_dir: Path,
    emb_dir: Path,
    device: str,
    timings: Dict[str, float],
) -> Path:
    """
    Process a single PDB (single MODEL file):
      - create merged processed PDB with A/B chains
      - write fastas
      - generate embeddings for A/B (ESM)
    Returns path to merged PDB.
    """
    stem = pdb_path.stem
    merged_pdb = processed_dir / f"{stem}.pdb"

    t0 = perf_counter()
    seqH, seqL, seqAg = build_merged_structure(
        pdb_path=pdb_path,
        heavy_chain_id=heavy,
        light_chain_id=light,
        antigen_chain_id=antigen,
        out_pdb=merged_pdb,
    )
    fasta_hl, fasta_ab = write_fastas(stem, fasta_dir, seqH, seqL, seqAg)
    t1 = perf_counter()
    timings["prep_s"] += (t1 - t0)

    # ESM embeddings for A and B records
    t2 = perf_counter()
    generate_esm_embeddings(fasta_ab, emb_dir, device=device)
    t3 = perf_counter()
    timings["esm_s"] += (t3 - t2)

    return merged_pdb


def run_batched_inference(
    pdb_folder: Path,
    out_root: Path,
    model_path: Path,
    heavy: str,
    light: str,
    antigen: str,
    antigen_chainid_for_graph: str,
    num_cores: int,
    graph_batch_size: int,
    dl_workers: int,
    batch_size: int,
    device: str,
    tmp_base: Optional[Path] = None,
    keep_intermediates: bool = False,
) -> List[Path]:
    """
    End-to-end:
      - split ensembles → per-model pdbs
      - per batch: preprocess + embeddings, annotate, graphs, inject emb, cluster, predict
    """
    pdb_folder = pdb_folder.resolve()
    out_root = out_root.resolve()
    safe_mkdir(out_root)

    # Discover inputs
    pdbs_in = sorted(pdb_folder.glob("*.pdb"))
    if not pdbs_in:
        raise FileNotFoundError(f"No PDBs found in {pdb_folder}")

    # Expand ensembles to per-model pdbs into a staging area
    staging = out_root / "staging_models"
    safe_mkdir(staging)

    expanded: List[Path] = []
    for pdb in pdbs_in:
        out_dir = staging / pdb.stem
        expanded.extend(split_models(pdb, out_dir))

    log.info(f"Found {len(pdbs_in)} input PDBs → expanded to {len(expanded)} model PDBs")

    batches = list(chunk_list(expanded, graph_batch_size))
    log.info(f"Batches: {len(batches)} (size ~{graph_batch_size})")

    pred_files: List[Path] = []

    for bi, batch in enumerate(batches):
        tag = f"batch{bi:04d}"
        log.info(f"=== Processing {tag}: n={len(batch)} ===")

        # TOTAL batch walltime (requested)
        tB0 = perf_counter()

        # Per-batch workspace
        batch_root = out_root / tag
        processed_dir = safe_mkdir(batch_root / "processed")
        fasta_dir = safe_mkdir(batch_root / "fastas")
        emb_dir = safe_mkdir(batch_root / "embeddings")
        anno_dir = safe_mkdir(batch_root / "annotations")

        # Per-batch timings (requested)
        timings: Dict[str, float] = {
            "prep_s": 0.0,
            "esm_s": 0.0,
            "graphs_s": 0.0,
            "infer_s": 0.0,
        }

        # Preprocess each per-model pdb
        for pdb_model in batch:
            try:
                process_one_pdb(
                    pdb_path=pdb_model,
                    heavy=heavy,
                    light=light,
                    antigen=antigen,
                    processed_dir=processed_dir,
                    fasta_dir=fasta_dir,
                    emb_dir=emb_dir,
                    device=device,
                    timings=timings,
                )
            except Exception as e:
                log.warning(f"[SKIP] {pdb_model.name}: {e}")

        log.info(f"[TIME] {tag} PREP: {timings['prep_s']:.3f}s   ESM: {timings['esm_s']:.3f}s")

        # Annotate (uses *_HL.fasta)
        annotate_folder_one_by_one_mp(
            processed_dir,
            fasta_dir,
            output_dir=str(anno_dir),
            n_cores=num_cores,
            antigen_chainid=antigen_chainid_for_graph,  # usually "B" for merged PDB
        )
        region_json = anno_dir / "annotations_cdrs.json"
        correct_region_json(region_json)

        # Graph gen + embed inject + clustering
        tG0 = perf_counter()
        graph_hdf5 = out_root / f"graphs_{tag}.hdf5"
        gen_graphs(
            pdb_dir=processed_dir,
            outfile=graph_hdf5,
            region_json=region_json,
            n_cores=num_cores,
            antigen_chainid=antigen_chainid_for_graph,
            tmp_base=tmp_base,
        )

        inject_embeddings(graph_hdf5, emb_dir)
        cluster_mcl(graph_hdf5)
        tG1 = perf_counter()
        timings["graphs_s"] += (tG1 - tG0)
        log.info(f"[TIME] {tag} GRAPHS: {timings['graphs_s']:.3f}s")

        # Predict
        tI0 = perf_counter()
        pred_hdf5 = out_root / f"pred_{tag}.hdf5"
        predict(
            graph_hdf5=graph_hdf5,
            model_path=model_path,
            out_pred_hdf5=pred_hdf5,
            device=device,
            dl_workers=dl_workers,
            batch_size=batch_size,
            num_cores=num_cores,
        )
        tI1 = perf_counter()
        timings["infer_s"] += (tI1 - tI0)
        log.info(f"[TIME] {tag} INFER: {timings['infer_s']:.3f}s")

        pred_files.append(pred_hdf5)
        log.info(f"[DONE] {tag} → {pred_hdf5}")

        if not keep_intermediates:
            shutil.rmtree(batch_root, ignore_errors=True)

        tB1 = perf_counter()
        log.info(f"[TIME] {tag} TOTAL: {(tB1 - tB0):.3f}s")

    if not keep_intermediates:
        shutil.rmtree(staging, ignore_errors=True)

    return pred_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-folder", required=True, help="Folder containing input *.pdb")
    ap.add_argument("--out", required=True, help="Output folder for batch graphs/preds")
    ap.add_argument("--model-path", required=True, help="Path to pretrained DeepRank-Ab .pth.tar")
    ap.add_argument("--heavy", required=True, help="Heavy chain id in input PDB (e.g. H)")
    ap.add_argument("--light", default="-", help="Light chain id in input PDB (e.g. L) or '-' for VHH")
    ap.add_argument("--antigen", required=True, help="Antigen chain id in input PDB (e.g. T)")
    ap.add_argument("--antigen-chainid-for-graph", default="B",
                    help="Antigen chain id in MERGED processed PDB for graph features (default B)")
    ap.add_argument("--num-cores", type=int, default=int(os.environ.get("NUM_CORES", "32")))
    ap.add_argument("--graph-batch-size", type=int, default=int(os.environ.get("GRAPH_BATCH_SIZE", "500")))
    ap.add_argument("--dl-workers", type=int, default=int(os.environ.get("DL_WORKERS", "8")),
                    help="PyTorch DataLoader workers for inference")
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")),
                    help="Inference batch size")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--tmp-base", default=os.environ.get("TMPDIR", None),
                    help="Base temp dir (e.g. local SSD TMPDIR).")
    ap.add_argument("--keep-intermediates", action="store_true", help="Keep per-batch processed/fastas/embeddings.")
    args = ap.parse_args()

    pdb_folder = Path(args.pdb_folder)
    out_root = Path(args.out)
    model_path = Path(args.model_path)

    tmp_base = Path(args.tmp_base) if args.tmp_base else None

    preds = run_batched_inference(
        pdb_folder=pdb_folder,
        out_root=out_root,
        model_path=model_path,
        heavy=args.heavy,
        light=args.light,
        antigen=args.antigen,
        antigen_chainid_for_graph=args.antigen_chainid_for_graph,
        num_cores=args.num_cores,
        graph_batch_size=args.graph_batch_size,
        dl_workers=args.dl_workers,
        batch_size=args.batch_size,
        device=args.device,
        tmp_base=tmp_base,
        keep_intermediates=args.keep_intermediates,
    )

    log.info("All batches complete.")
    log.info("Prediction files:")
    for p in preds:
        log.info(f"  {p}")


if __name__ == "__main__":
    main()

