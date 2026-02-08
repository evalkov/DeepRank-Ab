#!/usr/bin/env python3
"""
Large-scale (batched) DeepRank-Ab inference with VHH (H-only) support.

UPDATED (2026-01-31): 4-GPU sharded, scalar-only ESM embeddings

This script runs batched DeepRank-Ab inference and generates ESM embeddings once per batch,
sharded across multiple GPUs. Embeddings are stored as scalar-per-residue arrays in HDF5 part
files and injected into the graph HDF5 before inference.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
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

from esm import FastaBatchedDataset, pretrained


log = logging.getLogger("drab-batch")
log.setLevel(logging.INFO)
_hdl = logging.StreamHandler()
_hdl.setFormatter(logging.Formatter(" [%(levelname)s] %(message)s"))
log.addHandler(_hdl)

NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]

DEFAULT_TOKS_PER_BATCH = int(os.environ.get("ESM_TOKS_PER_BATCH", "12288"))
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500
ESM_MODEL = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")


def is_missing_chain(x: str) -> bool:
    return (x is None) or (str(x).strip() == "") or (str(x).strip() == "-") or (str(x).strip().lower() == "none")


def chunk_list(items: List[Path], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_models(pdb_path: Path, out_dir: Path) -> List[Path]:
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
    return "".join(str(pp.get_sequence()) for pp in peptides)


def _copy_residue(res):
    return res.copy()


def build_merged_structure(
    pdb_path: Path,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    out_pdb: Path,
) -> Tuple[str, str, str]:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model_in = struct[0]

    seqH = chain_sequence_from_pdb(pdb_path, heavy_chain_id)
    seqL = "" if is_missing_chain(light_chain_id) else chain_sequence_from_pdb(pdb_path, light_chain_id)
    seqAg = chain_sequence_from_pdb(pdb_path, antigen_chain_id)

    if heavy_chain_id not in model_in:
        raise ValueError(f"{pdb_path.name}: heavy chain '{heavy_chain_id}' not found")
    if antigen_chain_id not in model_in:
        raise ValueError(f"{pdb_path.name}: antigen chain '{antigen_chain_id}' not found")
    if (not is_missing_chain(light_chain_id)) and (light_chain_id not in model_in):
        raise ValueError(f"{pdb_path.name}: light chain '{light_chain_id}' not found")

    s = Structure(pdb_path.stem)
    m = Model(0)
    s.add(m)
    chainA = Chain("A")
    chainB = Chain("B")
    m.add(chainA)
    m.add(chainB)

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

    append_chain(chainA, [heavy_chain_id, light_chain_id])
    append_chain(chainB, [antigen_chain_id])

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(out_pdb))

    return seqH, seqL, seqAg


def write_hl_fasta(stem: str, fasta_dir: Path, seqH: str, seqL: str) -> Path:
    safe_mkdir(fasta_dir)
    fasta_hl = fasta_dir / f"{stem}_HL.fasta"
    with open(fasta_hl, "w") as f:
        if seqH:
            f.write(f">{stem}.H\n{seqH}\n")
        if seqL:
            f.write(f">{stem}.L\n{seqL}\n")
    return fasta_hl


def correct_region_json(region_json: Path) -> None:
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
            embedding_path=None,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class SeqRecord:
    label: str
    mol: str
    chain: str
    length: int
    sequence: str


def write_manifest_tsv(records: List[SeqRecord], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as f:
        f.write("label\tmol\tchain\tlength\tsequence\n")
        for r in records:
            f.write(f"{r.label}\t{r.mol}\t{r.chain}\t{r.length}\t{r.sequence}\n")


def shard_records_balanced(records: List[SeqRecord], n_shards: int) -> Tuple[List[List[SeqRecord]], Dict[str, int]]:
    items = sorted(records, key=lambda r: (r.length + 2), reverse=True)
    shards: List[List[SeqRecord]] = [[] for _ in range(n_shards)]
    loads = [0 for _ in range(n_shards)]
    label_to_shard: Dict[str, int] = {}
    for r in items:
        i = min(range(n_shards), key=lambda k: loads[k])
        shards[i].append(r)
        loads[i] += (r.length + 2)
        label_to_shard[r.label] = i
    return shards, label_to_shard


def write_shard_tsv(records: List[SeqRecord], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as f:
        f.write("label\tlength\tsequence\n")
        for r in records:
            f.write(f"{r.label}\t{r.length}\t{r.sequence}\n")


def shard_tsv_to_fasta(shard_tsv: Path, out_fasta: Path) -> int:
    n = 0
    with open(shard_tsv, "r") as fin, open(out_fasta, "w") as fout:
        _ = fin.readline()
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label, _len, seq = parts[0], parts[1], parts[2]
            if not seq:
                continue
            fout.write(f">{label}\n{seq}\n")
            n += 1
    return n


def validate_embedding_parts(shard_tsvs: List[Path], part_h5s: List[Path]) -> None:
    for i, (stsv, ph5) in enumerate(zip(shard_tsvs, part_h5s)):
        if not ph5.is_file():
            raise FileNotFoundError(f"Missing embedding part file: {ph5}")
        expected: Dict[str, int] = {}
        with open(stsv, "r") as f:
            _ = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                label, length_s, _seq = line.split("\t", 2)
                expected[label] = int(length_s)
        with h5py.File(ph5, "r") as h:
            if "scalar" not in h:
                raise RuntimeError(f"{ph5}: missing group 'scalar'")
            g = h["scalar"]
            missing = []
            badlen = []
            for label, exp_len in expected.items():
                if label not in g:
                    missing.append(label)
                    continue
                got_len = int(g[label].shape[0])
                if got_len != exp_len:
                    badlen.append((label, exp_len, got_len))
        if missing:
            raise RuntimeError(f"Embedding validation failed shard {i}: missing {len(missing)} labels (e.g. {missing[:3]})")
        if badlen:
            raise RuntimeError(f"Embedding validation failed shard {i}: {len(badlen)} length mismatches (e.g. {badlen[:3]})")


def run_esm_worker(
    shard_tsv: Path,
    out_h5: Path,
    report_json: Path,
    done_sentinel: Path,
    device: str,
    toks_per_batch: int,
    scalar_dtype: str,
) -> None:
    t0 = perf_counter()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    fasta_path = out_h5.with_suffix(".fasta")
    n_records = shard_tsv_to_fasta(shard_tsv, fasta_path)

    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(str(fasta_path))
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]
    layer = REPR_LAYERS[0]

    if scalar_dtype.lower() not in ("float16", "float32"):
        raise ValueError(f"scalar_dtype must be float16 or float32, got {scalar_dtype}")
    np_dtype = "float16" if scalar_dtype.lower() == "float16" else "float32"

    total_len = 0
    with h5py.File(out_h5, "w") as h:
        h.attrs["esm_model"] = ESM_MODEL
        h.attrs["repr_layer"] = int(layer)
        h.attrs["scalar_def"] = "mean_over_embedding_dim"
        h.attrs["truncation_seq_length"] = int(TRUNCATION_SEQ_LENGTH)
        h.attrs["toks_per_batch"] = int(toks_per_batch)
        h.attrs["dtype"] = np_dtype
        g = h.create_group("scalar")

        with torch.no_grad():
            for labels, strs, toks in loader:
                if device.startswith("cuda") and torch.cuda.is_available():
                    toks = toks.cuda(non_blocking=True)
                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                reps = out["representations"][repr_layers[0]]

                for i, label in enumerate(labels):
                    seq_len = len(strs[i])
                    trunc = min(TRUNCATION_SEQ_LENGTH, seq_len)
                    reps_i = reps[i, 1 : trunc + 1].detach()
                    scalar = reps_i.mean(dim=1)
                    scalar = scalar.to(dtype=torch.float16 if np_dtype == "float16" else torch.float32)
                    arr = scalar.cpu().numpy()
                    total_len += int(arr.shape[0])
                    g.create_dataset(str(label), data=arr, compression="lzf", chunks=True)

    report = {
        "shard_tsv": str(shard_tsv),
        "out_h5": str(out_h5),
        "n_records": int(n_records),
        "total_residues": int(total_len),
        "wall_s": float(perf_counter() - t0),
        "device": device,
        "toks_per_batch": int(toks_per_batch),
        "scalar_dtype": np_dtype,
        "esm_model": ESM_MODEL,
        "repr_layer": int(layer),
    }
    report_json.write_text(json.dumps(report, indent=2))
    done_sentinel.write_text("ok\n")


def run_sharded_embeddings_subprocess(
    embeddings_dir: Path,
    n_gpus: int,
    toks_per_batch: int,
    scalar_dtype: str,
    python_exe: str,
    script_path: Path,
    label_to_shard: Dict[str, int],
) -> Tuple[List[Path], float]:
    part_h5s: List[Path] = []
    shard_tsvs: List[Path] = []
    reports: List[Path] = []
    sentinels: List[Path] = []

    for i in range(n_gpus):
        shard_tsv = embeddings_dir / f"shard{i}.tsv"
        out_h5 = embeddings_dir / f"emb_part{i}.h5"
        report_json = embeddings_dir / f"emb_part{i}.report.json"
        done = embeddings_dir / f"embed_done.part{i}"
        shard_tsvs.append(shard_tsv)
        part_h5s.append(out_h5)
        reports.append(report_json)
        sentinels.append(done)
        if done.exists():
            done.unlink()
        if report_json.exists():
            report_json.unlink()

    (embeddings_dir / "label_to_shard.json").write_text(json.dumps(label_to_shard, indent=2))

    procs: List[subprocess.Popen] = []
    t0 = perf_counter()

    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        args = [
            python_exe, str(script_path),
            "--esm-worker",
            "--shard-tsv", str(shard_tsvs[i]),
            "--out-h5", str(part_h5s[i]),
            "--report-json", str(reports[i]),
            "--done-sentinel", str(sentinels[i]),
            "--device", "cuda",
            "--esm-toks-per-batch", str(toks_per_batch),
            "--esm-scalar-dtype", str(scalar_dtype),
        ]
        procs.append(subprocess.Popen(args, env=env))

    rc = [p.wait() for p in procs]
    bad = [(i, r) for i, r in enumerate(rc) if r != 0]
    if bad:
        raise RuntimeError(f"One or more ESM workers failed: {bad}")

    missing = [str(s) for s in sentinels if not s.is_file()]
    if missing:
        raise RuntimeError(f"Missing ESM sentinels: {missing}")

    validate_embedding_parts(shard_tsvs, part_h5s)

    worker_reports = []
    for r in reports:
        if r.is_file():
            worker_reports.append(json.loads(r.read_text()))
    (embeddings_dir / "embed_report.json").write_text(json.dumps(
        {"n_gpus": n_gpus, "toks_per_batch": toks_per_batch, "scalar_dtype": scalar_dtype, "workers": worker_reports},
        indent=2
    ))

    crit = max((wr.get("wall_s", 0.0) for wr in worker_reports), default=0.0)
    return part_h5s, float(crit if crit > 0 else (perf_counter() - t0))


def inject_embeddings_from_parts(graph_hdf5: Path, part_h5s: List[Path], label_to_shard_json: Path) -> None:
    label_to_shard = json.loads(label_to_shard_json.read_text())
    part_handles = [h5py.File(p, "r") for p in part_h5s]
    try:
        scalar_groups = [h["scalar"] for h in part_handles]
        missing_labels = 0
        oob = 0

        with h5py.File(graph_hdf5, "r+") as f:
            for mol in f.keys():
                residues = f[mol]["nodes"][()]
                emb = torch.zeros((len(residues), 1), dtype=torch.float32)
                cache: Dict[str, object] = {}

                for i, res in enumerate(residues):
                    chain = res[0].decode() if isinstance(res[0], (bytes, bytearray)) else str(res[0])
                    resid = int(res[1].decode()) if isinstance(res[1], (bytes, bytearray)) else int(res[1])

                    label = f"{mol}.{chain}"
                    shard = label_to_shard.get(label, None)
                    if shard is None or shard < 0 or shard >= len(scalar_groups):
                        missing_labels += 1
                        continue

                    if label not in cache:
                        g = scalar_groups[shard]
                        if label not in g:
                            missing_labels += 1
                            continue
                        cache[label] = g[label][()]

                    arr = cache[label]
                    j = resid - 1
                    if 0 <= j < len(arr):
                        emb[i, 0] = float(arr[j])
                    else:
                        oob += 1

                grp = f[mol].require_group("node_data")
                if "embedding" in grp:
                    del grp["embedding"]
                grp.create_dataset("embedding", data=emb.numpy())

        if missing_labels:
            log.warning(f"[EMB] Missing labels during injection: {missing_labels}")
        if oob:
            log.warning(f"[EMB] Residue indices out-of-bounds during injection: {oob}")

    finally:
        for h in part_handles:
            try:
                h.close()
            except Exception:
                pass


def cluster_mcl(graph_hdf5: Path) -> None:
    dataset = HDF5DataSet(name="EvalSet", root="./", database=str(graph_hdf5))
    PreCluster(dataset, method="mcl")


def predict(graph_hdf5: Path, model_path: Path, out_pred_hdf5: Path, device: str, dl_workers: int, batch_size: int, num_cores: int) -> None:
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


def process_one_pdb_no_esm(
    pdb_path: Path,
    heavy: str,
    light: str,
    antigen: str,
    processed_dir: Path,
    fasta_dir: Path,
    timings: Dict[str, float],
) -> Tuple[Path, Optional[SeqRecord], Optional[SeqRecord]]:
    stem = pdb_path.stem
    merged_pdb = processed_dir / f"{stem}.pdb"
    t0 = perf_counter()
    seqH, seqL, seqAg = build_merged_structure(pdb_path, heavy, light, antigen, merged_pdb)
    _ = write_hl_fasta(stem, fasta_dir, seqH, seqL)
    timings["prep_s"] += (perf_counter() - t0)

    seqA = (seqH or "") + (seqL or "")
    seqB = (seqAg or "")
    recA = SeqRecord(label=f"{stem}.A", mol=stem, chain="A", length=len(seqA), sequence=seqA) if seqA else None
    recB = SeqRecord(label=f"{stem}.B", mol=stem, chain="B", length=len(seqB), sequence=seqB) if seqB else None
    return merged_pdb, recA, recB


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
    tmp_base: Optional[Path],
    keep_intermediates: bool,
    esm_gpus: int,
    esm_toks_per_batch: int,
    esm_scalar_dtype: str,
) -> List[Path]:
    pdb_folder = pdb_folder.resolve()
    out_root = out_root.resolve()
    safe_mkdir(out_root)

    pdbs_in = sorted(pdb_folder.glob("*.pdb"))
    if not pdbs_in:
        raise FileNotFoundError(f"No PDBs found in {pdb_folder}")

    staging = out_root / "staging_models"
    safe_mkdir(staging)

    expanded: List[Path] = []
    for pdb in pdbs_in:
        expanded.extend(split_models(pdb, staging / pdb.stem))

    log.info(f"Found {len(pdbs_in)} input PDBs → expanded to {len(expanded)} model PDBs")
    batches = list(chunk_list(expanded, graph_batch_size))
    log.info(f"Batches: {len(batches)} (size ~{graph_batch_size})")

    python_exe = sys.executable
    script_path = Path(__file__).resolve()

    pred_files: List[Path] = []

    for bi, batch in enumerate(batches):
        tag = f"batch{bi:04d}"
        log.info(f"=== Processing {tag}: n={len(batch)} ===")
        tB0 = perf_counter()

        batch_root = out_root / tag
        processed_dir = safe_mkdir(batch_root / "processed")
        fasta_dir = safe_mkdir(batch_root / "fastas")
        emb_dir = safe_mkdir(batch_root / "embeddings")
        anno_dir = safe_mkdir(batch_root / "annotations")

        timings = {"prep_s": 0.0, "esm_s": 0.0, "graphs_s": 0.0, "infer_s": 0.0}

        records: List[SeqRecord] = []
        skipped = 0
        for pdb_model in batch:
            try:
                _, recA, recB = process_one_pdb_no_esm(pdb_model, heavy, light, antigen, processed_dir, fasta_dir, timings)
                if recA: records.append(recA)
                if recB: records.append(recB)
            except Exception as e:
                skipped += 1
                log.warning(f"[SKIP] {pdb_model.name}: {e}")

        log.info(f"[TIME] {tag} PREP: {timings['prep_s']:.3f}s (skipped {skipped})")

        write_manifest_tsv(records, emb_dir / "manifest.tsv")
        shards, label_to_shard = shard_records_balanced(records, n_shards=esm_gpus)
        shard_tsvs = []
        for i, shard_recs in enumerate(shards):
            stsv = emb_dir / f"shard{i}.tsv"
            write_shard_tsv(shard_recs, stsv)
            shard_tsvs.append(stsv)

        part_h5s, esm_crit = run_sharded_embeddings_subprocess(
            embeddings_dir=emb_dir,
            n_gpus=esm_gpus,
            toks_per_batch=esm_toks_per_batch,
            scalar_dtype=esm_scalar_dtype,
            python_exe=python_exe,
            script_path=script_path,
            label_to_shard=label_to_shard,
        )
        timings["esm_s"] = float(esm_crit)
        log.info(f"[TIME] {tag} ESM (crit): {timings['esm_s']:.3f}s")

        annotate_folder_one_by_one_mp(
            processed_dir,
            fasta_dir,
            output_dir=str(anno_dir),
            n_cores=num_cores,
            antigen_chainid=antigen_chainid_for_graph,
        )
        region_json = anno_dir / "annotations_cdrs.json"
        correct_region_json(region_json)

        tG0 = perf_counter()
        graph_hdf5 = out_root / f"graphs_{tag}.hdf5"
        gen_graphs(processed_dir, graph_hdf5, region_json, num_cores, antigen_chainid_for_graph, tmp_base)

        inject_embeddings_from_parts(graph_hdf5, part_h5s, emb_dir / "label_to_shard.json")
        cluster_mcl(graph_hdf5)
        timings["graphs_s"] = perf_counter() - tG0
        log.info(f"[TIME] {tag} GRAPHS+INJECT+CLUSTER: {timings['graphs_s']:.3f}s")

        tI0 = perf_counter()
        pred_hdf5 = out_root / f"pred_{tag}.hdf5"
        predict(graph_hdf5, model_path, pred_hdf5, device, dl_workers, batch_size, num_cores)
        timings["infer_s"] = perf_counter() - tI0
        log.info(f"[TIME] {tag} INFER: {timings['infer_s']:.3f}s")

        pred_files.append(pred_hdf5)
        log.info(f"[DONE] {tag} → {pred_hdf5}")

        if not keep_intermediates:
            shutil.rmtree(batch_root, ignore_errors=True)

        log.info(f"[TIME] {tag} TOTAL: {(perf_counter() - tB0):.3f}s")

    if not keep_intermediates:
        shutil.rmtree(staging, ignore_errors=True)

    return pred_files


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--esm-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard-tsv", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--report-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--done-sentinel", default=None, help=argparse.SUPPRESS)

    ap.add_argument("--esm-toks-per-batch", type=int, default=DEFAULT_TOKS_PER_BATCH)
    ap.add_argument("--esm-scalar-dtype", default=os.environ.get("ESM_SCALAR_DTYPE", "float16"))
    ap.add_argument("--esm-gpus", type=int, default=int(os.environ.get("ESM_GPUS", "4")))

    ap.add_argument("--pdb-folder", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--model-path", required=False)
    ap.add_argument("--heavy", required=False)
    ap.add_argument("--light", default="-")
    ap.add_argument("--antigen", required=False)
    ap.add_argument("--antigen-chainid-for-graph", default="B")
    ap.add_argument("--num-cores", type=int, default=int(os.environ.get("NUM_CORES", "32")))
    ap.add_argument("--graph-batch-size", type=int, default=int(os.environ.get("GRAPH_BATCH_SIZE", "500")))
    ap.add_argument("--dl-workers", type=int, default=int(os.environ.get("DL_WORKERS", "8")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")))
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--tmp-base", default=os.environ.get("TMPDIR", None))
    ap.add_argument("--keep-intermediates", action="store_true")

    args = ap.parse_args()

    if args.esm_worker:
        if not (args.shard_tsv and args.out_h5 and args.report_json and args.done_sentinel):
            raise SystemExit("esm-worker mode requires --shard-tsv, --out-h5, --report-json, --done-sentinel")
        run_esm_worker(
            shard_tsv=Path(args.shard_tsv),
            out_h5=Path(args.out_h5),
            report_json=Path(args.report_json),
            done_sentinel=Path(args.done_sentinel),
            device=args.device,
            toks_per_batch=int(args.esm_toks_per_batch),
            scalar_dtype=str(args.esm_scalar_dtype),
        )
        return

    if not (args.pdb_folder and args.out and args.model_path and args.heavy and args.antigen):
        raise SystemExit("Missing required args: --pdb-folder, --out, --model-path, --heavy, --antigen")

    tmp_base = Path(args.tmp_base) if args.tmp_base else None

    preds = run_batched_inference(
        pdb_folder=Path(args.pdb_folder),
        out_root=Path(args.out),
        model_path=Path(args.model_path),
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
        esm_gpus=int(args.esm_gpus),
        esm_toks_per_batch=int(args.esm_toks_per_batch),
        esm_scalar_dtype=str(args.esm_scalar_dtype),
    )

    log.info("All batches complete.")
    for p in preds:
        log.info(f"  {p}")


if __name__ == "__main__":
    main()
