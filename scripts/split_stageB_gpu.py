#!/usr/bin/env python3
from __future__ import annotations

"""
split_stageB_gpu.py

Stage B (GPU): process shards produced by Stage A CPU using batch ESM.

When processing multiple shards (--start-index/--count), sequences from all
pending shards are merged and ESM embeddings are computed in a single pass
across all GPUs. This reduces ESM model loads from S*G to G (for S shards
on G GPUs). Per-shard inject + infer + publish then runs sequentially,
preserving per-shard DONE sentinels for spot instance resilience.

Three-phase flow:
  Phase 0: Filter targets — skip shards already DONE or missing STAGEA_DONE
  Phase 1: Collect sequences from all pending shards (deduplicate by label)
  Phase 2: Batch ESM — one call to run_sharded_embeddings_subprocess()
  Phase 3: Per-shard inject + infer + publish with DONE checkpoint per shard

Spot instance resilience:
  - Preempted during Phase 2: all ESM work lost, but ran only once not per-shard
  - Preempted during Phase 3: completed shards have DONE sentinels; on restart
    is_done() skips them. Worst case = 1 shard's inject+infer lost (~seconds)

Assumed Stage A layout:
$RUN_ROOT/shards/shard_000000/
  graphs.h5
  manifest.tsv.gz
  meta_stageA.json
  STAGEA_DONE
"""

import argparse
import dataclasses as dc
import gzip
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import h5py
import torch

# ----------------------------
# DeepRank-Ab imports (repo-relative)
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# IMPORTANT PATH FIX:
# DeepRank-Ab code sometimes imports "tools.*" (e.g. tools.FocalLoss) even though
# the repo layout is src/tools/*. To make "tools" resolvable, we must add
# ROOT_DIR/src to sys.path, not just ROOT_DIR.
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))   # enables import tools.* -> src/tools/*
sys.path.insert(0, str(ROOT_DIR))  # enables import src.* -> <repo>/src/*

from src.NeuralNet_focal_EMA import NeuralNet  # noqa: E402
from src.EGNN import egnn  # noqa: E402

# ESM
from esm import FastaBatchedDataset, pretrained  # noqa: E402


# ----------------------------
# Logging
# ----------------------------
log = logging.getLogger("drab-stageB")
log.setLevel(logging.INFO)
_hdl = logging.StreamHandler()
_hdl.setFormatter(logging.Formatter(" [%(levelname)s] %(message)s"))
log.addHandler(_hdl)

# ----------------------------
# Model feature schema
# ----------------------------
NODE_FEATURES = ["atom_type", "polarity", "bsa", "region", "embedding"]
EDGE_FEATURES = ["voro_area", "covalent", "vdw", "orientation"]

# ----------------------------
# ESM configuration
# ----------------------------
DEFAULT_TOKS_PER_BATCH = int(os.environ.get("ESM_TOKS_PER_BATCH", "12288"))
TRUNCATION_SEQ_LENGTH = 2500
ESM_MODEL = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")
REPR_LAYERS = [33]


# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_copy(src: Path, dst: Path) -> None:
    safe_mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2) + "\n")


def _write_progress_B(
    preds_root: Path,
    shard_id: str,
    stage: str,
    n_sequences: int,
    started_at: str,
) -> None:
    atomic_write_json(preds_root / f"progress_shard_{shard_id}.json", {
        "shard_id": shard_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "stage": stage,
        "n_sequences": n_sequences,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })


def _write_progress_B_batch(
    preds_root: Path,
    stage: str,
    shard_ids: List[str],
    n_total_sequences: int,
    started_at: str,
    current_shard: str = "",
) -> None:
    atomic_write_json(preds_root / "progress_batch_esm.json", {
        "shard_ids": shard_ids,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "stage": stage,
        "n_total_sequences": n_total_sequences,
        "current_shard": current_shard,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


# ----------------------------
# Data model
# ----------------------------
@dc.dataclass
class SeqRecord:
    label: str  # required: "<mol>.<chain>" where chain is A or B
    length: int
    sequence: str


@dc.dataclass
class ShardResult:
    shard_id: str
    status: str  # ok | skipped | failed
    wall_s: float
    timings: Dict[str, float] = dc.field(default_factory=dict)
    notes: List[str] = dc.field(default_factory=list)
    error: Optional[str] = None


@dc.dataclass
class ShardSeqBatch:
    """Merged sequences from multiple data shards for batch ESM processing."""
    all_records: List[SeqRecord]
    shard_to_records: Dict[str, List[SeqRecord]]
    shard_to_count: Dict[str, int]
    shard_ids: List[str]


# ----------------------------
# Shard discovery + idempotency
# ----------------------------
def list_shards(shards_root: Path) -> List[str]:
    dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])
    out: List[str] = []
    for d in dirs:
        sid = d.name.replace("shard_", "")
        if sid:
            out.append(sid)
    return out


def stageA_done_path(shards_root: Path, shard_id: str) -> Path:
    return shards_root / f"shard_{shard_id}" / "STAGEA_DONE"


def stageB_paths(preds_root: Path, shard_id: str) -> Tuple[Path, Path, Path, Path]:
    pred = preds_root / f"pred_shard_{shard_id}.h5"
    done = preds_root / f"DONE_shard_{shard_id}.ok"
    failed = preds_root / f"FAILED_shard_{shard_id}.txt"
    meta = preds_root / f"meta_stageB_shard_{shard_id}.json"
    return pred, done, failed, meta


def is_done(preds_root: Path, shard_id: str) -> bool:
    pred, done, _failed, _meta = stageB_paths(preds_root, shard_id)
    return pred.is_file() and done.is_file()


# ----------------------------
# Manifest parsing (gz supported)
# ----------------------------
def _validate_label(label: str, path: Path) -> None:
    if "." not in label:
        raise RuntimeError(f"{path}: label '{label}' missing '.' (need <mol>.<chain>)")
    mol, chain = label.rsplit(".", 1)
    if chain not in ("A", "B"):
        raise RuntimeError(f"{path}: label '{label}' chain must be A or B (got '{chain}')")
    if not mol:
        raise RuntimeError(f"{path}: label '{label}' has empty mol")


def read_manifest_tsv(path: Path) -> List[SeqRecord]:
    """
    Supports either:
      (1) label, length, sequence
      (2) label, mol, chain, length, sequence
    and works for .tsv or .tsv.gz
    """
    opener = gzip.open if path.suffix == ".gz" else open

    records: List[SeqRecord] = []
    with opener(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h.strip(): i for i, h in enumerate(header)}

        if "label" not in idx:
            raise RuntimeError(f"{path}: manifest missing 'label' column. header={header}")
        if "sequence" not in idx:
            raise RuntimeError(f"{path}: manifest missing 'sequence' column. header={header}")

        i_label = idx["label"]
        i_seq = idx["sequence"]
        i_len = idx.get("length", -1)

        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) <= max(i_label, i_seq, i_len if i_len >= 0 else 0):
                continue

            label = parts[i_label].strip()
            seq = parts[i_seq].strip()
            if not label or not seq:
                continue

            _validate_label(label, path)

            length = len(seq)
            if i_len >= 0:
                try:
                    length = int(parts[i_len].strip())
                except Exception:
                    length = len(seq)

            records.append(SeqRecord(label=label, length=length, sequence=seq))

    return records


def load_sequences_for_shard(shard_dir: Path) -> List[SeqRecord]:
    m = shard_dir / "manifest.tsv.gz"
    if not m.is_file():
        m2 = shard_dir / "manifest.tsv"
        if m2.is_file():
            return read_manifest_tsv(m2)
        raise FileNotFoundError(f"{shard_dir}: missing manifest.tsv.gz")
    return read_manifest_tsv(m)


def collect_sequences_for_shards(
    shards_root: Path,
    shard_ids: List[str],
) -> ShardSeqBatch:
    """Load manifests from all pending shards and merge into one ShardSeqBatch.

    Deduplicates by label (keeps first occurrence). This is unlikely with
    properly partitioned shards but safe.
    """
    shard_to_records: Dict[str, List[SeqRecord]] = {}
    shard_to_count: Dict[str, int] = {}
    seen_labels: set = set()
    all_records: List[SeqRecord] = []

    for sid in shard_ids:
        shard_dir = shards_root / f"shard_{sid}"
        records = load_sequences_for_shard(shard_dir)
        shard_records: List[SeqRecord] = []
        for r in records:
            if r.label not in seen_labels:
                seen_labels.add(r.label)
                all_records.append(r)
                shard_records.append(r)
        shard_to_records[sid] = shard_records
        shard_to_count[sid] = len(shard_records)
        log.info(f"  shard {sid}: {len(records)} sequences ({len(shard_records)} unique)")

    return ShardSeqBatch(
        all_records=all_records,
        shard_to_records=shard_to_records,
        shard_to_count=shard_to_count,
        shard_ids=list(shard_ids),
    )


# ----------------------------
# ESM sharding
# ----------------------------
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
        for ln in fin:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            label, _len, seq = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not label or not seq:
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
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                label, length_s, _seq = ln.split("\t", 2)
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
                    reps_i = reps[i, 1 : trunc + 1].detach()  # (L, D)
                    scalar = reps_i.mean(dim=1)               # (L,)
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
    records: List[SeqRecord],
    n_gpus: int,
    toks_per_batch: int,
    scalar_dtype: str,
    python_exe: str,
    script_path: Path,
) -> Tuple[List[Path], float, Dict[str, int]]:
    part_h5s: List[Path] = []
    shard_tsvs: List[Path] = []
    reports: List[Path] = []
    sentinels: List[Path] = []

    shards, label_to_shard = shard_records_balanced(records, n_shards=n_gpus)

    for i in range(n_gpus):
        shard_tsv = embeddings_dir / f"shard{i}.tsv"
        out_h5 = embeddings_dir / f"emb_part{i}.h5"
        report_json = embeddings_dir / f"emb_part{i}.report.json"
        done = embeddings_dir / f"embed_done.part{i}"
        shard_tsvs.append(shard_tsv)
        part_h5s.append(out_h5)
        reports.append(report_json)
        sentinels.append(done)

        for p in (shard_tsv, out_h5, report_json, done, out_h5.with_suffix(".fasta")):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        write_shard_tsv(shards[i], shard_tsv)

    (embeddings_dir / "label_to_shard.json").write_text(json.dumps(label_to_shard, indent=2))

    procs: List[subprocess.Popen] = []
    t0 = perf_counter()

    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        args = [
            python_exe,
            str(script_path),
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
    return part_h5s, float(crit if crit > 0 else (perf_counter() - t0)), label_to_shard


# ----------------------------
# Embedding injection
# ----------------------------
def inject_embeddings_from_parts(graph_h5: Path, part_h5s: List[Path], label_to_shard: Dict[str, int]) -> Tuple[int, int]:
    """
    Inject ESM embeddings into graph HDF5.

    Optimized version: pre-loads all embeddings into memory at startup for
    vectorized access, avoiding repeated HDF5 lookups per residue.
    """
    import numpy as np

    # Pre-load all embeddings into a single dict (label -> numpy array)
    # This avoids repeated HDF5 lookups and enables vectorized injection
    all_embeddings: Dict[str, np.ndarray] = {}

    for part_h5 in part_h5s:
        with h5py.File(part_h5, "r") as h:
            if "scalar" not in h:
                continue
            scalar_group = h["scalar"]
            for label in scalar_group.keys():
                all_embeddings[label] = scalar_group[label][()]

    missing_labels = 0
    oob = 0

    with h5py.File(graph_h5, "r+") as f:
        for mol in f.keys():
            residues = f[mol]["nodes"][()]
            n_residues = len(residues)
            emb = np.zeros((n_residues, 1), dtype=np.float32)

            # Decode chain and resid arrays once (vectorized where possible)
            chains = []
            resids = []
            for res in residues:
                chain = res[0].decode() if isinstance(res[0], (bytes, bytearray)) else str(res[0])
                resid = int(res[1].decode()) if isinstance(res[1], (bytes, bytearray)) else int(res[1])
                chains.append(chain)
                resids.append(resid)

            # Group residues by chain for batch lookup
            chain_to_indices: Dict[str, List[Tuple[int, int]]] = {}
            for i, (chain, resid) in enumerate(zip(chains, resids)):
                chain_to_indices.setdefault(chain, []).append((i, resid))

            # Inject embeddings per chain (vectorized)
            for chain, idx_resid_list in chain_to_indices.items():
                label = f"{mol}.{chain}"
                if label not in all_embeddings:
                    missing_labels += len(idx_resid_list)
                    continue

                arr = all_embeddings[label]
                arr_len = len(arr)

                for i, resid in idx_resid_list:
                    j = resid - 1
                    if 0 <= j < arr_len:
                        emb[i, 0] = float(arr[j])
                    else:
                        oob += 1

            grp = f[mol].require_group("node_data")
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("embedding", data=emb)

    return missing_labels, oob


# ----------------------------
# DeepRank-Ab inference
# ----------------------------
def predict(graph_h5: Path, model_path: Path, out_pred_h5: Path, device: str, dl_workers: int, batch_size: int, num_cores: int, prefetch_factor: int = 4) -> None:
    if dl_workers < 0:
        dl_workers = 0
    if dl_workers > num_cores:
        dl_workers = num_cores

    net = NeuralNet(
        database=str(graph_h5),
        Net=egnn,
        node_feature=NODE_FEATURES,
        edge_feature=EDGE_FEATURES,
        target=None,
        task="reg",
        batch_size=batch_size,
        num_workers=dl_workers,
        prefetch_factor=prefetch_factor,
        device_name=device,
        shuffle=False,
        pretrained_model=str(model_path),
        cluster_nodes="mcl",
    )
    net.predict(database_test=str(graph_h5), hdf5=str(out_pred_h5))


# ----------------------------
# Core per-shard worker
# ----------------------------
def process_one_shard(
    run_root: Path,
    shards_root: Path,
    preds_root: Path,
    logs_root: Path,
    shard_id: str,

    model_path: Path,
    device: str,
    num_cores: int,
    dl_workers: int,
    batch_size: int,
    prefetch_factor: int,
    esm_gpus: int,
    esm_toks_per_batch: int,
    esm_scalar_dtype: str,
    local_base: Path,
) -> ShardResult:
    t0 = perf_counter()
    timings: Dict[str, float] = {}

    shard_dir = shards_root / f"shard_{shard_id}"
    if not shard_dir.is_dir():
        return ShardResult(shard_id=shard_id, status="failed", wall_s=0.0, error=f"missing shard dir: {shard_dir}")

    sdone = shard_dir / "STAGEA_DONE"
    if not sdone.is_file():
        return ShardResult(shard_id=shard_id, status="failed", wall_s=0.0, error=f"missing STAGEA_DONE: {sdone}")

    pred_pub, done_pub, failed_pub, meta_pub = stageB_paths(preds_root, shard_id)

    if is_done(preds_root, shard_id):
        return ShardResult(shard_id=shard_id, status="skipped", wall_s=perf_counter() - t0, notes=["already_done"])

    if failed_pub.exists():
        try:
            failed_pub.unlink()
        except Exception:
            pass
    if done_pub.exists():
        try:
            done_pub.unlink()
        except Exception:
            pass

    started_at = datetime.now().isoformat(timespec="seconds")
    _bprog = dict(preds_root=preds_root, shard_id=shard_id, started_at=started_at, n_sequences=0)

    workdir = safe_mkdir(local_base / f"stageB_shard_{shard_id}")
    emb_dir = safe_mkdir(workdir / "embeddings")

    graph_src = shard_dir / "graphs.h5"
    if not graph_src.is_file():
        raise FileNotFoundError(f"{shard_dir}: missing graphs.h5")

    graph_local = workdir / "graphs.h5"
    t_copy0 = perf_counter()
    shutil.copy2(graph_src, graph_local)
    timings["stage_graph_s"] = perf_counter() - t_copy0
    _write_progress_B(stage="copy", **_bprog)

    t_seq0 = perf_counter()
    records = load_sequences_for_shard(shard_dir)
    if not records:
        raise RuntimeError(f"{shard_dir}: no sequences loaded (manifest empty?)")
    timings["load_sequences_s"] = perf_counter() - t_seq0
    _bprog["n_sequences"] = len(records)
    _write_progress_B(stage="load_seqs", **_bprog)

    _write_progress_B(stage="esm", **_bprog)
    t_esm0 = perf_counter()
    part_h5s, esm_crit_s, label_to_shard = run_sharded_embeddings_subprocess(
        embeddings_dir=emb_dir,
        records=records,
        n_gpus=esm_gpus,
        toks_per_batch=esm_toks_per_batch,
        scalar_dtype=esm_scalar_dtype,
        python_exe=sys.executable,
        script_path=Path(__file__).resolve(),
    )
    timings["esm_crit_s"] = float(esm_crit_s)
    timings["esm_total_s"] = perf_counter() - t_esm0

    _write_progress_B(stage="inject", **_bprog)
    t_inj0 = perf_counter()
    missing_labels, oob = inject_embeddings_from_parts(graph_local, part_h5s, label_to_shard)
    timings["inject_s"] = perf_counter() - t_inj0
    if missing_labels:
        log.warning(f"[{shard_id}] injection: missing_labels={missing_labels}")
    if oob:
        log.warning(f"[{shard_id}] injection: oob_residue_indices={oob}")

    _write_progress_B(stage="infer", **_bprog)
    t_inf0 = perf_counter()
    pred_local = workdir / "pred.h5"
    predict(
        graph_h5=graph_local,
        model_path=model_path,
        out_pred_h5=pred_local,
        device=device,
        dl_workers=dl_workers,
        batch_size=batch_size,
        num_cores=num_cores,
        prefetch_factor=prefetch_factor,
    )
    timings["infer_s"] = perf_counter() - t_inf0

    _write_progress_B(stage="publish", **_bprog)
    t_pub0 = perf_counter()
    atomic_copy(pred_local, pred_pub)
    atomic_write_text(done_pub, "ok\n")
    timings["publish_s"] = perf_counter() - t_pub0

    meta = {
        "run_root": str(run_root),
        "shard_id": shard_id,
        "graph_src": str(graph_src),
        "pred_path": str(pred_pub),
        "done_path": str(done_pub),
        "model_path": str(model_path),
        "device": device,
        "esm_model": ESM_MODEL,
        "esm_gpus": int(esm_gpus),
        "esm_toks_per_batch": int(esm_toks_per_batch),
        "esm_scalar_dtype": str(esm_scalar_dtype),
        "missing_labels": int(missing_labels),
        "oob_residue_indices": int(oob),
        "timings": timings,
        "generated_at": datetime.now().isoformat(),
    }
    atomic_write_text(meta_pub, json.dumps(meta, indent=2) + "\n")
    _write_progress_B(stage="done", **_bprog)

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    return ShardResult(shard_id=shard_id, status="ok", wall_s=perf_counter() - t0, timings=timings)


def process_shard_post_esm(
    run_root: Path,
    shards_root: Path,
    preds_root: Path,
    shard_id: str,
    part_h5s: List[Path],
    label_to_shard: Dict[str, int],
    model_path: Path,
    device: str,
    num_cores: int,
    dl_workers: int,
    batch_size: int,
    prefetch_factor: int,
    local_base: Path,
    started_at: str,
    n_sequences: int,
) -> ShardResult:
    """Per-shard post-ESM work: copy graph -> inject embeddings -> infer -> publish."""
    t0 = perf_counter()
    timings: Dict[str, float] = {}

    shard_dir = shards_root / f"shard_{shard_id}"
    pred_pub, done_pub, failed_pub, meta_pub = stageB_paths(preds_root, shard_id)

    # Clean up stale sentinels
    for p in (failed_pub, done_pub):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    _bprog = dict(preds_root=preds_root, shard_id=shard_id, started_at=started_at, n_sequences=n_sequences)

    workdir = safe_mkdir(local_base / f"stageB_shard_{shard_id}")

    graph_src = shard_dir / "graphs.h5"
    if not graph_src.is_file():
        raise FileNotFoundError(f"{shard_dir}: missing graphs.h5")

    # Copy graph locally
    graph_local = workdir / "graphs.h5"
    t_copy0 = perf_counter()
    shutil.copy2(graph_src, graph_local)
    timings["stage_graph_s"] = perf_counter() - t_copy0
    _write_progress_B(stage="inject", **_bprog)

    # Inject embeddings from consolidated part_h5s
    t_inj0 = perf_counter()
    missing_labels, oob = inject_embeddings_from_parts(graph_local, part_h5s, label_to_shard)
    timings["inject_s"] = perf_counter() - t_inj0
    if missing_labels:
        log.warning(f"[{shard_id}] injection: missing_labels={missing_labels}")
    if oob:
        log.warning(f"[{shard_id}] injection: oob_residue_indices={oob}")

    # Inference
    _write_progress_B(stage="infer", **_bprog)
    t_inf0 = perf_counter()
    pred_local = workdir / "pred.h5"
    predict(
        graph_h5=graph_local,
        model_path=model_path,
        out_pred_h5=pred_local,
        device=device,
        dl_workers=dl_workers,
        batch_size=batch_size,
        num_cores=num_cores,
        prefetch_factor=prefetch_factor,
    )
    timings["infer_s"] = perf_counter() - t_inf0

    # Publish
    _write_progress_B(stage="publish", **_bprog)
    t_pub0 = perf_counter()
    atomic_copy(pred_local, pred_pub)
    atomic_write_text(done_pub, "ok\n")
    timings["publish_s"] = perf_counter() - t_pub0

    meta = {
        "run_root": str(run_root),
        "shard_id": shard_id,
        "graph_src": str(graph_src),
        "pred_path": str(pred_pub),
        "done_path": str(done_pub),
        "model_path": str(model_path),
        "device": device,
        "esm_model": ESM_MODEL,
        "esm_batch_mode": True,
        "missing_labels": int(missing_labels),
        "oob_residue_indices": int(oob),
        "timings": timings,
        "generated_at": datetime.now().isoformat(),
    }
    atomic_write_text(meta_pub, json.dumps(meta, indent=2) + "\n")
    _write_progress_B(stage="done", **_bprog)

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    return ShardResult(shard_id=shard_id, status="ok", wall_s=perf_counter() - t0, timings=timings)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--esm-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard-tsv", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--report-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--done-sentinel", default=None, help=argparse.SUPPRESS)

    ap.add_argument("--esm-toks-per-batch", type=int, default=DEFAULT_TOKS_PER_BATCH)
    ap.add_argument("--esm-scalar-dtype", default=os.environ.get("ESM_SCALAR_DTYPE", "float16"))
    ap.add_argument("--esm-gpus", type=int, default=int(os.environ.get("ESM_GPUS", "4")))

    ap.add_argument("--run-root", required=False, default=".", help="Run directory")
    ap.add_argument("--model-path", required=False, help="Path to DeepRank-Ab pretrained model")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--num-cores", type=int, default=getenv_int("NUM_CORES", 32))
    ap.add_argument("--dl-workers", type=int, default=getenv_int("DL_WORKERS", 8))
    ap.add_argument("--batch-size", type=int, default=getenv_int("BATCH_SIZE", 64))
    ap.add_argument("--prefetch-factor", type=int, default=getenv_int("PREFETCH_FACTOR", 4))

    ap.add_argument("--shard-id", default="", help="Process a single shard id (e.g. 000000)")
    ap.add_argument("--start-index", type=int, default=-1, help="Start shard index (0-based)")
    ap.add_argument("--count", type=int, default=0, help="How many shards to process from start-index")

    ap.add_argument("--shards-dir", default="", help="Override shards root (default: <run-root>/shards)")
    ap.add_argument("--preds-dir", default="", help="Override preds root (default: <run-root>/preds)")
    ap.add_argument("--logs-dir", default="", help="Override logs root (default: <run-root>/logs)")
    ap.add_argument("--local-base", default="", help="Override local temp base (default: $SLURM_TMPDIR or $TMPDIR or /tmp/<user>)")

    args = ap.parse_args()

    if args.esm_worker:
        if not (args.shard_tsv and args.out_h5 and args.report_json and args.done_sentinel):
            print("esm-worker mode requires --shard-tsv, --out-h5, --report-json, --done-sentinel", file=sys.stderr)
            return 2
        run_esm_worker(
            shard_tsv=Path(args.shard_tsv),
            out_h5=Path(args.out_h5),
            report_json=Path(args.report_json),
            done_sentinel=Path(args.done_sentinel),
            device=args.device,
            toks_per_batch=int(args.esm_toks_per_batch),
            scalar_dtype=str(args.esm_scalar_dtype),
        )
        return 0

    run_root = Path(args.run_root).resolve()
    if not args.model_path:
        print("ERROR: --model-path is required (stageB driver mode)", file=sys.stderr)
        return 2
    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    shards_root = Path(args.shards_dir).resolve() if args.shards_dir else (run_root / "shards")
    preds_root = Path(args.preds_dir).resolve() if args.preds_dir else (run_root / "preds")
    logs_root = Path(args.logs_dir).resolve() if args.logs_dir else (run_root / "logs")
    safe_mkdir(preds_root)
    safe_mkdir(logs_root)

    all_shards = list_shards(shards_root)
    if not all_shards:
        print(f"ERROR: no shards found in {shards_root}", file=sys.stderr)
        return 2

    if args.shard_id:
        targets = [args.shard_id]
    elif args.start_index >= 0 and args.count > 0:
        start = args.start_index
        end = min(start + args.count, len(all_shards))
        if start >= len(all_shards):
            print(f"[StageB] start-index {start} >= N_SHARDS {len(all_shards)}; nothing to do.")
            return 0
        targets = all_shards[start:end]
    else:
        print("ERROR: provide --shard-id OR (--start-index and --count)", file=sys.stderr)
        return 2

    if args.local_base:
        local_base = Path(args.local_base).resolve()
    else:
        base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        if base:
            local_base = Path(base).resolve() / f"drab_stageB_{os.environ.get('SLURM_JOB_ID','nojid')}_{os.environ.get('SLURM_ARRAY_TASK_ID','na')}"
        else:
            local_base = Path("/tmp") / os.environ.get("USER", "user") / f"drab_stageB_{os.environ.get('SLURM_JOB_ID','nojid')}_{os.environ.get('SLURM_ARRAY_TASK_ID','na')}"
    safe_mkdir(local_base)

    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "na")

    log.info(f"RUN_ROOT:   {run_root}")
    log.info(f"SHARDS:     {shards_root} (N={len(all_shards)})")
    log.info(f"PREDS:      {preds_root}")
    log.info(f"LOGS:       {logs_root}")
    log.info(f"MODEL:      {model_path}")
    log.info(f"TARGETS:    {len(targets)} shards (first={targets[0]} last={targets[-1]})")
    log.info(f"LOCAL_BASE: {local_base}")

    esm_gpus = int(args.esm_gpus)
    esm_toks = int(args.esm_toks_per_batch)
    esm_dtype = str(args.esm_scalar_dtype)

    t_all0 = perf_counter()
    results: List[ShardResult] = []

    # ------------------------------------------------------------------
    # PHASE 0: Filter targets — skip shards already DONE or missing STAGEA_DONE
    # ------------------------------------------------------------------
    pending_sids: List[str] = []
    for sid in targets:
        if not stageA_done_path(shards_root, sid).is_file():
            results.append(ShardResult(shard_id=sid, status="failed", wall_s=0.0, error="missing STAGEA_DONE"))
            log.error(f"[{sid}] missing STAGEA_DONE -> fail")
            continue

        if is_done(preds_root, sid):
            results.append(ShardResult(shard_id=sid, status="skipped", wall_s=0.0, notes=["already_done"]))
            continue

        pending_sids.append(sid)

    if not pending_sids:
        log.info("All target shards already done or missing STAGEA_DONE. Nothing to do.")
    else:
        log.info(f"PENDING:    {len(pending_sids)} shards: {pending_sids}")
        started_at = datetime.now().isoformat(timespec="seconds")
        _batch_prog = dict(preds_root=preds_root, shard_ids=pending_sids,
                           n_total_sequences=0, started_at=started_at)

        # ------------------------------------------------------------------
        # PHASE 1: Collect sequences from all pending shards
        # ------------------------------------------------------------------
        log.info("--- Phase 1: collecting sequences ---")
        _write_progress_B_batch(stage="collect_seqs", **_batch_prog)
        t_collect0 = perf_counter()
        seq_batch = collect_sequences_for_shards(shards_root, pending_sids)
        t_collect = perf_counter() - t_collect0
        _batch_prog["n_total_sequences"] = len(seq_batch.all_records)
        log.info(f"  total unique sequences: {len(seq_batch.all_records)} (collected in {t_collect:.1f}s)")

        if not seq_batch.all_records:
            log.warning("No sequences found across pending shards — skipping ESM.")
            for sid in pending_sids:
                results.append(ShardResult(shard_id=sid, status="failed", wall_s=0.0, error="no sequences in manifest"))
        else:
            # ------------------------------------------------------------------
            # PHASE 2: Batch ESM — ONCE for all shards
            # ------------------------------------------------------------------
            log.info("--- Phase 2: batch ESM embeddings ---")
            _write_progress_B_batch(stage="esm", **_batch_prog)

            emb_dir = safe_mkdir(local_base / "batch_embeddings")
            t_esm0 = perf_counter()
            part_h5s, esm_crit_s, label_to_shard = run_sharded_embeddings_subprocess(
                embeddings_dir=emb_dir,
                records=seq_batch.all_records,
                n_gpus=esm_gpus,
                toks_per_batch=esm_toks,
                scalar_dtype=esm_dtype,
                python_exe=sys.executable,
                script_path=Path(__file__).resolve(),
            )
            t_esm_total = perf_counter() - t_esm0
            log.info(f"  ESM done: crit={esm_crit_s:.1f}s total={t_esm_total:.1f}s")

            # ------------------------------------------------------------------
            # PHASE 3: Per-shard inject + infer + publish
            # ------------------------------------------------------------------
            log.info("--- Phase 3: per-shard inject + infer + publish ---")
            _write_progress_B_batch(stage="per_shard", **_batch_prog)

            for sid in pending_sids:
                n_seq = seq_batch.shard_to_count.get(sid, 0)
                log.info(f"=== SHARD {sid} ({n_seq} seqs) ===")
                _write_progress_B_batch(stage="per_shard", current_shard=sid, **_batch_prog)
                try:
                    r = process_shard_post_esm(
                        run_root=run_root,
                        shards_root=shards_root,
                        preds_root=preds_root,
                        shard_id=sid,
                        part_h5s=part_h5s,
                        label_to_shard=label_to_shard,
                        model_path=model_path,
                        device=args.device,
                        num_cores=int(args.num_cores),
                        dl_workers=int(args.dl_workers),
                        batch_size=int(args.batch_size),
                        prefetch_factor=int(args.prefetch_factor),
                        local_base=local_base,
                        started_at=started_at,
                        n_sequences=n_seq,
                    )
                    results.append(r)
                    log.info(f"[{sid}] status={r.status} wall={r.wall_s:.1f}s")
                except Exception as e:
                    tb = traceback.format_exc()
                    _pred_pub, done_pub, failed_pub, _meta_pub = stageB_paths(preds_root, sid)
                    msg = f"FAILED shard {sid}: {e}\n\n{tb}\n"
                    try:
                        atomic_write_text(failed_pub, msg)
                    except Exception:
                        pass
                    if done_pub.exists():
                        try:
                            done_pub.unlink()
                        except Exception:
                            pass
                    results.append(ShardResult(shard_id=sid, status="failed", wall_s=0.0, error=str(e), notes=["see_FAILED_sentinel"]))
                    log.error(f"[{sid}] FAILED: {e}")

            _write_progress_B_batch(stage="done", **_batch_prog)

    wall_all = perf_counter() - t_all0

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "run_root": str(run_root),
        "job_id": job_id,
        "array_task_id": task_id,
        "device": args.device,
        "model_path": str(model_path),
        "esm_model": ESM_MODEL,
        "esm_gpus": esm_gpus,
        "esm_toks_per_batch": esm_toks,
        "esm_scalar_dtype": esm_dtype,
        "num_cores": int(args.num_cores),
        "dl_workers": int(args.dl_workers),
        "batch_size": int(args.batch_size),
        "targets": targets,
        "n_targets": len(targets),
        "n_ok": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "wall_total_s": float(wall_all),
        "results": [dc.asdict(r) for r in results],
    }

    out_json = logs_root / f"task_{job_id}_{task_id}_summary.json"
    atomic_write_text(out_json, json.dumps(summary, indent=2) + "\n")
    log.info(f"Wrote summary: {out_json}")

    try:
        shutil.rmtree(local_base, ignore_errors=True)
    except Exception:
        pass

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

