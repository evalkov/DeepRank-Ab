#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import h5py

# DeepRank-Ab imports (assumes this file lives in DeepRank-Ab/scripts)
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import PPBuilder

from src.GraphGenMP import GraphHDF5
from src.DataSet import HDF5DataSet, PreCluster
from src.tools.annotate import annotate_folder_one_by_one_mp


# -----------------------
# Utilities
# -----------------------

def is_missing_chain(x: str) -> bool:
    return (x is None) or (str(x).strip() == "") or (str(x).strip() == "-") or (str(x).strip().lower() == "none")

def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)

def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2) + "\n")

def gz_write_tsv(path: Path, header: str, rows: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f:
        f.write(header)
        for r in rows:
            f.write(r)
    tmp.replace(path)

def split_models_if_ensemble(pdb_path: Path, out_dir: Path) -> List[Path]:
    """
    FIX for your '_model_0' complaint:
    - If PDB has 1 model: return [original pdb_path] (no suffix).
    - If PDB has >1 model: write *_model_{id}.pdb files.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    models = list(struct)
    if len(models) <= 1:
        return [pdb_path]

    io = PDBIO()
    safe_mkdir(out_dir)
    saved: List[Path] = []
    for m in models:
        mid = m.id
        out = out_dir / f"{pdb_path.stem}_model_{mid}.pdb"
        io.set_structure(m)
        io.save(str(out))
        saved.append(out)
    return saved

def chain_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    """Legacy function - parses PDB from file. Use _chain_sequence_from_model for efficiency."""
    if is_missing_chain(chain_id):
        return ""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = struct[0]
    return _chain_sequence_from_model(model, chain_id)

def _chain_sequence_from_model(model, chain_id: str) -> str:
    """Extract sequence from an already-parsed model (avoids re-parsing PDB)."""
    if is_missing_chain(chain_id):
        return ""
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
    """
    Builds a 2-chain merged PDB:
      chain A = heavy + light (if present)
      chain B = antigen
    Residues renumbered from 1..N in each output chain.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model_in = struct[0]

    # Extract sequences from already-parsed model (avoids 3 redundant file parses)
    seqH = _chain_sequence_from_model(model_in, heavy_chain_id)
    seqL = _chain_sequence_from_model(model_in, light_chain_id)
    seqAg = _chain_sequence_from_model(model_in, antigen_chain_id)

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


def _prep_one_pdb(args: Tuple) -> Tuple[str, Optional[str], Optional[str], Optional[str], bool]:
    """
    Worker function for parallel PDB prep.
    Returns (stem, seqH, seqL, seqAg, success).
    """
    pdb_path, heavy, light, antigen, merged_dir, fasta_dir = args
    stem = pdb_path.stem
    out_pdb = merged_dir / f"{stem}.pdb"
    try:
        seqH, seqL, seqAg = build_merged_structure(pdb_path, heavy, light, antigen, out_pdb)
        # Write HL fasta for annotate
        fasta = fasta_dir / f"{stem}_HL.fasta"
        with open(fasta, "w") as f:
            if seqH:
                f.write(f">{stem}.H\n{seqH}\n")
            if seqL:
                f.write(f">{stem}.L\n{seqL}\n")
        return (stem, seqH, seqL, seqAg, True)
    except Exception:
        return (stem, None, None, None, False)


def correct_region_json(region_json: Path) -> None:
    # DeepRank-Ab annotate sometimes uses keys like "foo.pdb" – normalize to "foo"
    if not region_json.is_file():
        raise FileNotFoundError(f"Missing region JSON: {region_json}")
    data = json.loads(region_json.read_text())
    new = {k.replace(".pdb", ""): v for k, v in data.items()}
    atomic_write_json(region_json, new)

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

def ensure_zero_embedding(graph_h5: Path) -> None:
    """
    Create node_data/embedding zeros so the graph is self-contained.
    Stage B will overwrite embedding with real ESM scalars.
    """
    with h5py.File(graph_h5, "r+") as h:
        for mol in h.keys():
            nodes = h[mol]["nodes"][()]
            n = len(nodes)
            grp = h[mol].require_group("node_data")
            if "embedding" in grp:
                continue
            grp.create_dataset("embedding", data=[[0.0]] * n)

def cluster_mcl(graph_hdf5: Path) -> None:
    dataset = HDF5DataSet(name="EvalSet", root="./", database=str(graph_hdf5))
    PreCluster(dataset, method="mcl")


# -----------------------
# Sharding
# -----------------------

def build_shard_lists(
    pdb_root: Path,
    shard_lists_dir: Path,
    target_gb: float,
    max_per_shard: int,
    min_per_shard: int,
    glob_pat: str = "*.pdb",
) -> List[Path]:
    """
    Create shard_XXXX.lst under shard_lists_dir.

    Uses PDB file size as a *proxy* to keep shards near target_gb.
    It’s not perfect, but it’s stable, fast, and good enough to avoid pathological shard sizes.
    """
    safe_mkdir(shard_lists_dir)
    if "**" in glob_pat:
        pdbs = sorted(pdb_root.glob(glob_pat))   # recursive when glob contains **
    else:
        # default: search both top-level and one level down
        pdbs = sorted(pdb_root.glob(glob_pat))
        if not pdbs:
            pdbs = sorted(pdb_root.glob(f"**/{glob_pat}"))
    if not pdbs:
        raise SystemExit(f"No PDBs found in {pdb_root} ({glob_pat})")

    target_bytes = int(target_gb * (1024**3))
    shards: List[List[Path]] = []
    cur: List[Path] = []
    cur_bytes = 0

    def flush():
        nonlocal cur, cur_bytes
        if cur:
            shards.append(cur)
        cur = []
        cur_bytes = 0

    for p in pdbs:
        sz = p.stat().st_size
        if cur and (len(cur) >= max_per_shard or (cur_bytes + sz) > target_bytes) and len(cur) >= min_per_shard:
            flush()
        cur.append(p)
        cur_bytes += sz

    flush()

    out_lists: List[Path] = []
    for i, shard in enumerate(shards):
        out = shard_lists_dir / f"shard_{i:06d}.lst"
        tmp = out.with_suffix(".tmp")
        tmp.write_text("".join(str(x) + "\n" for x in shard))
        tmp.replace(out)
        out_lists.append(out)

    return out_lists


# -----------------------
# Stage A execution for one shard
# -----------------------

@dataclass
class StageAResult:
    shard_id: str
    n_inputs: int
    n_models: int
    n_ok: int
    n_fail: int
    prep_s: float
    annotate_s: float
    graphs_s: float
    cluster_s: float
    graphs_bytes: int

def run_stageA_one_shard(
    shard_id: str,
    shard_list: Path,
    exchange_shards_dir: Path,
    heavy: str,
    light: str,
    antigen: str,
    antigen_chainid_for_graph: str,
    num_cores: int,
    tmp_base: Optional[Path],
    do_cluster: bool,
) -> StageAResult:
    """
    Produces exchange/shards/shard_<id>/graphs.h5 + manifest + metadata + STAGEA_DONE
    """
    t0 = perf_counter()
    shard_dir = exchange_shards_dir / f"shard_{shard_id}"
    safe_mkdir(shard_dir)

    # Idempotency: if STAGEA_DONE exists, skip (safe for requeues)
    done = shard_dir / "STAGEA_DONE"
    if done.exists():
        meta = json.loads((shard_dir / "meta_stageA.json").read_text())
        return StageAResult(**meta["result"])

    local_root = Path(tempfile.mkdtemp(prefix=f"drab_stageA_{shard_id}_", dir=str(tmp_base) if tmp_base else None))
    pdbs_dir = safe_mkdir(local_root / "pdbs")
    merged_dir = safe_mkdir(local_root / "merged")
    fasta_dir = safe_mkdir(local_root / "fastas")
    anno_dir = safe_mkdir(local_root / "annotations")

    # Read input list
    inputs = [Path(x.strip()) for x in shard_list.read_text().splitlines() if x.strip()]
    if not inputs:
        raise SystemExit(f"{shard_list}: empty")

    # Stage PDBs to local (parallel I/O for network filesystems)
    def _copy_pdb(src: Path) -> Path:
        dst = pdbs_dir / src.name
        shutil.copy2(src, dst)
        return dst

    max_copy_workers = min(16, len(inputs)) if inputs else 1
    with ThreadPoolExecutor(max_workers=max_copy_workers) as ex:
        list(ex.map(_copy_pdb, inputs))

    staged = sorted(pdbs_dir.glob("*.pdb"))

    # Expand only if ensemble
    expanded: List[Path] = []
    for p in staged:
        expanded.extend(split_models_if_ensemble(p, local_root / "models" / p.stem))

    # PREP: build merged 2-chain PDBs + HL fasta (for annotate), and manifest for ESM later
    # Parallelized across cores for better throughput
    t1 = perf_counter()
    prep_args = [(p, heavy, light, antigen, merged_dir, fasta_dir) for p in expanded]
    max_prep_workers = min(num_cores, len(expanded)) if expanded else 1

    ok = 0
    fail = 0
    manifest_rows: List[str] = []

    with ProcessPoolExecutor(max_workers=max_prep_workers) as ex:
        for stem, seqH, seqL, seqAg, success in ex.map(_prep_one_pdb, prep_args):
            if success:
                # Manifest for ESM:
                seqA = (seqH or "") + (seqL or "")
                seqB = (seqAg or "")
                if seqA:
                    manifest_rows.append(f"{stem}.A\t{stem}\tA\t{len(seqA)}\t{seqA}\n")
                if seqB:
                    manifest_rows.append(f"{stem}.B\t{stem}\tB\t{len(seqB)}\t{seqB}\n")
                ok += 1
            else:
                fail += 1

    prep_s = perf_counter() - t1

    # Annotate (CPU)
    t2 = perf_counter()
    annotate_folder_one_by_one_mp(
        merged_dir,
        fasta_dir,
        output_dir=str(anno_dir),
        n_cores=num_cores,
        antigen_chainid=antigen_chainid_for_graph,
    )
    annotate_s = perf_counter() - t2

    region_json = anno_dir / "annotations_cdrs.json"
    correct_region_json(region_json)

    # Graphs (CPU)
    t3 = perf_counter()
    graphs_local = local_root / "graphs.h5"
    gen_graphs(merged_dir, graphs_local, region_json, num_cores, antigen_chainid_for_graph, tmp_base)
    ensure_zero_embedding(graphs_local)
    graphs_s = perf_counter() - t3

    # Cluster (CPU) — keep it here so GPU stage is truly “GPU-only”
    cluster_s = 0.0
    if do_cluster:
        t4 = perf_counter()
        cluster_mcl(graphs_local)
        cluster_s = perf_counter() - t4

    # Publish artifacts atomically
    graphs_out = shard_dir / "graphs.h5"
    manifest_out = shard_dir / "manifest.tsv.gz"
    meta_out = shard_dir / "meta_stageA.json"

    tmp_graphs = graphs_out.with_suffix(".h5.tmp")
    shutil.copy2(graphs_local, tmp_graphs)
    tmp_graphs.replace(graphs_out)

    gz_write_tsv(
        manifest_out,
        header="label\tmol\tchain\tlength\tsequence\n",
        rows=manifest_rows,
    )

    res = StageAResult(
        shard_id=shard_id,
        n_inputs=len(inputs),
        n_models=len(expanded),
        n_ok=ok,
        n_fail=fail,
        prep_s=prep_s,
        annotate_s=annotate_s,
        graphs_s=graphs_s,
        cluster_s=cluster_s,
        graphs_bytes=graphs_out.stat().st_size,
    )

    atomic_write_json(meta_out, {
        "stage": "A",
        "shard_id": shard_id,
        "shard_list": str(shard_list),
        "exchange_dir": str(shard_dir),
        "chains": {"heavy": heavy, "light": light, "antigen": antigen, "antigen_chainid_for_graph": antigen_chainid_for_graph},
        "num_cores": num_cores,
        "do_cluster": bool(do_cluster),
        "result": asdict(res),
        "wall_s": perf_counter() - t0,
    })

    atomic_write_text(done, "ok\n")
    shutil.rmtree(local_root, ignore_errors=True)
    return res


# -----------------------
# CLI
# -----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-root", required=True, help="Folder of input PDBs (BeeGFS)")
    ap.add_argument("--run-root", required=True, help="Run root (BeeGFS) e.g. .../deeprankab_run_015")
    ap.add_argument("--make-shards", action="store_true", help="Create shard lists then exit")
    ap.add_argument("--target-shard-gb", type=float, default=5.0)
    ap.add_argument("--min-per-shard", type=int, default=200)
    ap.add_argument("--max-per-shard", type=int, default=1200)
    ap.add_argument("--glob", default="*.pdb")

    ap.add_argument("--shard-id", default="", help="Run Stage A for a specific shard id (e.g. 000000)")
    ap.add_argument("--heavy", required=True)
    ap.add_argument("--light", default="-")
    ap.add_argument("--antigen", required=True)
    ap.add_argument("--antigen-chainid-for-graph", default="B")
    ap.add_argument("--num-cores", type=int, default=int(os.environ.get("NUM_CORES", "32")))
    ap.add_argument("--tmp-base", default=os.environ.get("TMPDIR", ""))
    ap.add_argument("--no-cluster", action="store_true", help="Skip MCL clustering in Stage A (not recommended)")

    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    exchange = safe_mkdir(run_root / "exchange")
    shard_lists_dir = safe_mkdir(exchange / "shard_lists")
    shards_dir = safe_mkdir(exchange / "shards")

    if args.make_shards:
        out_lists = build_shard_lists(
            pdb_root=Path(args.pdb_root).resolve(),
            shard_lists_dir=shard_lists_dir,
            target_gb=float(args.target_shard_gb),
            max_per_shard=int(args.max_per_shard),
            min_per_shard=int(args.min_per_shard),
            glob_pat=str(args.glob),
        )
        print(f"✓ Wrote {len(out_lists)} shard lists in {shard_lists_dir}")
        return 0

    if not args.shard_id:
        print("ERROR: provide --make-shards or --shard-id", file=os.sys.stderr)
        return 2

    shard_list = shard_lists_dir / f"shard_{args.shard_id}.lst"
    if not shard_list.is_file():
        raise SystemExit(f"Missing shard list: {shard_list}")

    tmp_base = Path(args.tmp_base) if args.tmp_base else None
    res = run_stageA_one_shard(
        shard_id=args.shard_id,
        shard_list=shard_list,
        exchange_shards_dir=shards_dir,
        heavy=args.heavy,
        light=args.light,
        antigen=args.antigen,
        antigen_chainid_for_graph=args.antigen_chainid_for_graph,
        num_cores=args.num_cores,
        tmp_base=tmp_base,
        do_cluster=(not args.no_cluster),
    )
    gb = res.graphs_bytes / (1024**3)
    print(f"✓ StageA shard_{res.shard_id}: graphs={gb:.2f} GB ok={res.n_ok} fail={res.n_fail}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

