#!/usr/bin/env python3
"""
prodigy_batch_worker.py

High-performance PRODIGY batch processor using the library directly.
Eliminates subprocess overhead by importing Prodigy once per worker.

Designed for million-scale PDB processing on HPC clusters.
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger("prodigy-batch")

# Output fields (matches original script for compatibility)
FIELDS = [
    "pdb",
    "n_chains",
    "n_residues",
    "ic_total",
    "charged_charged",
    "charged_polar",
    "charged_apolar",
    "polar_polar",
    "apolar_polar",
    "apolar_apolar",
    "pct_apolar_nis",
    "pct_charged_nis",
    "pct_polar_nis",
    "dG_kcal_mol",
    "Kd_M",
    "temperature_C",
    "status",
    "error",
]


@dataclass
class ProdigyResult:
    """Container for PRODIGY analysis results."""
    pdb: str = ""
    n_chains: int = 0
    n_residues: int = 0
    ic_total: int = 0
    charged_charged: float = 0.0
    charged_polar: float = 0.0
    charged_apolar: float = 0.0
    polar_polar: float = 0.0
    apolar_polar: float = 0.0
    apolar_apolar: float = 0.0
    pct_apolar_nis: float = 0.0
    pct_charged_nis: float = 0.0
    pct_polar_nis: float = 0.0
    dG_kcal_mol: float = 0.0
    Kd_M: float = 0.0
    temperature_C: float = 25.0
    status: str = ""
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "pdb": self.pdb,
            "n_chains": self.n_chains,
            "n_residues": self.n_residues,
            "ic_total": self.ic_total,
            "charged_charged": self.charged_charged,
            "charged_polar": self.charged_polar,
            "charged_apolar": self.charged_apolar,
            "polar_polar": self.polar_polar,
            "apolar_polar": self.apolar_polar,
            "apolar_apolar": self.apolar_apolar,
            "pct_apolar_nis": self.pct_apolar_nis,
            "pct_charged_nis": self.pct_charged_nis,
            "pct_polar_nis": self.pct_polar_nis,
            "dG_kcal_mol": self.dG_kcal_mol,
            "Kd_M": self.Kd_M,
            "temperature_C": self.temperature_C,
            "status": self.status,
            "error": self.error,
        }


# Global worker state (initialized once per process)
_worker_parser = None
_worker_prodigy_cls = None


def _init_worker():
    """
    Initialize expensive imports once per worker process.
    Called automatically by ProcessPoolExecutor.
    """
    global _worker_parser, _worker_prodigy_cls

    from Bio.PDB import PDBParser
    from prodigy_prot.modules.prodigy import Prodigy

    _worker_parser = PDBParser(QUIET=True)
    _worker_prodigy_cls = Prodigy


def _process_single_pdb(
    pdb_path: str,
    selection_a: str,
    selection_b: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
) -> ProdigyResult:
    """
    Process a single PDB file using the pre-initialized Prodigy class.
    This runs in a worker process where imports are already done.
    """
    global _worker_parser, _worker_prodigy_cls

    p = Path(pdb_path)
    result = ProdigyResult(pdb=p.stem, temperature_C=temperature)

    try:
        # Parse structure
        structure = _worker_parser.get_structure(p.stem, str(p))
        model = structure[0]

        # Count chains and residues
        chains = list(model.get_chains())
        result.n_chains = len(chains)
        result.n_residues = sum(1 for _ in model.get_residues())

        # Run PRODIGY analysis
        # Selection format: list of chain group strings
        # e.g., ["H", "T"] means chain H vs chain T
        # e.g., ["H,L", "T"] means chains H+L vs chain T
        prod = _worker_prodigy_cls(
            model,
            name=p.stem,
            selection=[selection_a, selection_b],
            temp=temperature,
        )
        prod.predict(distance_cutoff=distance_cutoff, acc_threshold=acc_threshold)

        # Extract results from Prodigy's dict
        d = prod.as_dict()

        result.ic_total = int(d.get("ICs", 0))
        result.charged_charged = float(d.get("CC", 0.0))
        result.charged_polar = float(d.get("CP", 0.0))
        result.charged_apolar = float(d.get("AC", 0.0))
        result.polar_polar = float(d.get("PP", 0.0))
        result.apolar_polar = float(d.get("AP", 0.0))
        result.apolar_apolar = float(d.get("AA", 0.0))
        result.pct_apolar_nis = float(d.get("nis_a", 0.0))
        result.pct_charged_nis = float(d.get("nis_c", 0.0))
        result.pct_polar_nis = float(d.get("nis_p", 0.0))
        result.dG_kcal_mol = float(d.get("ba_val", 0.0))
        result.Kd_M = float(d.get("kd_val", 0.0))
        result.status = "ok"

    except Exception as e:
        result.status = "fail"
        result.error = str(e)[:2000]  # Truncate long errors

    return result


def _worker_process_batch(args: Tuple) -> List[ProdigyResult]:
    """
    Process a batch of PDBs in a single worker process.
    Batching reduces IPC overhead when processing many small files.
    """
    pdb_paths, selection_a, selection_b, temperature, distance_cutoff, acc_threshold = args

    results = []
    for pdb_path in pdb_paths:
        r = _process_single_pdb(
            pdb_path, selection_a, selection_b,
            temperature, distance_cutoff, acc_threshold
        )
        results.append(r)
    return results


def run_batch_prodigy(
    pdb_paths: List[Path],
    selection_a: str,
    selection_b: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    n_workers: int,
    batch_size: int = 50,
) -> List[ProdigyResult]:
    """
    Process multiple PDBs in parallel using ProcessPoolExecutor.
    
    Args:
        pdb_paths: List of PDB file paths
        selection_a: First selection (chain(s) for partner A)
        selection_b: Second selection (chain(s) for partner B)
        temperature: Temperature in Celsius for Kd calculation
        distance_cutoff: Distance cutoff for contacts (Angstroms)
        acc_threshold: Accessibility threshold for NIS
        n_workers: Number of parallel worker processes
        batch_size: Number of PDBs per work unit (reduces IPC overhead)
    
    Returns:
        List of ProdigyResult objects
    """
    if not pdb_paths:
        return []

    # Create batches to reduce IPC overhead
    batches = []
    for i in range(0, len(pdb_paths), batch_size):
        batch_paths = [str(p) for p in pdb_paths[i:i + batch_size]]
        batches.append((
            batch_paths,
            selection_a,
            selection_b,
            temperature,
            distance_cutoff,
            acc_threshold,
        ))

    log.info(f"Processing {len(pdb_paths)} PDBs in {len(batches)} batches with {n_workers} workers")

    all_results: List[ProdigyResult] = []
    completed = 0

    t0 = perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_worker_process_batch, batch): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                completed += len(batch_results)

                # Progress logging every 10% or 1000 structures
                if completed % max(1000, len(pdb_paths) // 10) < batch_size:
                    elapsed = perf_counter() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    log.info(f"Progress: {completed}/{len(pdb_paths)} ({100*completed/len(pdb_paths):.1f}%) - {rate:.1f} PDBs/sec")

            except Exception as e:
                log.error(f"Batch {batch_idx} failed: {e}")
                # Create failure results for this batch
                batch_paths = batches[batch_idx][0]
                for pdb_path in batch_paths:
                    all_results.append(ProdigyResult(
                        pdb=Path(pdb_path).stem,
                        status="fail",
                        error=f"Batch failure: {e}"
                    ))

    elapsed = perf_counter() - t0
    log.info(f"Completed {len(all_results)} PDBs in {elapsed:.1f}s ({len(all_results)/elapsed:.1f} PDBs/sec)")

    return all_results


def write_results_tsv(results: List[ProdigyResult], out_path: Path) -> None:
    """Write results to TSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.pdb):
            writer.writerow(r.to_dict())


def discover_structures(input_dir: Path, extensions: Tuple[str, ...] = (".pdb", ".cif", ".mmcif")) -> List[Path]:
    """
    Recursively find all structure files under input_dir.
    Follows symlinks, excludes manifest files.
    """
    structures = []
    for ext in extensions:
        # Case-insensitive matching
        structures.extend(input_dir.rglob(f"*{ext}"))
        structures.extend(input_dir.rglob(f"*{ext.upper()}"))

    # Filter out manifest files
    structures = [p for p in structures if "manifest" not in p.name.lower()]

    return sorted(set(structures))


def main():
    parser = argparse.ArgumentParser(
        description="High-performance PRODIGY batch processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument("--input-dir", required=True, help="Directory containing PDB/CIF files")
    parser.add_argument("--output-tsv", required=True, help="Output TSV file path")
    parser.add_argument("--file-list", default="", help="Optional: file containing list of PDB paths (one per line)")

    # PRODIGY parameters
    parser.add_argument("--selection-a", default="H", help="Chain selection for partner A (comma-separated for multiple)")
    parser.add_argument("--selection-b", default="T", help="Chain selection for partner B (comma-separated for multiple)")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature (Celsius) for Kd calculation")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Distance cutoff for contacts (Angstroms)")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Accessibility threshold for NIS calculation")

    # Parallelization
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0 = auto-detect from SLURM or CPU count)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="PDBs per work unit (higher = less IPC overhead)")

    # Subset selection (for SLURM array jobs)
    parser.add_argument("--start-index", type=int, default=0, help="Start index in file list (0-based)")
    parser.add_argument("--count", type=int, default=0, help="Number of files to process (0 = all from start-index)")

    args = parser.parse_args()

    # Determine worker count
    if args.workers <= 0:
        # Try SLURM env, then fall back to CPU count
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
        if slurm_cpus:
            args.workers = int(slurm_cpus)
        else:
            args.workers = mp.cpu_count()
    log.info(f"Using {args.workers} worker processes")

    # Discover or load structure files
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        log.error(f"Input directory does not exist: {input_dir}")
        sys.exit(2)

    if args.file_list and Path(args.file_list).is_file():
        log.info(f"Loading file list from: {args.file_list}")
        with open(args.file_list) as f:
            all_pdbs = [Path(line.strip()) for line in f if line.strip()]
    else:
        log.info(f"Discovering structures in: {input_dir}")
        all_pdbs = discover_structures(input_dir)

    if not all_pdbs:
        log.error(f"No structure files found in {input_dir}")
        sys.exit(2)

    log.info(f"Total structures discovered: {len(all_pdbs)}")

    # Apply subset selection
    start = args.start_index
    count = args.count if args.count > 0 else (len(all_pdbs) - start)
    end = min(start + count, len(all_pdbs))

    if start >= len(all_pdbs):
        log.info(f"Start index {start} >= total {len(all_pdbs)}; nothing to do")
        sys.exit(0)

    pdbs_to_process = all_pdbs[start:end]
    log.info(f"Processing subset: indices {start} to {end-1} ({len(pdbs_to_process)} structures)")

    # Run PRODIGY analysis
    results = run_batch_prodigy(
        pdb_paths=pdbs_to_process,
        selection_a=args.selection_a,
        selection_b=args.selection_b,
        temperature=args.temperature,
        distance_cutoff=args.distance_cutoff,
        acc_threshold=args.acc_threshold,
        n_workers=args.workers,
        batch_size=args.batch_size,
    )

    # Write output
    out_path = Path(args.output_tsv)
    write_results_tsv(results, out_path)
    log.info(f"Wrote {len(results)} results to {out_path}")

    # Summary stats
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    log.info(f"Summary: {n_ok} succeeded, {n_fail} failed")

    # Tolerate small failure rates (<5%) - normal for large datasets
    if len(results) == 0:
        log.error("No results produced")
        return 1
    
    fail_rate = n_fail / len(results)
    if fail_rate > 0.05:  # More than 5% failed
        log.error(f"High failure rate: {fail_rate*100:.1f}% ({n_fail}/{len(results)})")
        return 1
    elif n_fail > 0:
        log.warning(f"Some structures failed ({n_fail}/{len(results)}, {fail_rate*100:.2f}%), but under 5% threshold")
        return 0
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
