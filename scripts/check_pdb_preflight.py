#!/usr/bin/env python3
from __future__ import annotations

"""
check_pdb_preflight.py

Sample a subset of input PDBs and flag obvious structural/data issues that can
break or degrade DeepRank-Ab Stage A/B processing.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
UNSUPPORTED_AA = {"ASX", "GLX", "SEC", "PYL", "UNK"}

DEFAULT_THRESHOLDS = {
    "hydrogen_fraction_warn": 0.02,
    "altloc_fraction_fail": 0.05,
    "backbone_missing_warn": 0.02,
    "backbone_missing_fail": 0.10,
    "noncanonical_fraction_warn": 0.05,
    "interface_min_dist_warn": 12.0,
    "interface_min_dist_fail": 25.0,
    "max_abs_coord_fail": 1.0e4,
}


@dataclass
class Finding:
    id: str
    severity: str  # WARN | FAIL
    message: str
    value: Optional[object] = None
    threshold: Optional[object] = None

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "id": self.id,
            "severity": self.severity,
            "message": self.message,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.threshold is not None:
            out["threshold"] = self.threshold
        return out


def _is_missing_chain(chain_id: str) -> bool:
    return chain_id.strip() in {"", "-", "none", "None"}


def _chain_label(chain_id: str) -> str:
    return "<blank>" if chain_id == "" else chain_id


def _fmt_ratio(x: float) -> float:
    # Keep JSON/TSV compact but stable.
    return round(float(x), 6)


def _is_hydrogen(atom_name: str, element: str) -> bool:
    e = element.strip().upper()
    if e == "H":
        return True
    an = atom_name.strip().upper()
    if not an:
        return False
    if an.startswith("H"):
        return True
    return len(an) >= 2 and an[0].isdigit() and an[1] == "H"


def _safe_float(chunk: str) -> Optional[float]:
    try:
        return float(chunk.strip())
    except Exception:
        return None


def _min_pair_distance(coords_a: Sequence[Tuple[float, float, float]],
                       coords_b: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    if not coords_a or not coords_b:
        return None
    b = np.asarray(coords_b, dtype=np.float64)
    n_b = max(1, b.shape[0])
    # Keep temporary diff array bounded on large structures.
    target_pairs = 2_000_000
    chunk = max(1, min(1024, target_pairs // n_b))
    min_d2 = float("inf")
    for i in range(0, len(coords_a), chunk):
        a = np.asarray(coords_a[i:i + chunk], dtype=np.float64)
        diff = a[:, None, :] - b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        cur = float(np.min(d2))
        if cur < min_d2:
            min_d2 = cur
    return math.sqrt(min_d2) if math.isfinite(min_d2) else None


def _scan_pdb_atoms(path: Path, max_abs_coord_fail: float) -> Dict[str, object]:
    model_markers = 0
    chains_found: Set[str] = set()
    atom_chains: Set[str] = set()
    atom_count = 0
    hydrogen_count = 0
    oxt_count = 0
    malformed_coord_count = 0
    non_finite_coord_count = 0
    extreme_coord_count = 0

    all_residues: Set[Tuple[str, str, str]] = set()
    atom_residues: Set[Tuple[str, str, str]] = set()
    residue_atoms: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    resname_by_residue: Dict[Tuple[str, str, str], str] = {}
    altloc_residues: Set[Tuple[str, str, str]] = set()

    chain_coords_non_h: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    with path.open("r", errors="replace") as fh:
        for line in fh:
            rec = line[0:6].strip().upper()
            if rec == "MODEL":
                model_markers += 1
                continue
            if rec not in {"ATOM", "HETATM"}:
                continue

            atom_name = line[12:16].strip().upper()
            altloc = line[16:17].strip()
            resname = line[17:20].strip().upper()
            chain_id = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            element = line[76:78].strip()

            resid = (chain_id, resseq, icode)
            chains_found.add(chain_id)
            all_residues.add(resid)

            if altloc:
                altloc_residues.add(resid)

            if atom_name == "OXT":
                oxt_count += 1

            if rec == "ATOM":
                atom_count += 1
                atom_chains.add(chain_id)
                atom_residues.add(resid)
                residue_atoms[resid].add(atom_name)
                resname_by_residue.setdefault(resid, resname)

                if _is_hydrogen(atom_name, element):
                    hydrogen_count += 1

                x = _safe_float(line[30:38])
                y = _safe_float(line[38:46])
                z = _safe_float(line[46:54])
                if x is None or y is None or z is None:
                    malformed_coord_count += 1
                else:
                    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                        non_finite_coord_count += 1
                    if max(abs(x), abs(y), abs(z)) > max_abs_coord_fail:
                        extreme_coord_count += 1
                    if not _is_hydrogen(atom_name, element):
                        chain_coords_non_h[chain_id].append((x, y, z))

    model_count = model_markers if model_markers > 0 else 1
    return {
        "model_count_text": model_count,
        "chains_found": chains_found,
        "atom_chains": atom_chains,
        "atom_count": atom_count,
        "hydrogen_count": hydrogen_count,
        "oxt_count": oxt_count,
        "malformed_coord_count": malformed_coord_count,
        "non_finite_coord_count": non_finite_coord_count,
        "extreme_coord_count": extreme_coord_count,
        "all_residues": all_residues,
        "atom_residues": atom_residues,
        "residue_atoms": residue_atoms,
        "resname_by_residue": resname_by_residue,
        "altloc_residues": altloc_residues,
        "chain_coords_non_h": chain_coords_non_h,
    }


def _add_finding(findings: List[Finding], finding: Finding) -> None:
    findings.append(finding)


def analyze_pdb(path: Path,
                heavy: str,
                light: str,
                antigen: str,
                parser_cls,
                thresholds: Dict[str, float]) -> Dict[str, object]:
    findings: List[Finding] = []

    required_chains = [heavy, antigen] + ([] if _is_missing_chain(light) else [light])
    required_chain_set = set(required_chains)
    ab_chains = [heavy] + ([] if _is_missing_chain(light) else [light])

    scan = _scan_pdb_atoms(path, thresholds["max_abs_coord_fail"])
    atom_count = int(scan["atom_count"])
    hydrogen_count = int(scan["hydrogen_count"])
    oxt_count = int(scan["oxt_count"])
    chains_found: Set[str] = set(scan["chains_found"])
    atom_chains: Set[str] = set(scan["atom_chains"])
    all_residues: Set[Tuple[str, str, str]] = set(scan["all_residues"])
    atom_residues: Set[Tuple[str, str, str]] = set(scan["atom_residues"])
    residue_atoms: Dict[Tuple[str, str, str], Set[str]] = dict(scan["residue_atoms"])
    resname_by_residue: Dict[Tuple[str, str, str], str] = dict(scan["resname_by_residue"])
    altloc_residues: Set[Tuple[str, str, str]] = set(scan["altloc_residues"])
    chain_coords_non_h: Dict[str, List[Tuple[float, float, float]]] = dict(scan["chain_coords_non_h"])

    # Parseability check via Biopython (separate from text scan).
    parse_error: Optional[str] = None
    model_count = int(scan["model_count_text"])
    try:
        parser = parser_cls(QUIET=True)
        structure = parser.get_structure(path.stem, str(path))
        models = list(structure.get_models())
        if models:
            model_count = len(models)
    except Exception as e:
        parse_error = str(e)
        _add_finding(findings, Finding(
            id="parse_error",
            severity="FAIL",
            message="Biopython could not parse this PDB.",
            value=parse_error,
        ))

    if atom_count == 0:
        _add_finding(findings, Finding(
            id="no_atom_records",
            severity="FAIL",
            message="No ATOM records found.",
        ))

    # Chain checks.
    for chain_id in required_chains:
        if chain_id not in chains_found:
            _add_finding(findings, Finding(
                id="missing_required_chain",
                severity="FAIL",
                message=f"Required chain '{_chain_label(chain_id)}' not found.",
                value=_chain_label(chain_id),
            ))
        elif chain_id not in atom_chains:
            _add_finding(findings, Finding(
                id="empty_required_chain",
                severity="FAIL",
                message=f"Required chain '{_chain_label(chain_id)}' has no ATOM records.",
                value=_chain_label(chain_id),
            ))

    # Multi-model.
    if model_count > 1:
        _add_finding(findings, Finding(
            id="multiple_models",
            severity="WARN",
            message="Multiple MODEL entries present; Stage A will split these into separate model files.",
            value=model_count,
            threshold=1,
        ))

    # OXT / hydrogens / altloc.
    hydrogen_frac = (hydrogen_count / atom_count) if atom_count > 0 else 0.0
    if oxt_count > 0:
        _add_finding(findings, Finding(
            id="has_oxt",
            severity="WARN",
            message="OXT atoms present.",
            value=oxt_count,
            threshold=0,
        ))
    if hydrogen_frac > thresholds["hydrogen_fraction_warn"]:
        _add_finding(findings, Finding(
            id="high_hydrogen_fraction",
            severity="WARN",
            message="Hydrogen atom fraction is high.",
            value=_fmt_ratio(hydrogen_frac),
            threshold=thresholds["hydrogen_fraction_warn"],
        ))

    altloc_res_frac = (len(altloc_residues) / len(all_residues)) if all_residues else 0.0
    if altloc_residues:
        _add_finding(findings, Finding(
            id="has_altloc",
            severity="WARN",
            message="Alternate-location residues detected.",
            value=len(altloc_residues),
            threshold=0,
        ))
    if altloc_res_frac > thresholds["altloc_fraction_fail"]:
        _add_finding(findings, Finding(
            id="high_altloc_fraction",
            severity="FAIL",
            message="High fraction of residues with alternate locations.",
            value=_fmt_ratio(altloc_res_frac),
            threshold=thresholds["altloc_fraction_fail"],
        ))

    malformed_coord_count = int(scan["malformed_coord_count"])
    if malformed_coord_count > 0:
        _add_finding(findings, Finding(
            id="malformed_coordinates",
            severity="FAIL",
            message="Malformed coordinate fields found in ATOM records.",
            value=malformed_coord_count,
            threshold=0,
        ))
    non_finite_coord_count = int(scan["non_finite_coord_count"])
    if non_finite_coord_count > 0:
        _add_finding(findings, Finding(
            id="non_finite_coords",
            severity="FAIL",
            message="Non-finite coordinates (NaN/Inf) detected.",
            value=non_finite_coord_count,
            threshold=0,
        ))
    extreme_coord_count = int(scan["extreme_coord_count"])
    if extreme_coord_count > 0:
        _add_finding(findings, Finding(
            id="extreme_coords",
            severity="FAIL",
            message="Extreme coordinate magnitudes detected.",
            value=extreme_coord_count,
            threshold=thresholds["max_abs_coord_fail"],
        ))

    # Residue-level checks.
    relevant_residues = [
        resid for resid in atom_residues
        if (not required_chain_set or resid[0] in required_chain_set)
    ]
    if not relevant_residues:
        relevant_residues = list(atom_residues)

    missing_backbone = 0
    noncanonical = 0
    unsupported_seen: Set[str] = set()
    for resid in relevant_residues:
        resname = resname_by_residue.get(resid, "")
        atoms = residue_atoms.get(resid, set())
        if not {"N", "CA", "C"}.issubset(atoms):
            missing_backbone += 1
        if resname and resname not in CANONICAL_AA:
            noncanonical += 1
            if resname in UNSUPPORTED_AA:
                unsupported_seen.add(resname)

    denom = max(1, len(relevant_residues))
    missing_backbone_frac = missing_backbone / denom
    noncanonical_frac = noncanonical / denom

    if missing_backbone_frac > thresholds["backbone_missing_fail"]:
        _add_finding(findings, Finding(
            id="backbone_missing_fail",
            severity="FAIL",
            message="Too many residues are missing backbone atoms (N/CA/C).",
            value=_fmt_ratio(missing_backbone_frac),
            threshold=thresholds["backbone_missing_fail"],
        ))
    elif missing_backbone_frac > thresholds["backbone_missing_warn"]:
        _add_finding(findings, Finding(
            id="backbone_missing_warn",
            severity="WARN",
            message="Some residues are missing backbone atoms (N/CA/C).",
            value=_fmt_ratio(missing_backbone_frac),
            threshold=thresholds["backbone_missing_warn"],
        ))

    if unsupported_seen:
        _add_finding(findings, Finding(
            id="unsupported_residue_name",
            severity="FAIL",
            message="Unsupported residue names detected for current graph feature mapping.",
            value=sorted(unsupported_seen),
        ))

    if noncanonical_frac > thresholds["noncanonical_fraction_warn"]:
        _add_finding(findings, Finding(
            id="high_noncanonical_fraction",
            severity="WARN",
            message="High fraction of non-canonical residues detected.",
            value=_fmt_ratio(noncanonical_frac),
            threshold=thresholds["noncanonical_fraction_warn"],
        ))

    # Ab-Ag proximity check.
    ab_coords: List[Tuple[float, float, float]] = []
    for c in ab_chains:
        ab_coords.extend(chain_coords_non_h.get(c, []))
    ag_coords = chain_coords_non_h.get(antigen, [])
    min_ab_ag_dist = _min_pair_distance(ab_coords, ag_coords)

    if min_ab_ag_dist is not None:
        if min_ab_ag_dist > thresholds["interface_min_dist_fail"]:
            _add_finding(findings, Finding(
                id="interface_too_far_fail",
                severity="FAIL",
                message="Antibody-antigen chains are too far apart (likely no interface).",
                value=_fmt_ratio(min_ab_ag_dist),
                threshold=thresholds["interface_min_dist_fail"],
            ))
        elif min_ab_ag_dist > thresholds["interface_min_dist_warn"]:
            _add_finding(findings, Finding(
                id="interface_too_far_warn",
                severity="WARN",
                message="Antibody-antigen chains are far apart.",
                value=_fmt_ratio(min_ab_ag_dist),
                threshold=thresholds["interface_min_dist_warn"],
            ))

    status = "PASS"
    for f in findings:
        if f.severity == "FAIL":
            status = "FAIL"
            break
        if f.severity == "WARN":
            status = "WARN"

    metrics = {
        "n_models": model_count,
        "chains_found": sorted(chains_found),
        "atom_count": atom_count,
        "hydrogen_frac": _fmt_ratio(hydrogen_frac),
        "oxt_count": oxt_count,
        "altloc_res_frac": _fmt_ratio(altloc_res_frac),
        "noncanonical_frac": _fmt_ratio(noncanonical_frac),
        "unsupported_resnames": sorted(unsupported_seen),
        "missing_backbone_frac": _fmt_ratio(missing_backbone_frac),
        "min_ab_ag_dist": None if min_ab_ag_dist is None else _fmt_ratio(min_ab_ag_dist),
        "parse_error": parse_error,
    }

    return {
        "path": str(path.resolve()),
        "status": status,
        "checks": [f.to_dict() for f in findings],
        "metrics": metrics,
    }


def _discover_pdbs(pdb_root: Path, glob_pat: str) -> List[Path]:
    files = [p for p in pdb_root.glob(glob_pat) if p.is_file()]
    if not files and glob_pat != "*.pdb":
        files = [p for p in pdb_root.glob("*.pdb") if p.is_file()]
    return sorted(set(files))


def _sample_files(files: List[Path], n: int, mode: str, seed: Optional[int]) -> List[Path]:
    if n <= 0:
        return []
    if len(files) <= n:
        return files
    if mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(files, n))
    return files[:n]


def _build_recommendations(file_reports: List[Dict[str, object]],
                           total: int,
                           heavy: str,
                           light: str,
                           antigen: str) -> Dict[str, object]:
    issue_file_counts: Counter[str] = Counter()
    for fr in file_reports:
        seen = {c["id"] for c in fr["checks"]}
        issue_file_counts.update(seen)

    recs: List[Dict[str, str]] = []
    cure_reasons: List[str] = []
    cure_count = issue_file_counts.get("has_oxt", 0) + issue_file_counts.get("high_hydrogen_fraction", 0)

    if issue_file_counts.get("has_oxt", 0) > 0:
        cure_reasons.append("oxt")
    if issue_file_counts.get("high_hydrogen_fraction", 0) > 0:
        cure_reasons.append("hydrogens")

    if cure_reasons:
        frac = cure_count / max(1, total)
        severity = "high" if frac >= 0.20 else "medium"
        msg = (
            "Detected OXT/hydrogen issues; consider curing inputs before Stage A using "
            "`/Users/valkove2/Documents/GitHub/DeepRank-Ab/scripts/cure_pdbs.sh`."
        )
        if severity == "high":
            msg += " Issue prevalence is high in the sampled set."
        recs.append({"severity": severity, "message": msg})

    if issue_file_counts.get("missing_required_chain", 0) > 0 or issue_file_counts.get("empty_required_chain", 0) > 0:
        recs.append({
            "severity": "high",
            "message": (
                f"Verify chain mapping for this dataset (expected heavy={heavy}, light={light}, antigen={antigen}) "
                "before launching Stage A."
            ),
        })

    if issue_file_counts.get("unsupported_residue_name", 0) > 0:
        recs.append({
            "severity": "high",
            "message": (
                "Unsupported residue names were found. Replace/normalize residues (or adjust feature mapping) "
                "before Stage A."
            ),
        })

    if issue_file_counts.get("multiple_models", 0) > 0:
        recs.append({
            "severity": "low",
            "message": (
                "Multiple MODEL entries are present. Stage A will split models, which can increase shard runtime."
            ),
        })

    if issue_file_counts.get("interface_too_far_fail", 0) > 0 or issue_file_counts.get("interface_too_far_warn", 0) > 0:
        recs.append({
            "severity": "medium",
            "message": (
                "Some structures have weak/no antibody-antigen proximity; these may produce poor or empty interface signals."
            ),
        })

    return {
        "recommend_cure_pdbs": bool(cure_reasons),
        "cure_reasons": sorted(set(cure_reasons)),
        "items": recs,
    }


def _write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "status",
        "n_models",
        "chains_found",
        "atom_count",
        "hydrogen_frac",
        "oxt_count",
        "altloc_res_frac",
        "noncanonical_frac",
        "unsupported_resnames",
        "missing_backbone_frac",
        "min_ab_ag_dist",
        "issues",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight validator for DeepRank-Ab input PDBs.")
    ap.add_argument("pdb_root", type=Path, help="Directory containing input PDBs.")
    ap.add_argument("--sample-n", type=int, default=50, help="Number of PDBs to inspect (default: 50).")
    ap.add_argument("--glob", default="**/*.pdb", help="Glob pattern under pdb_root (default: **/*.pdb).")
    ap.add_argument("--sample-mode", choices=["first", "random"], default="first", help="How to pick sampled files.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed when --sample-mode random.")
    ap.add_argument("--heavy", default="H", help="Heavy chain ID (default: H).")
    ap.add_argument("--light", default="-", help="Light chain ID, '-' for none (default: -).")
    ap.add_argument("--antigen", default="T", help="Antigen chain ID (default: T).")
    ap.add_argument("--json-out", type=Path, default=Path("preflight_report.json"), help="JSON report output path.")
    ap.add_argument("--tsv-out", type=Path, default=Path("preflight_report.tsv"), help="TSV report output path.")
    ap.add_argument("--fail-on", choices=["fail", "warn", "never"], default="fail",
                    help="Exit non-zero on FAIL, WARN+FAIL, or never.")
    ap.add_argument("--strict", action="store_true", help="Equivalent to --fail-on warn.")
    args = ap.parse_args()

    if args.strict:
        args.fail_on = "warn"
    if args.sample_n <= 0:
        print("ERROR: --sample-n must be > 0", file=sys.stderr)
        return 4

    try:
        from Bio.PDB import PDBParser as parser_cls
    except Exception as e:
        print(
            "ERROR: Biopython is required for parseability checks. "
            f"Install/activate environment with Bio.PDB available. ({e})",
            file=sys.stderr,
        )
        return 4

    pdb_root = args.pdb_root.expanduser().resolve()
    if not pdb_root.is_dir():
        print(f"ERROR: pdb_root is not a directory: {pdb_root}", file=sys.stderr)
        return 3

    files = _discover_pdbs(pdb_root, args.glob)
    if not files:
        print(f"ERROR: no files found under {pdb_root} with glob '{args.glob}'", file=sys.stderr)
        return 3

    sampled = _sample_files(files, args.sample_n, args.sample_mode, args.seed)
    thresholds = dict(DEFAULT_THRESHOLDS)

    file_reports: List[Dict[str, object]] = []
    for p in sampled:
        try:
            file_reports.append(analyze_pdb(
                p,
                heavy=args.heavy,
                light=args.light,
                antigen=args.antigen,
                parser_cls=parser_cls,
                thresholds=thresholds,
            ))
        except Exception as e:
            file_reports.append({
                "path": str(p.resolve()),
                "status": "FAIL",
                "checks": [{
                    "id": "checker_internal_error",
                    "severity": "FAIL",
                    "message": "Checker failed while processing this file.",
                    "value": str(e),
                }],
                "metrics": {},
            })

    status_counts = Counter(fr["status"] for fr in file_reports)
    issue_counts: Counter[str] = Counter()
    for fr in file_reports:
        issue_ids = {c["id"] for c in fr["checks"]}
        issue_counts.update(issue_ids)

    recommendations = _build_recommendations(
        file_reports=file_reports,
        total=len(file_reports),
        heavy=args.heavy,
        light=args.light,
        antigen=args.antigen,
    )

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pdb_root": str(pdb_root),
            "glob": args.glob,
            "sample_mode": args.sample_mode,
            "sample_n_requested": args.sample_n,
            "sample_n_actual": len(sampled),
            "heavy": args.heavy,
            "light": args.light,
            "antigen": args.antigen,
        },
        "thresholds": thresholds,
        "summary": {
            "total": len(file_reports),
            "pass": status_counts.get("PASS", 0),
            "warn": status_counts.get("WARN", 0),
            "fail": status_counts.get("FAIL", 0),
        },
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "files": file_reports,
        "recommendations": recommendations,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n")

    tsv_rows: List[Dict[str, object]] = []
    for fr in file_reports:
        m = fr.get("metrics", {})
        chains = m.get("chains_found", [])
        chains_s = ",".join(_chain_label(c) for c in chains) if chains else ""
        unsupported = m.get("unsupported_resnames", [])
        issues = ",".join(sorted({c["id"] for c in fr["checks"]}))
        tsv_rows.append({
            "path": fr["path"],
            "status": fr["status"],
            "n_models": m.get("n_models", ""),
            "chains_found": chains_s,
            "atom_count": m.get("atom_count", ""),
            "hydrogen_frac": m.get("hydrogen_frac", ""),
            "oxt_count": m.get("oxt_count", ""),
            "altloc_res_frac": m.get("altloc_res_frac", ""),
            "noncanonical_frac": m.get("noncanonical_frac", ""),
            "unsupported_resnames": ",".join(unsupported) if unsupported else "",
            "missing_backbone_frac": m.get("missing_backbone_frac", ""),
            "min_ab_ag_dist": m.get("min_ab_ag_dist", ""),
            "issues": issues,
        })
    _write_tsv(args.tsv_out, tsv_rows)

    # Console summary
    print(f"Scanned {len(sampled)} / {len(files)} PDB files from: {pdb_root}")
    print(
        f"Status: PASS={status_counts.get('PASS', 0)} "
        f"WARN={status_counts.get('WARN', 0)} "
        f"FAIL={status_counts.get('FAIL', 0)}"
    )
    if issue_counts:
        print("Top issues:")
        for issue_id, count in sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            print(f"  - {issue_id}: {count}")

    for rec in recommendations.get("items", []):
        sev = rec.get("severity", "info").upper()
        msg = rec.get("message", "")
        print(f"[{sev}] {msg}")

    print(f"JSON report: {args.json_out.resolve()}")
    print(f"TSV report:  {args.tsv_out.resolve()}")

    fail_count = status_counts.get("FAIL", 0)
    warn_count = status_counts.get("WARN", 0)
    if args.fail_on == "fail":
        return 2 if fail_count > 0 else 0
    if args.fail_on == "warn":
        return 2 if (fail_count > 0 or warn_count > 0) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
