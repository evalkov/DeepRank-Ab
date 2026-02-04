"""
Unified Voronota-LT wrapper for contact areas and BSA computation.

Key goals:
- One voronota-lt call can produce BOTH atom-atom contacts and complex residue SAS.
- Chain SAS calculations can be run concurrently (two subprocesses) if BSA is needed.
- OpenMP threading is controlled via --processors, but defaults must be HPC-safe:
    * For throughput (many Python worker processes), you typically want processors=1
      to avoid oversubscription (e.g., 48 workers × 4 threads = 192 runnable threads).

Environment variables:
- VORO_PROCESSORS: OpenMP threads per voronota-lt call (default: 1)
- VORO_CHAIN_PARALLEL: Run chain A/B SAS in parallel (default: 1)
    * Set to 0 for HPC with many workers to avoid oversubscription
    * With 48 workers and VORO_CHAIN_PARALLEL=1: peak 96 subprocesses
    * With 48 workers and VORO_CHAIN_PARALLEL=0: peak 48 subprocesses
- VORONOTA_LT: Path to voronota-lt binary (optional)

This module provides:
- VoroContacts: unified engine
- VoronotaAreas: drop-in compatibility wrapper
- BSA: drop-in compatibility wrapper
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional, Iterable, Any

import numpy as np


AtomKey = Tuple[str, str, str, str]          # (chainID, resSeq, resName, atomName)
ResidueKey = Tuple[str, int, str]            # (chainID, resSeq, resName)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return max(1, int(v))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean from environment variable (0/1, true/false, yes/no)."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _pick_header_index(header: Iterable[str], candidates: Iterable[str]) -> Optional[int]:
    h = list(header)
    for c in candidates:
        if c in h:
            return h.index(c)
    return None


class VoroContacts:
    """
    Unified voronota-lt wrapper for:
      - atom-atom contact areas (contacts table)
      - residue-level SAS areas for complex and chains (cells residue-level table)
      - residue BSA via ASA_unbound(chain) - ASA_complex

    The OpenMP thread count used by each voronota-lt subprocess is `processors`.
    Default behavior:
      - If processors is None: read env var VORO_PROCESSORS (default: 1).
    """

    def __init__(
        self,
        pdb_path: str | Path,
        probe: float = 1.4,
        processors: Optional[int] = None,
        compute_contacts: bool = True,
        compute_bsa: bool = True,
        chain_ids: Tuple[str, str] = ("A", "B"),
        voronota_exec: Optional[str] = None,
    ):
        self.pdb_path = str(pdb_path)
        self.probe = float(probe)
        self.processors = int(processors) if processors is not None else _env_int("VORO_PROCESSORS", 1)
        self.chain_ids = tuple(chain_ids)
        self.voronota_exec = voronota_exec or self._get_voronota_executable()

        # Results storage
        self.contact_areas: Dict[AtomKey, Dict[AtomKey, float]] = {}
        self.complex_sas: Dict[ResidueKey, float] = {}
        self.chain_sas: Dict[str, Dict[ResidueKey, float]] = {}
        self.bsa_data: Dict[ResidueKey, list] = {}
        self.bsa_data_xyz: Dict[ResidueKey, tuple] = {}

        if compute_contacts and compute_bsa:
            self._run_combined()
        elif compute_contacts:
            self._run_contacts_only()
        elif compute_bsa:
            self._run_bsa_only()

    # ----------------------------
    # Execution helpers
    # ----------------------------

    def _run_combined(self) -> None:
        """
        Run a single voronota-lt call that writes:
          - contacts.tsv (atom contacts)
          - cells.tsv    (residue-level cells for complex SAS)
        Then run chain SAS for each chain in parallel.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            contacts_file = os.path.join(tmpdir, "contacts.tsv")
            cells_file = os.path.join(tmpdir, "cells.tsv")

            cmd = [
                self.voronota_exec,
                "--input", self.pdb_path,
                "--probe", str(self.probe),
                "--processors", str(self.processors),
                "--write-contacts-to-file", contacts_file,
                "--write-cells-residue-level-to-file", cells_file,
                "--quiet",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"voronota-lt failed (combined): {result.stderr.strip()}")

            self.contact_areas = self._parse_contacts(Path(contacts_file).read_text())
            self.complex_sas = self._parse_cells_residue(Path(cells_file).read_text())

        self._run_chain_sas_parallel()

    def _run_contacts_only(self) -> None:
        cmd = [
            self.voronota_exec,
            "--input", self.pdb_path,
            "--probe", str(self.probe),
            "--processors", str(self.processors),
            "--print-contacts",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"voronota-lt failed (contacts): {result.stderr.strip()}")
        self.contact_areas = self._parse_contacts(result.stdout)

    def _run_bsa_only(self) -> None:
        # Complex residue SAS
        cmd = [
            self.voronota_exec,
            "--input", self.pdb_path,
            "--probe", str(self.probe),
            "--processors", str(self.processors),
            "--print-cells-residue-level",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"voronota-lt failed (complex SAS): {result.stderr.strip()}")
        self.complex_sas = self._parse_cells_residue(result.stdout)

        # Chain SAS in parallel
        self._run_chain_sas_parallel()

    def _run_chain_sas_parallel(self) -> None:
        """
        Compute residue-level SAS per chain.

        Parallelism is controlled by VORO_CHAIN_PARALLEL env var:
          - VORO_CHAIN_PARALLEL=1 (default): Run chain A and B in parallel (2 subprocesses)
          - VORO_CHAIN_PARALLEL=0: Run sequentially (1 subprocess at a time)

        For HPC with many Python workers, set VORO_CHAIN_PARALLEL=0 to avoid
        oversubscription (e.g., 48 workers × 2 chains = 96 concurrent processes).
        """
        parallel = _env_bool("VORO_CHAIN_PARALLEL", default=True)

        def compute_chain_sas(chain_id: str) -> Tuple[str, Dict[ResidueKey, float]]:
            # For chain SAS, you can optionally split processors; but for HPC throughput
            # it's usually best to keep this 1 unless you intentionally reduce Python workers.
            chain_procs = max(1, self.processors // 2) if self.processors > 1 else 1

            cmd = [
                self.voronota_exec,
                "--input", self.pdb_path,
                "--probe", str(self.probe),
                "--processors", str(chain_procs),
                "--restrict-input-atoms", f"[-chain {chain_id}]",
                "--print-cells-residue-level",
                "--quiet",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"voronota-lt failed (chain {chain_id} SAS): {result.stderr.strip()}"
                )
            return chain_id, self._parse_cells_residue(result.stdout)

        self.chain_sas = {}

        if parallel:
            # Run chain SAS calculations in parallel (2 concurrent subprocesses)
            with ThreadPoolExecutor(max_workers=max(1, len(self.chain_ids))) as ex:
                futures = [ex.submit(compute_chain_sas, c) for c in self.chain_ids]
                for fut in as_completed(futures):
                    chain_id, sas_map = fut.result()
                    self.chain_sas[chain_id] = sas_map
        else:
            # Run chain SAS calculations sequentially (1 subprocess at a time)
            for chain_id in self.chain_ids:
                _, sas_map = compute_chain_sas(chain_id)
                self.chain_sas[chain_id] = sas_map

    # ----------------------------
    # Parsers (robust to header naming)
    # ----------------------------

    def _parse_contacts(self, output: str) -> Dict[AtomKey, Dict[AtomKey, float]]:
        """
        Parse voronota-lt contacts TSV into a nested dict:
          contact_areas[atom1_key][atom2_key] = area

        This parser is header-driven and tolerates multiple header naming conventions.
        """
        contact_areas: Dict[AtomKey, Dict[AtomKey, float]] = {}
        text = output.strip()
        if not text:
            return contact_areas

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return contact_areas

        header = lines[0].split("\t")
        # Candidate column names observed across voronota/voronota-lt variants
        c1 = _pick_header_index(header, ["ID1_chain", "chain1", "chainID1", "a1_chain", "atom1_chain", "chainID_1", "ID_chain_1"])
        r1 = _pick_header_index(header, ["ID1_rnum", "rnum1", "resSeq1", "a1_rnum", "atom1_rnum", "ID_rnum_1"])
        n1 = _pick_header_index(header, ["ID1_rname", "rname1", "resName1", "a1_rname", "atom1_rname", "ID_rname_1"])
        a1 = _pick_header_index(header, ["ID1_aname", "aname1", "atomName1", "a1_aname", "atom1_aname", "ID_aname_1"])

        c2 = _pick_header_index(header, ["ID2_chain", "chain2", "chainID2", "a2_chain", "atom2_chain", "chainID_2", "ID_chain_2"])
        r2 = _pick_header_index(header, ["ID2_rnum", "rnum2", "resSeq2", "a2_rnum", "atom2_rnum", "ID_rnum_2"])
        n2 = _pick_header_index(header, ["ID2_rname", "rname2", "resName2", "a2_rname", "atom2_rname", "ID_rname_2"])
        a2 = _pick_header_index(header, ["ID2_aname", "aname2", "atomName2", "a2_aname", "atom2_aname", "ID_aname_2"])

        area_idx = _pick_header_index(header, ["area", "contact_area", "contact_area_value", "area_value", "value"])

        # Fallback: if header detection fails, assume the common voronota-lt layout:
        # ID1_chain ID1_rnum ID1_rname ID1_aname ID2_chain ID2_rnum ID2_rname ID2_aname ... area(last or near-last)
        if None in (c1, r1, n1, a1, c2, r2, n2, a2):
            c1, r1, n1, a1 = 0, 1, 2, 3
            c2, r2, n2, a2 = 4, 5, 6, 7
        if area_idx is None:
            # Try last column as a conservative fallback
            area_idx = len(header) - 1

        for line in lines[1:]:
            parts = line.split("\t")
            need = max(c1, r1, n1, a1, c2, r2, n2, a2, area_idx)
            if len(parts) <= need:
                continue

            at1_key: AtomKey = (parts[c1], parts[r1], parts[n1], parts[a1])
            at2_key: AtomKey = (parts[c2], parts[r2], parts[n2], parts[a2])

            try:
                area = float(parts[area_idx])
            except ValueError:
                continue

            contact_areas.setdefault(at1_key, {})[at2_key] = area

        return contact_areas

    def _parse_cells_residue(self, output: str) -> Dict[ResidueKey, float]:
        """
        Parse voronota-lt residue-level cells TSV into:
          sas[(chain, resnum, resname)] = sas_area
        """
        sas_data: Dict[ResidueKey, float] = {}
        text = output.strip()
        if not text:
            return sas_data

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return sas_data

        header = lines[0].split("\t")
        chain_idx = _pick_header_index(header, ["ID_chain", "chain", "chainID"])
        rnum_idx = _pick_header_index(header, ["ID_rnum", "rnum", "resSeq", "resSeq_num"])
        rname_idx = _pick_header_index(header, ["ID_rname", "rname", "resName"])
        sas_idx = _pick_header_index(header, ["sas_area", "sas", "asa", "asa_area"])

        # Reasonable fallback if headers aren’t present/recognized
        if chain_idx is None:
            chain_idx = 0
        if rnum_idx is None:
            rnum_idx = 1
        if rname_idx is None:
            rname_idx = 2
        if sas_idx is None:
            # Many voronota-lt outputs put sas_area around col 4/5; last is safer than a wrong fixed index.
            sas_idx = len(header) - 1

        need = max(chain_idx, rnum_idx, rname_idx, sas_idx)
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) <= need:
                continue
            try:
                key: ResidueKey = (parts[chain_idx], int(parts[rnum_idx]), parts[rname_idx])
                sas_data[key] = float(parts[sas_idx])
            except ValueError:
                continue

        return sas_data

    # ----------------------------
    # BSA computation
    # ----------------------------

    def compute_bsa(self, contact_residues: Iterable[Tuple[Any, ...]], sqldb: Optional[Any] = None) -> None:
        """
        Compute BSA for residues that are in contact (or any residue list you pass).

        `contact_residues` items are expected to be tuples like:
          (chainID, resSeq, resName)  OR (chainID, resSeq)

        Populates:
          self.bsa_data[(chain, resSeq, resName)] = [bsa]
          self.bsa_data_xyz[...] = (chain_index, x, y, z)  if sqldb provided and xyz available
        """
        if not self.complex_sas or not self.chain_sas:
            raise RuntimeError("compute_bsa requires complex_sas and chain_sas. Run with compute_bsa=True first.")

        chain_to_index = {c: i for i, c in enumerate(self.chain_ids)}

        for res in contact_residues:
            if not res:
                continue
            chain = str(res[0])
            try:
                resSeq = int(res[1])
            except Exception:
                continue
            resName = str(res[2]) if len(res) > 2 and res[2] is not None else None

            # Find the exact key in complex_sas (resName may differ/absent)
            key = None
            if resName is not None:
                candidate = (chain, resSeq, resName)
                if candidate in self.complex_sas:
                    key = candidate
            if key is None:
                # match by chain+resSeq
                for k in self.complex_sas.keys():
                    if k[0] == chain and k[1] == resSeq:
                        key = k
                        resName = k[2]
                        break
            if key is None or resName is None:
                continue

            asa_complex = float(self.complex_sas.get(key, 0.0))
            asa_unbound = float(self.chain_sas.get(chain, {}).get(key, 0.0))
            bsa = asa_unbound - asa_complex

            result_key: ResidueKey = (chain, resSeq, resName)
            self.bsa_data[result_key] = [bsa]

            if sqldb is not None:
                # Attempt xyz as mean of atoms in that residue/chain
                try:
                    coords = sqldb.get("x,y,z", resSeq=resSeq, chainID=chain)
                    coords = np.asarray(coords, dtype=float)
                    if coords.size == 0:
                        continue
                    xyz = coords.mean(axis=0)
                    chain_idx = int(chain_to_index.get(chain, 0))
                    self.bsa_data_xyz[result_key] = (chain_idx, float(xyz[0]), float(xyz[1]), float(xyz[2]))
                except Exception:
                    # Keep bsa_data even if xyz cannot be computed
                    pass

    # ----------------------------
    # Voronota executable resolution
    # ----------------------------

    @staticmethod
    def _get_voronota_executable() -> str:
        """
        Resolution order:
          1) $VORONOTA_LT (explicit path)
          2) ./voronota-lt next to this repo in src/tools/voronota/voronota-lt
          3) `voronota-lt` from PATH
        """
        env = os.environ.get("VORONOTA_LT", "").strip()
        if env:
            p = Path(env)
            if p.exists():
                return str(p)

        script_dir = Path(__file__).resolve().parent           # .../src/tools
        # In your tree: .../src/tools/voronota/voronota-lt
        candidate = script_dir / "voronota" / "voronota-lt"
        if candidate.exists():
            return str(candidate)

        # Fall back to PATH
        return "voronota-lt"


# ----------------------------
# Compatibility wrappers
# ----------------------------

class VoronotaAreas:
    """Drop-in replacement for original VoronotaAreas using optimized VoroContacts."""

    def __init__(self, pdb_path: str | Path, probe: float = 1.4, processors: Optional[int] = None):
        self._vc = VoroContacts(
            pdb_path,
            probe=probe,
            processors=processors,          # None -> env VORO_PROCESSORS (default 1)
            compute_contacts=True,
            compute_bsa=False,
        )
        self.contact_areas = self._vc.contact_areas

    @staticmethod
    def get_atom_key(atom) -> AtomKey:
        return (str(atom.chainID), str(atom.resSeq), str(atom.resName), str(atom.name))

    def get_contact_areas(self, atoms1, atoms2) -> np.float64:
        areas = []
        for at1 in atoms1:
            at1k = self.get_atom_key(at1)
            for at2 in atoms2:
                if getattr(at1, "residue", None) == getattr(at2, "residue", None):
                    areas.append(0.0)
                    continue
                at2k = self.get_atom_key(at2)
                area = (
                    self.contact_areas.get(at1k, {}).get(at2k, 0.0)
                    or self.contact_areas.get(at2k, {}).get(at1k, 0.0)
                )
                areas.append(float(area))
        return np.sum(np.asarray(areas, dtype=float))


class BSA:
    """Drop-in replacement for original BSA using optimized VoroContacts."""

    def __init__(
        self,
        pdb_data: str | Path,
        sqldb: Optional[Any] = None,
        chainA: str = "A",
        chainB: str = "B",
        probe: float = 1.4,
        processors: Optional[int] = None,
    ):
        from pdb2sql.interface import interface

        self.pdb_data = str(pdb_data)
        if sqldb is None:
            self.sql = interface(self.pdb_data)
            self._owns_sql = True
        else:
            self.sql = sqldb
            self._owns_sql = False

        self.chains_label = [chainA, chainB]
        self.probe = float(probe)
        self.processors = processors  # None -> env VORO_PROCESSORS (default 1)
        self._vc: Optional[VoroContacts] = None

        self.bsa_data: Dict[ResidueKey, list] = {}
        self.bsa_data_xyz: Dict[ResidueKey, tuple] = {}

    def close(self) -> None:
        if getattr(self, "_owns_sql", False) and getattr(self, "sql", None) is not None:
            try:
                self.sql.close()
            except Exception:
                pass

    def __del__(self):
        # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def get_structure(self) -> None:
        """Compute SAS for complex and chains."""
        self._vc = VoroContacts(
            self.pdb_data,
            probe=self.probe,
            processors=self.processors,
            compute_contacts=False,
            compute_bsa=True,
            chain_ids=tuple(self.chains_label),
        )

    def get_contact_residue_sasa(self, cutoff: float = 8.5) -> None:
        """
        Compute BSA for contact residues.

        Populates:
          self.bsa_data, self.bsa_data_xyz
        """
        if self._vc is None:
            raise RuntimeError("Call get_structure() first")

        contact_res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(contact_res.keys())
        if len(keys) < 2:
            # Nothing to do
            self.bsa_data = {}
            self.bsa_data_xyz = {}
            return

        all_residues = contact_res[keys[0]] + contact_res[keys[1]]

        self._vc.compute_bsa(all_residues, self.sql)
        self.bsa_data = self._vc.bsa_data
        self.bsa_data_xyz = self._vc.bsa_data_xyz

