"""
Unified Voronota-LT wrapper for contact areas and BSA computation.

Optimizations over separate VoroArea + BSA:
- Single voronota-lt call for contacts + cells (instead of 2 separate calls)
- Multi-threading via --processors flag
- Parallel chain calculations using concurrent.futures
- Reduces total subprocess calls from 4 to 3 (or 1 if BSA not needed)
"""

import subprocess
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class VoroContacts:
    """
    Unified voronota-lt wrapper for contact areas and SAS/BSA computation.

    Combines functionality of VoronotaAreas and BSA classes with optimizations:
    - Single call for contacts + complex SAS
    - Parallel chain SAS calculations
    - Configurable multi-threading
    """

    def __init__(
        self,
        pdb_path,
        probe=1.4,
        processors=4,
        compute_contacts=True,
        compute_bsa=True,
        chain_ids=('A', 'B'),
    ):
        """
        Initialize and compute contact areas and/or BSA.

        Args:
            pdb_path: Path to PDB file
            probe: Rolling probe radius in Angstroms (default: 1.4)
            processors: Number of OpenMP threads per voronota-lt call (default: 4)
            compute_contacts: Whether to compute atom-atom contacts (default: True)
            compute_bsa: Whether to compute BSA (default: True)
            chain_ids: Chain identifiers for BSA calculation (default: ('A', 'B'))
        """
        self.pdb_path = str(pdb_path)
        self.probe = probe
        self.processors = processors
        self.chain_ids = chain_ids
        self.voronota_exec = self._get_voronota_executable()

        # Results storage
        self.contact_areas = {}      # For VoroArea compatibility
        self.complex_sas = {}        # Residue-level SAS for complex
        self.chain_sas = {}          # Per-chain residue-level SAS
        self.bsa_data = {}           # BSA results (for BSA compatibility)
        self.bsa_data_xyz = {}       # BSA xyz keys (for BSA compatibility)

        # Run computations
        if compute_contacts and compute_bsa:
            self._run_combined()
        elif compute_contacts:
            self._run_contacts_only()
        elif compute_bsa:
            self._run_bsa_only()

    def _run_combined(self):
        """Run single voronota-lt call for contacts + complex SAS, then parallel chain SAS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            contacts_file = os.path.join(tmpdir, 'contacts.tsv')
            cells_file = os.path.join(tmpdir, 'cells.tsv')

            # Single call for contacts + complex cells
            cmd = [
                self.voronota_exec,
                '--input', self.pdb_path,
                '--probe', str(self.probe),
                '--processors', str(self.processors),
                '--write-contacts-to-file', contacts_file,
                '--write-cells-residue-level-to-file', cells_file,
                '--quiet',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"voronota-lt failed: {result.stderr}"
                )

            # Parse results
            with open(contacts_file, 'r') as f:
                self.contact_areas = self._parse_contacts(f.read())
            with open(cells_file, 'r') as f:
                self.complex_sas = self._parse_cells_residue(f.read())

        # Parallel chain SAS calculations
        self._run_chain_sas_parallel()

    def _run_contacts_only(self):
        """Run voronota-lt for contacts only."""
        cmd = [
            self.voronota_exec,
            '--input', self.pdb_path,
            '--probe', str(self.probe),
            '--processors', str(self.processors),
            '--print-contacts',
            '--quiet',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"voronota-lt failed: {result.stderr}")
        self.contact_areas = self._parse_contacts(result.stdout)

    def _run_bsa_only(self):
        """Run voronota-lt for BSA only (complex + chains)."""
        # Complex SAS
        cmd = [
            self.voronota_exec,
            '--input', self.pdb_path,
            '--probe', str(self.probe),
            '--processors', str(self.processors),
            '--print-cells-residue-level',
            '--quiet',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"voronota-lt failed: {result.stderr}")
        self.complex_sas = self._parse_cells_residue(result.stdout)

        # Parallel chain SAS
        self._run_chain_sas_parallel()

    def _run_chain_sas_parallel(self):
        """Run chain SAS calculations in parallel."""
        def compute_chain_sas(chain_id):
            cmd = [
                self.voronota_exec,
                '--input', self.pdb_path,
                '--probe', str(self.probe),
                '--processors', str(max(1, self.processors // 2)),  # Split processors
                '--restrict-input-atoms', f'[-chain {chain_id}]',
                '--print-cells-residue-level',
                '--quiet',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"voronota-lt failed for chain {chain_id}: {result.stderr}")
            return chain_id, self._parse_cells_residue(result.stdout)

        # Run chain calculations in parallel
        with ThreadPoolExecutor(max_workers=len(self.chain_ids)) as executor:
            futures = [executor.submit(compute_chain_sas, chain) for chain in self.chain_ids]
            for future in futures:
                chain_id, sas_data = future.result()
                self.chain_sas[chain_id] = sas_data

    def _parse_contacts(self, output: str) -> dict:
        """Parse voronota-lt contacts output."""
        contact_areas = {}
        lines = output.strip().split('\n')
        if not lines:
            return contact_areas

        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 12:
                continue

            # Key format: (chainID, resSeq, resName, atomName)
            at1_key = (parts[1], parts[2], parts[3], parts[4])
            at2_key = (parts[5], parts[6], parts[7], parts[8])
            area = float(parts[11])

            at1_areas = contact_areas.setdefault(at1_key, {})
            at1_areas[at2_key] = area

        return contact_areas

    def _parse_cells_residue(self, output: str) -> dict:
        """Parse voronota-lt residue-level cells output."""
        sas_data = {}
        lines = output.strip().split('\n')
        if not lines:
            return sas_data

        # Parse header
        header = lines[0].split('\t')
        try:
            chain_idx = header.index('ID_chain')
            rnum_idx = header.index('ID_rnum')
            rname_idx = header.index('ID_rname')
            sas_idx = header.index('sas_area')
        except ValueError:
            chain_idx, rnum_idx, rname_idx, sas_idx = 0, 1, 2, 4

        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) <= max(chain_idx, rnum_idx, rname_idx, sas_idx):
                continue

            key = (parts[chain_idx], int(parts[rnum_idx]), parts[rname_idx])
            sas_data[key] = float(parts[sas_idx])

        return sas_data

    def compute_bsa(self, contact_residues, sqldb=None):
        """
        Compute BSA for contact residues.

        Args:
            contact_residues: List of (chainID, resSeq, resName) tuples
            sqldb: Optional pdb2sql interface for xyz coordinates

        Populates self.bsa_data and self.bsa_data_xyz for compatibility.
        """
        for res in contact_residues:
            chain = res[0]
            resSeq = int(res[1])
            resName = res[2] if len(res) > 2 else None

            # Find matching key in complex_sas
            key = None
            for k in self.complex_sas.keys():
                if k[0] == chain and k[1] == resSeq:
                    key = k
                    if resName is None:
                        resName = k[2]
                    break

            if key is None:
                continue

            asa_complex = self.complex_sas.get(key, 0.0)
            asa_unbound = self.chain_sas.get(chain, {}).get(key, 0.0)
            bsa = asa_unbound - asa_complex

            result_key = (chain, resSeq, resName)
            self.bsa_data[result_key] = [bsa]

            # Compute xyz key if sqldb provided
            if sqldb is not None:
                chain_idx = {'A': 0, 'B': 1}.get(chain, 0)
                xyz = np.mean(sqldb.get('x,y,z', resSeq=resSeq, chainID=chain), axis=0)
                self.bsa_data_xyz[result_key] = tuple([chain_idx] + xyz.tolist())

    @staticmethod
    def _get_voronota_executable() -> str:
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent
        return str(root_dir / "tools" / "voronota" / "voronota-lt")


# Compatibility wrappers for existing code

class VoronotaAreas:
    """Drop-in replacement for original VoronotaAreas using optimized VoroContacts."""

    def __init__(self, pdb_path, probe=1.4, processors=4):
        self._vc = VoroContacts(
            pdb_path,
            probe=probe,
            processors=processors,
            compute_contacts=True,
            compute_bsa=False,
        )
        self.contact_areas = self._vc.contact_areas

    @staticmethod
    def get_atom_key(atom) -> tuple:
        return (
            str(atom.chainID),
            str(atom.resSeq),
            str(atom.resName),
            str(atom.name),
        )

    def get_contact_areas(self, atoms1, atoms2) -> np.float64:
        areas = []
        for at1 in atoms1:
            at1k = self.get_atom_key(at1)
            for at2 in atoms2:
                if at1.residue == at2.residue:
                    areas.append(0)
                    continue
                at2k = self.get_atom_key(at2)
                area = (
                    self.contact_areas.get(at1k, {}).get(at2k, 0.0) or
                    self.contact_areas.get(at2k, {}).get(at1k, 0.0)
                )
                areas.append(area)
        return np.sum(areas)


class BSA:
    """Drop-in replacement for original BSA using optimized VoroContacts."""

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B', probe=1.4, processors=4):
        from pdb2sql.interface import interface

        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
            self._owns_sql = True
        else:
            self.sql = sqldb
            self._owns_sql = False

        self.chains_label = [chainA, chainB]
        self.probe = probe
        self.processors = processors
        self._vc = None

        self.bsa_data = {}
        self.bsa_data_xyz = {}

    def get_structure(self):
        """Compute SAS for complex and chains."""
        if not isinstance(self.pdb_data, str):
            raise ValueError("BSA requires a PDB file path")

        self._vc = VoroContacts(
            self.pdb_data,
            probe=self.probe,
            processors=self.processors,
            compute_contacts=False,
            compute_bsa=True,
            chain_ids=tuple(self.chains_label),
        )

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute BSA for contact residues."""
        if self._vc is None:
            raise RuntimeError("Call get_structure() first")

        contact_res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(contact_res.keys())
        all_residues = contact_res[keys[0]] + contact_res[keys[1]]

        self._vc.compute_bsa(all_residues, self.sql)
        self.bsa_data = self._vc.bsa_data
        self.bsa_data_xyz = self._vc.bsa_data_xyz
