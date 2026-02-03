import subprocess
from pathlib import Path
import numpy as np
from pdb2sql.interface import interface


class BSA:
    """
    Compute Buried Surface Area (BSA) using Voronota-LT.

    BSA = SAS(unbound chain) - SAS(complex)

    This replaces the freesasa-based implementation with voronota-lt,
    which is faster and already used for contact area computation.
    """

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B', probe=1.4):
        """
        Initialize BSA calculator.

        Args:
            pdb_data: PDB filename (str) or pdb2sql interface
            sqldb: Optional existing pdb2sql interface instance
            chainA: First chain ID (default: 'A')
            chainB: Second chain ID (default: 'B')
            probe: Rolling probe radius in Angstroms (default: 1.4)
        """
        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
            self._owns_sql = True
        else:
            self.sql = sqldb
            self._owns_sql = False

        self.chains_label = [chainA, chainB]
        self.probe = probe
        self.voronota_exec = self._get_voronota_executable()

        # Will be populated by get_structure()
        self.complex_sas = None
        self.chain_sas = None

        # Will be populated by get_contact_residue_sasa()
        self.bsa_data = {}
        self.bsa_data_xyz = {}

    def get_structure(self):
        """
        Compute SAS for complex and individual chains using voronota-lt.

        Runs voronota-lt 3 times:
        - Once on full complex
        - Once per chain (with atom restriction)
        """
        # Get PDB path
        if isinstance(self.pdb_data, str):
            pdb_path = self.pdb_data
        else:
            raise ValueError("BSA with voronota-lt requires a PDB file path")

        # 1. Compute SAS for full complex
        self.complex_sas = self._run_voronota_cells(pdb_path)

        # 2. Compute SAS for each chain separately
        self.chain_sas = {}
        for chain in self.chains_label:
            self.chain_sas[chain] = self._run_voronota_cells(
                pdb_path,
                restrict_atoms=f"[-chain {chain}]"
            )

    def _run_voronota_cells(self, pdb_path, restrict_atoms=None):
        """
        Run voronota-lt to get residue-level SAS areas.

        Args:
            pdb_path: Path to PDB file
            restrict_atoms: Optional atom restriction expression (e.g., "[-chain A]")

        Returns:
            Dict mapping (chainID, resSeq, resName) -> sas_area
        """
        cmd = [
            self.voronota_exec,
            '--input', str(pdb_path),
            '--probe', str(self.probe),
            '--print-cells-residue-level',
            '--quiet',
        ]
        if restrict_atoms:
            cmd.extend(['--restrict-input-atoms', restrict_atoms])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"voronota-lt failed with return code {result.returncode}: "
                f"{result.stderr}"
            )

        return self._parse_cells_residue_level(result.stdout)

    def _parse_cells_residue_level(self, output: str) -> dict:
        """
        Parse voronota-lt residue-level cells output.

        Expected format (tab-separated):
        ID_chain  ID_rnum  ID_rname  total_area  sas_area  volume  ...

        Returns:
            Dict mapping (chainID, resSeq, resName) -> sas_area
        """
        sas_data = {}
        lines = output.strip().split('\n')

        if not lines:
            return sas_data

        # Parse header to find column indices
        header = lines[0].split('\t')
        try:
            chain_idx = header.index('ID_chain')
            rnum_idx = header.index('ID_rnum')
            rname_idx = header.index('ID_rname')
            sas_idx = header.index('sas_area')
        except ValueError as e:
            # Fallback to positional indices if header parsing fails
            chain_idx, rnum_idx, rname_idx, sas_idx = 0, 1, 2, 4

        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) <= max(chain_idx, rnum_idx, rname_idx, sas_idx):
                continue

            chain = parts[chain_idx]
            resSeq = int(parts[rnum_idx])
            resName = parts[rname_idx]
            sas_area = float(parts[sas_idx])

            key = (chain, resSeq, resName)
            sas_data[key] = sas_area

        return sas_data

    def get_contact_residue_sasa(self, cutoff=8.5):
        """
        Compute BSA for contact residues.

        BSA = SAS(unbound) - SAS(complex)

        Args:
            cutoff: Distance cutoff for contact residues (default: 8.5 Å)
        """
        if self.complex_sas is None or self.chain_sas is None:
            raise RuntimeError("Call get_structure() before get_contact_residue_sasa()")

        self.bsa_data = {}
        self.bsa_data_xyz = {}

        # Get contact residues from pdb2sql
        contact_res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(contact_res.keys())
        all_residues = contact_res[keys[0]] + contact_res[keys[1]]

        # Batch query xyz coordinates for efficiency
        xyz_cache = {}
        for r in all_residues:
            chain, resSeq = r[0], r[1]
            xyz = np.mean(
                self.sql.get('x,y,z', resSeq=resSeq, chainID=chain), axis=0
            )
            xyz_cache[(chain, resSeq)] = xyz

        # Compute BSA for each contact residue
        for r in all_residues:
            chain, resSeq = r[0], r[1]

            # Get resName - r might be (chain, resSeq) or (chain, resSeq, resName)
            if len(r) >= 3:
                resName = r[2]
            else:
                # Query resName from database
                resName = self.sql.get('resName', resSeq=resSeq, chainID=chain)[0]

            key = (chain, resSeq, resName)

            # SAS in complex
            asa_complex = self.complex_sas.get(key, 0.0)

            # SAS when unbound (chain only)
            asa_unbound = self.chain_sas[chain].get(key, 0.0)

            # BSA = unbound - complex
            bsa = asa_unbound - asa_complex

            # Create xyz key for compatibility: (chain_idx, x, y, z)
            chain_idx = {'A': 0, 'B': 1}.get(chain, 0)
            xyz = xyz_cache[(chain, resSeq)]
            xyzkey = tuple([chain_idx] + xyz.tolist())

            # Store results (as list for compatibility with original interface)
            self.bsa_data[key] = [bsa]
            self.bsa_data_xyz[key] = xyzkey

    @staticmethod
    def _get_voronota_executable() -> str:
        """Get path to voronota-lt executable."""
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent
        voronota_exec = root_dir / "tools" / "voronota" / "voronota-lt"
        return str(voronota_exec)


# Legacy freesasa-based implementation for fallback/comparison
class BSA_Freesasa:
    """
    Original freesasa-based BSA implementation.
    Kept for comparison/fallback purposes.
    """

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B'):
        try:
            import freesasa
            self.freesasa = freesasa
        except ImportError:
            raise ImportError(
                "freesasa not found. Install with: pip install freesasa"
            )

        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
        else:
            self.sql = sqldb
        self.chains_label = [chainA, chainB]

        freesasa.setVerbosity(freesasa.nowarnings)

    def get_structure(self):
        """Prepare freesasa Structure objects for complex and chains."""
        # Full complex
        if isinstance(self.pdb_data, str):
            self.complex = self.freesasa.Structure(self.pdb_data)
        else:
            self.complex = self.freesasa.Structure()
            atomdata = self.sql.get('name,resName,resSeq,chainID,x,y,z')
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName)  # Fixed: was atomName[0]
                self.complex.addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )

        self.result_complex = self.freesasa.calc(self.complex)

        # Individual chains
        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = self.freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z', chainID=label
            )
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName)  # Fixed: was atomName[0]
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )
            self.result_chains[label] = self.freesasa.calc(self.chains[label])

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute BSA for contact residues using freesasa."""
        self.bsa_data = {}
        self.bsa_data_xyz = {}

        res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(res.keys())
        res = res[keys[0]] + res[keys[1]]

        for r in res:
            chain, resSeq = r[0], r[1]
            resName = r[2] if len(r) >= 3 else self.sql.get(
                'resName', resSeq=resSeq, chainID=chain
            )[0]

            # SAS in complex
            select_str = ('res, (resi %d) and (chain %s)' % (resSeq, chain),)
            asa_complex = self.freesasa.selectArea(
                select_str, self.complex, self.result_complex
            )['res']

            # SAS when unbound
            select_str = ('res, resi %d' % resSeq,)
            asa_unbound = self.freesasa.selectArea(
                select_str, self.chains[chain], self.result_chains[chain]
            )['res']

            bsa = asa_unbound - asa_complex

            chain_idx = {'A': 0, 'B': 1}.get(chain, 0)
            xyz = np.mean(
                self.sql.get('x,y,z', resSeq=resSeq, chainID=chain), axis=0
            )
            xyzkey = tuple([chain_idx] + xyz.tolist())

            key = (chain, resSeq, resName)
            self.bsa_data[key] = [bsa]
            self.bsa_data_xyz[key] = xyzkey
