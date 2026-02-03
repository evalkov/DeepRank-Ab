import subprocess
from pathlib import Path
import numpy as np


class VoronotaAreas:
    """
    Compute atom-atom contact areas using Voronota-LT.

    Uses a single voronota-lt subprocess call instead of the legacy two-step
    voronota pipeline (get-balls-from-atoms-file | calculate-contacts).
    This is 2-3x faster due to:
    - Single process invocation
    - Radical tessellation (default) instead of additively weighted
    - Optional multi-threading via OpenMP
    """

    def __init__(self, pdb_path, probe=1.4, processors=1):
        """
        Initialize and compute contact areas for a PDB structure.

        Args:
            pdb_path: Path to PDB file
            probe: Rolling probe radius in Angstroms (default: 1.4)
            processors: Number of OpenMP threads (default: 1)
        """
        self.voronota_exec = self.get_voronota_executable()
        self.probe = probe
        self.processors = processors
        self.contact_areas = self._get_voronota_contacts(pdb_path)

    def _get_voronota_contacts(self, pdb_path) -> dict[tuple, dict[tuple, float]]:
        """
        Run voronota-lt and parse contact areas.

        Returns:
            Nested dict mapping atom keys to their contact partners and areas.
            Key format: (chainID, resSeq, resName, atomName)
        """
        result = subprocess.run(
            [
                self.voronota_exec,
                '--input', str(pdb_path),
                '--probe', str(self.probe),
                '--processors', str(self.processors),
                '--print-contacts',
                '--quiet',
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"voronota-lt failed with return code {result.returncode}: "
                f"{result.stderr}"
            )

        return self._parse_lt_contacts(result.stdout)

    def _parse_lt_contacts(self, output: str) -> dict[tuple, dict[tuple, float]]:
        """
        Parse voronota-lt tab-separated contact output.

        Voronota-lt output format (tab-separated):
        ia  ID1_chain  ID1_rnum  ID1_rname  ID1_atom  ID2_chain  ID2_rnum  ID2_rname  ID2_atom  ID1_index  ID2_index  area  arc_length  distance

        Returns:
            Nested dict: {atom1_key: {atom2_key: area, ...}, ...}
            Key format: (chainID, resSeq, resName, atomName)
        """
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

            # Extract atom identifiers: (chainID, resSeq, resName, atomName)
            # Columns: 1=ID1_chain, 2=ID1_rnum, 3=ID1_rname, 4=ID1_atom
            #          5=ID2_chain, 6=ID2_rnum, 7=ID2_rname, 8=ID2_atom
            #          11=area
            at1_key = (parts[1], parts[2], parts[3], parts[4])
            at2_key = (parts[5], parts[6], parts[7], parts[8])
            area = float(parts[11])

            at1_areas = contact_areas.setdefault(at1_key, {})
            at1_areas[at2_key] = area

        return contact_areas

    @staticmethod
    def get_voronota_executable() -> str:
        """Get path to voronota-lt executable."""
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent
        voronota_exec = root_dir / "tools" / "voronota" / "voronota-lt"
        return str(voronota_exec)

    def get_contact_areas(self, atoms1, atoms2) -> np.float64:
        """
        Sum contact areas between two sets of atoms.

        Args:
            atoms1: Iterable of atom objects with .residue attribute
            atoms2: Iterable of atom objects with .residue attribute

        Returns:
            Total contact area between the atom sets
        """
        areas = []
        for at1 in atoms1:
            at1k = self.get_atom_key(at1)
            for at2 in atoms2:
                # Skip contacts within same residue
                if at1.residue == at2.residue:
                    areas.append(0)
                    continue

                at2k = self.get_atom_key(at2)
                # Try both key orderings
                area = (
                    self.contact_areas.get(at1k, {}).get(at2k, 0.0) or
                    self.contact_areas.get(at2k, {}).get(at1k, 0.0)
                )
                areas.append(area)

        return np.sum(areas)

    @staticmethod
    def get_atom_key(atom) -> tuple:
        """
        Create a lookup key for an atom object.

        Args:
            atom: Atom object with chainID, resSeq, resName, name attributes

        Returns:
            Tuple key: (chainID, resSeq, resName, atomName)
        """
        return (
            str(atom.chainID),
            str(atom.resSeq),
            str(atom.resName),
            str(atom.name),
        )


# Legacy class for backwards compatibility with original voronota binary
class VoronotaAreasLegacy:
    """
    Original two-step voronota implementation.
    Kept for comparison/fallback purposes.
    """

    def __init__(self, pdb_path):
        self.voronota_exec = self.get_voronota_executable()
        self.contact_areas = self._get_voronota_contacts(pdb_path)

    def _get_voronota_contacts(self, pdb_path) -> dict[tuple, dict[tuple, float]]:
        balls_outputs, contacts_outputs = self.run_voro_contacts(
            pdb_path, self.voronota_exec
        )
        balls_data = self.load_balls(balls_outputs)
        contacts_data = self.load_contacts(contacts_outputs)

        contact_areas = {
            balls_data[at1_index]: {
                balls_data[at2_index]: contact_area
                for at2_index, contact_area in at1_contact_areas.items()
            }
            for at1_index, at1_contact_areas in contacts_data.items()
        }
        return contact_areas

    @staticmethod
    def load_contacts(_contacts_outputs) -> dict[int, dict[int, float]]:
        """Parse contacts output: 'b1 b2 area' format."""
        lines = _contacts_outputs.decode("utf-8")
        contacts_data = {}
        for line in lines.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(" ")
            at1_index = int(parts[0])
            at2_index = int(parts[1])
            area = float(parts[2])
            at1_areas = contacts_data.setdefault(at1_index, {})
            at1_areas[at2_index] = area
        return contacts_data

    @staticmethod
    def load_balls(_balls_data) -> dict[int, tuple]:
        """Parse balls output: 'x y z r # atomID chainID resSeq resName atomName'."""
        lines = _balls_data.decode("utf-8")
        balls_data = {}
        for index, line in enumerate(lines.split("\n")):
            if not line.strip():
                continue
            atom_dt = line.split("#")[-1].strip().split(" ")
            desired_at_dt = tuple(atom_dt[:5])
            balls_data[index] = desired_at_dt
        return balls_data

    @staticmethod
    def get_voronota_executable() -> str:
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent
        voronota_exec = root_dir / "tools" / "voronota" / "voronota"
        return str(voronota_exec)

    @staticmethod
    def run_voro_contacts(pdb_fpath, voronota_exec) -> tuple[bytes, bytes]:
        """Run two-step voronota pipeline."""
        with open(pdb_fpath, "rb") as fin:
            pdb_lines = fin.read()

        balls = subprocess.Popen(
            [voronota_exec, "get-balls-from-atoms-file"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        balls_outputs, _ = balls.communicate(input=pdb_lines)

        contacts = subprocess.Popen(
            [voronota_exec, "calculate-contacts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        contacts_outputs, _ = contacts.communicate(input=balls_outputs)

        return balls_outputs, contacts_outputs
