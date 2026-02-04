# Select voronota binary for contact area computation.
# VORONOTA_BINARY options:
#   - "voronota"     (default) - legacy v1.28, additive tessellation, validated
#   - "voronota_129" or "voronota-opt" - v1.29, additive tessellation
#   - "voronota-lt"  - radical tessellation, faster but less accurate
# Legacy USE_VORONOTA_LT is still supported for backwards compatibility.
import os

def _get_voronota_binary():
    """Determine which voronota binary to use."""
    binary = os.environ.get("VORONOTA_BINARY", "").strip().lower()
    if binary:
        return binary
    # Fallback to legacy USE_VORONOTA_LT for backwards compatibility
    use_lt = os.environ.get("USE_VORONOTA_LT", "0").strip()
    if use_lt not in ("0", "false", "no", "off", ""):
        return "voronota-lt"
    return "voronota"

_VORONOTA_BINARY = _get_voronota_binary()
_USE_LT = _VORONOTA_BINARY == "voronota-lt"

import subprocess
from pathlib import Path
import numpy as np


class VoronotaAreasLegacy:
    """
    Original two-step voronota implementation using legacy voronota binary.

    Uses additively weighted Voronoi tessellation (bisecting planes shifted by radii).
    This is the scientifically validated algorithm from the original DeepRank.

    Pipeline: voronota get-balls-from-atoms-file | voronota calculate-contacts
    """

    def __init__(self, pdb_path, probe=1.4):
        self.voronota_exec = self.get_voronota_executable()
        self.probe = probe
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
        """
        Parse balls output: 'x y z r # atomID chainID resSeq resName atomName'.

        Returns 4-tuple keys (chainID, resSeq, resName, atomName) to match
        the interface expected by AtomGraph and ResidueGraph.
        """
        lines = _balls_data.decode("utf-8")
        balls_data = {}
        for index, line in enumerate(lines.split("\n")):
            if not line.strip():
                continue
            # Format: x y z r # atomID chainID resSeq resName atomName
            atom_dt = line.split("#")[-1].strip().split(" ")
            # Skip atomID (index 0), use chainID, resSeq, resName, atomName
            if len(atom_dt) >= 5:
                desired_at_dt = (atom_dt[1], atom_dt[2], atom_dt[3], atom_dt[4])
            else:
                continue
            balls_data[index] = desired_at_dt
        return balls_data

    @staticmethod
    def get_voronota_executable() -> str:
        """Get path to voronota binary based on VORONOTA_BINARY env var."""
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent
        voronota_dir = root_dir / "tools" / "voronota"

        # Map VORONOTA_BINARY to actual binary name
        binary_map = {
            "voronota": "voronota",
            "voronota_129": "voronota-opt",
            "voronota-opt": "voronota-opt",
        }
        binary_name = binary_map.get(_VORONOTA_BINARY, "voronota")
        return str(voronota_dir / binary_name)

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


# Conditional export based on USE_VORONOTA_LT environment variable
if _USE_LT:
    from tools.VoroContacts import VoronotaAreas
else:
    VoronotaAreas = VoronotaAreasLegacy
