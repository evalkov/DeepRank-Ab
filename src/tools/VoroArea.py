# Re-export optimized VoronotaAreas from VoroContacts
from tools.VoroContacts import VoronotaAreas

# Legacy implementation below for fallback to original two-step voronota binary

import subprocess
from pathlib import Path
import numpy as np


class VoronotaAreasLegacy:
    """
    Original two-step voronota implementation using legacy voronota binary.
    Kept for comparison/fallback purposes.

    Pipeline: voronota get-balls-from-atoms-file | voronota calculate-contacts
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
