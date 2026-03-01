"""Reader for .qv / .qv.idx quiver files.

Quiver files are concatenated PDB archives produced by RFdiffusion/Rosetta
pipelines.  Each structure is preceded by QV_TAG / QV_SCORE header lines and
may be followed by a Rosetta pose-energy table.  A companion .qv.idx file
maps tag names to byte offsets for random access.

Usage
-----
    reader = QuiverReader("structures.qv")
    tags   = reader.tags()                  # all tag names
    pdb    = reader.extract_pdb(0)          # clean PDB text (str)
    score  = reader.scores(0)               # {"ddg": -14.07, ...}

    # write a single structure to disk
    reader.extract_to_file(0, "/tmp/design_20_best.pdb")
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------- helpers for line filtering --------------------------------

def _is_hydrogen(line: str) -> bool:
    """Return True if an ATOM/HETATM line describes a hydrogen."""
    # PDB element symbol lives in columns 77-78 (0-indexed 76:78).
    if len(line) >= 78:
        return line[76:78].strip() == "H"
    # Fallback: Rosetta-style names like 1H, 2HB, 3HG1 in column 13-16.
    name = line[12:16].strip()
    if name and name[0].isdigit():
        return len(name) > 1 and name[1] == "H"
    return name.startswith("H")


def _is_oxt(line: str) -> bool:
    """Return True if an ATOM/HETATM line is an OXT pseudo-atom."""
    return line[12:16].strip() == "OXT"


# --------------- data classes ---------------------------------------------

@dataclass
class QuiverEntry:
    """One structure inside a quiver archive."""
    tag: str
    byte_offset: int
    scores: Dict[str, float] = field(default_factory=dict)


# --------------- reader ---------------------------------------------------

class QuiverReader:
    """Random-access reader for .qv + .qv.idx quiver archives."""

    def __init__(self, qv_path: str, idx_path: str | None = None):
        self.qv_path = qv_path
        self.idx_path = idx_path or qv_path + ".idx"
        self._entries: List[QuiverEntry] = []
        self._sorted_offsets: List[int] = []
        self._tag_to_index: Dict[str, int] = {}
        self._load_index()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def tags(self) -> List[str]:
        """Return all tag names in index order."""
        return [e.tag for e in self._entries]

    def find(self, tag: str) -> int:
        """Return entry index for *tag*, or -1 if not found."""
        return self._tag_to_index.get(tag, -1)

    def entry(self, index: int) -> QuiverEntry:
        return self._entries[index]

    def extract_pdb(self, index: int) -> str:
        """Return clean PDB text for entry *index*.

        Strips QV headers, Rosetta energy tables, hydrogens, and OXT.
        Ensures the text ends with an END record.
        """
        start = self._entries[index].byte_offset
        end = self._find_end_offset(start)

        lines: list[str] = []
        in_energy_table = False
        past_first_tag = False
        has_end = False

        with open(self.qv_path, "r") as fh:
            fh.seek(start)
            while True:
                raw = fh.readline()
                if not raw:
                    break

                # Respect byte boundary
                if end > 0 and fh.tell() > end:
                    break

                # QV_TAG — skip ours, stop at the next entry's
                if raw.startswith("QV_TAG"):
                    if past_first_tag:
                        break
                    past_first_tag = True
                    continue
                past_first_tag = True

                # QV_SCORE — parse and skip
                if raw.startswith("QV_SCORE"):
                    self._parse_score_line(raw, self._entries[index])
                    continue

                # Rosetta energy table
                if raw.startswith("#BEGIN_POSE_ENERGIES_TABLE") or (
                    raw.startswith("label ") and "fa_atr" in raw
                ):
                    in_energy_table = True
                    continue
                if in_energy_table:
                    if raw.startswith("#END_POSE_ENERGIES_TABLE"):
                        in_energy_table = False
                    continue

                # Skip comment lines (e.g. "# All scores below ...")
                if raw.startswith("# "):
                    continue

                # Filter ATOM/HETATM lines
                rec = raw[:6]
                if rec == "ATOM  " or rec == "HETATM":
                    if _is_hydrogen(raw) or _is_oxt(raw):
                        continue

                # Track END
                stripped = raw.strip()
                if stripped == "END":
                    has_end = True

                lines.append(raw)

        if not has_end:
            lines.append("END\n")

        return "".join(lines)

    def extract_to_file(self, index: int, out_path: str) -> str:
        """Write clean PDB for entry *index* to *out_path*. Returns path."""
        pdb_text = self.extract_pdb(index)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(pdb_text)
        os.replace(tmp, out_path)
        return out_path

    def scores(self, index: int) -> Dict[str, float]:
        """Return parsed QV_SCORE dict for entry *index*.

        If scores haven't been parsed yet (extract_pdb not called),
        triggers a lightweight parse of just the score line.
        """
        entry = self._entries[index]
        if not entry.scores:
            self._read_score_only(index)
        return dict(entry.scores)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        with open(self.idx_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                tag = parts[0]
                offset = int(parts[1])
                self._entries.append(QuiverEntry(tag=tag, byte_offset=offset))

        self._tag_to_index = {e.tag: i for i, e in enumerate(self._entries)}
        self._sorted_offsets = sorted(e.byte_offset for e in self._entries)

    def _find_end_offset(self, start: int) -> int:
        """Return byte offset of the next entry after *start*, or -1 for last."""
        pos = bisect.bisect_right(self._sorted_offsets, start)
        if pos < len(self._sorted_offsets):
            return self._sorted_offsets[pos]
        return -1

    @staticmethod
    def _parse_score_line(line: str, entry: QuiverEntry) -> None:
        """Parse 'QV_SCORE tag key=val|key=val|...' into entry.scores."""
        parts = line.split(None, 2)  # ["QV_SCORE", tag, rest]
        if len(parts) < 3:
            return
        for kv in parts[2].strip().split("|"):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                entry.scores[k] = float(v)
            except ValueError:
                entry.scores[k] = v  # keep as string if not numeric

    def _read_score_only(self, index: int) -> None:
        """Seek to entry and read just the QV_SCORE line."""
        start = self._entries[index].byte_offset
        with open(self.qv_path, "r") as fh:
            fh.seek(start)
            for raw in fh:
                if raw.startswith("QV_SCORE"):
                    self._parse_score_line(raw, self._entries[index])
                    return
                # Stop after a few lines — score is always near the top
                if raw.startswith("ATOM") or raw.startswith("HETATM"):
                    return
