#!/usr/bin/env python3
"""
forcefield_dr2.py

Parse a simple force field layout (topology, VDW params, residue classes, patches)
and expose a minimal interface to get atomic charges and van der Waals parameters.
"""

import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

_log = logging.getLogger(__name__)

# --- .param parsing (Lennard-Jones) ---

class VanderwaalsParam:
    """Nonbonded parameters for Lennard-Jones interactions."""
    def __init__(
        self,
        epsilon_main: float,
        sigma_main: float,
        epsilon_14: float,
        sigma_14: float,
    ):
        self.epsilon_main = epsilon_main
        self.sigma_main   = sigma_main
        self.epsilon_14   = epsilon_14
        self.sigma_14     = sigma_14

    def __str__(self) -> str:
        return f"{self.epsilon_main}, {self.sigma_main}, {self.epsilon_14}, {self.sigma_14}"


class ParamParser:
    """Parse vdw parameters from a .param file."""
    @staticmethod
    def parse(path: str) -> Dict[str, VanderwaalsParam]:
        params: Dict[str, VanderwaalsParam] = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('NONBonded'):
                    _, type_, eps_m, sig_m, eps_14, sig_14 = line.split()
                    params[type_] = VanderwaalsParam(
                        float(eps_m), float(sig_m),
                        float(eps_14), float(sig_14),
                    )
                else:
                    raise ValueError(f"unparsable param line: {line}")
        return params


# --- .top parsing (atom types, charges) ---

class TopRow:
    """One atom entry from the topology."""
    def __init__(self, residue: str, atom: str, properties: Dict[str, Union[float, str]]):
        self.residue_name = residue
        self.atom_name    = atom
        self.properties   = properties

    def __getitem__(self, key: str) -> Union[float, str]:
        return self.properties[key]


class TopParser:
    """Parse atom types and charges from a .top file."""
    _LINE_RE = re.compile(r'^([A-Z0-9]{3})\s+atom\s+([A-Z0-9]{1,4})\s+(.+?)\s+end')
    _VAR_RE  = re.compile(r'([^\s]+)\s*=\s*([^\s\(\)]+|\(.*\))')
    _NUM_RE  = re.compile(r'^-?[0-9]+(?:\.[0-9]+)?$')

    @staticmethod
    def parse(path: str) -> List[TopRow]:
        rows: List[TopRow] = []
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                m = TopParser._LINE_RE.match(line)
                if not m:
                    continue
                res_name, atom_name, rest = m.groups()
                props: Dict[str, Union[float, str]] = {}
                for vm in TopParser._VAR_RE.finditer(rest):
                    k, v = vm.groups()
                    v = v.strip()
                    if v.startswith('(') and v.endswith(')'):
                        v = v[1:-1]
                    if TopParser._NUM_RE.match(v):
                        props[k.lower()] = float(v)
                    else:
                        props[k.lower()] = v
                rows.append(TopRow(res_name.upper(), atom_name.upper(), props))
        return rows


# --- residue-classes parsing (presence/absence rules) ---

class ResidueClassRule:
    """Atom presence/absence rules for a residue class."""
    def __init__(
        self,
        class_name: str,
        aa_names: Union[str, List[str]],
        present: List[str],
        absent: List[str],
    ):
        self.class_name         = class_name
        self.amino_acid_names   = aa_names
        self.present_atom_names = present
        self.absent_atom_names  = absent

    def matches(self, aa: str, atom_names: List[str]) -> bool:
        # check amino-acid name
        if self.amino_acid_names != 'all' and aa not in (
            self.amino_acid_names
            if isinstance(self.amino_acid_names, list)
            else [self.amino_acid_names]
        ):
            return False
        # none of the forbidden atoms should be present
        if any(a in self.absent_atom_names for a in atom_names):
            return False
        # all required atoms should be present
        return all(a in atom_names for a in self.present_atom_names)


class ResidueClassParser:
    """Parse residue-classes file with atom-list rules."""
    _HEADER_RE = re.compile(r'^([A-Z]{3,4})\s*:\s*name\s*=\s*(all|[A-Z]{3})')
    _ATOM_RE   = re.compile(r'(present|absent)\(([A-Z0-9, ]+)\)')

    @staticmethod
    def parse(path: str) -> List[ResidueClassRule]:
        rules: List[ResidueClassRule] = []
        with open(path, 'r') as f:
            for line in f:
                m = ResidueClassParser._HEADER_RE.match(line)
                if not m:
                    continue
                cls, aa = m.groups()
                aa_list = aa if aa == 'all' else [x.strip() for x in aa.split(',')]
                present, absent = [], []
                tail = line[m.end():]
                for mm in ResidueClassParser._ATOM_RE.finditer(tail):
                    typ, names = mm.groups()
                    items = [n.strip() for n in names.split(',')]
                    if typ == 'present':
                        present.extend(items)
                    else:
                        absent.extend(items)
                rules.append(ResidueClassRule(cls, aa_list, present, absent))
        return rules


# --- patch parsing (MODIFY/ADD/DELETE ATOM) ---

class PatchActionType(Enum):
    MODIFY = 1
    ADD    = 2
    DELETE = 3


class PatchSelection:
    def __init__(self, residue_type: str, atom_name: str):
        self.residue_type = residue_type
        self.atom_name    = atom_name


class PatchAction:
    def __init__(
        self,
        action_type: PatchActionType,
        selection: PatchSelection,
        params: Dict[str, Any]
    ):
        self.type      = action_type
        self.selection = selection
        self.params    = params

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def __getitem__(self, key: str) -> Any:
        return self.params[key]


class PatchParser:
    """Parse patch.top lines for ATOM actions."""
    STRING_VAR_PATTERN = re.compile(r"([A-Z]+)=([A-Z0-9]+)")
    NUMBER_VAR_PATTERN = re.compile(r"([A-Z]+)=(-?[0-9]+\.[0-9]+)")
    ACTION_PATTERN     = re.compile(
        r"^([A-Z]{3,4})\s+([A-Z]+)\s+ATOM\s+([A-Z0-9]{1,3})\s+(.*)$"
    )

    @staticmethod
    def _parse_action_type(s: str) -> PatchActionType:
        for t in PatchActionType:
            if t.name == s:
                return t
        raise ValueError(f"unmatched patch action: {s!r}")

    @staticmethod
    def parse(path: str) -> List[PatchAction]:
        actions: List[PatchAction] = []
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith(('#', '!')):
                    continue
                m = PatchParser.ACTION_PATTERN.match(line)
                if not m:
                    raise ValueError(f"unmatched patch action: {line!r}")
                residue, act, atom, rest = m.groups()
                action_type = PatchParser._parse_action_type(act)
                sel = PatchSelection(residue, atom)
                params: Dict[str, Any] = {}
                # parse string and numeric vars from the tail
                for w in PatchParser.STRING_VAR_PATTERN.finditer(rest):
                    params[w.group(1)] = w.group(2)
                for w in PatchParser.NUMBER_VAR_PATTERN.finditer(rest):
                    params[w.group(1)] = float(w.group(2))
                actions.append(PatchAction(action_type, sel, params))
        return actions


# --- force-field interface ---

class AtomicForcefield:
    """Load parameters and apply patch overrides."""
    def __init__(self, ff_dir: str):
        top_path    = os.path.join(ff_dir, 'protein-allhdg5-5_new.top')
        patch_path  = os.path.join(ff_dir, 'patch.top')
        rc_path     = os.path.join(ff_dir, 'residue-classes')
        param_path  = os.path.join(ff_dir, 'protein-allhdg5-4_new.param')

        self._top_rows = {
            (r.residue_name, r.atom_name): r
            for r in TopParser.parse(top_path)
        }
        self._patch_actions  = PatchParser.parse(patch_path)
        self._residue_rules  = ResidueClassParser.parse(rc_path)
        self._vdw_parameters = ParamParser.parse(param_path)

    def _find_residue_class(
        self,
        residue_name: str,
        atom_names: List[str]
    ) -> Optional[str]:
        for rule in self._residue_rules:
            if rule.matches(residue_name, atom_names):
                return rule.class_name
        return None

    def get_vanderwaals(
        self,
        residue_name: str,
        atom_name: str,
        atom_list: List[str]
    ) -> VanderwaalsParam:
        """Return vdw parameters for residue+atom, with patch override."""
        key = (residue_name, atom_name)
        row = self._top_rows.get(key)
        if not row:
            _log.warning(f"unknown atom {key} in TOP; returning zero vdw")
            return VanderwaalsParam(0, 0, 0, 0)

        type_ = row['type']
        cls   = self._find_residue_class(residue_name, atom_list)
        if cls:
            for act in self._patch_actions:
                if (act.type in (PatchActionType.MODIFY, PatchActionType.ADD)
                    and act.selection.residue_type == cls
                    and act.selection.atom_name    == atom_name
                    and 'TYPE' in act):
                    type_ = act['TYPE']

        return self._vdw_parameters.get(type_, VanderwaalsParam(0, 0, 0, 0))

    def get_charge(
        self,
        residue_name: str,
        atom_name: str,
        atom_list: List[str]
    ) -> float:
        """Return atomic charge for residue+atom, with patch override."""
        key = (residue_name, atom_name)
        row = self._top_rows.get(key)
        charge: Optional[float] = None
        if row and 'charge' in row.properties:
            charge = float(row.properties['charge'])

        cls = self._find_residue_class(residue_name, atom_list)
        if cls:
            for act in self._patch_actions:
                if (act.type in (PatchActionType.MODIFY, PatchActionType.ADD)
                    and act.selection.residue_type == cls
                    and act.selection.atom_name    == atom_name
                    and 'CHARGE' in act):
                    charge = float(act['CHARGE'])

        if charge is None:
            _log.warning(f"unknown atom {key} charge; returning zero")
            return 0.0
        return charge


# convenience alias
atomic_forcefield = AtomicForcefield
