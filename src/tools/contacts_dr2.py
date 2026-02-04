#!/usr/bin/env python3
"""
contacts.py

Compute interatomic distances and nonbonded interaction energies (electrostatic and
van der Waals), then aggregate these quantities to residue–residue edges on a
NetworkX graph.
"""

import os
import numpy as np
from scipy.spatial import distance_matrix
from typing import NamedTuple, List
import sys 
from pathlib import Path

# Force-field interface
from .forcefield_dr2 import AtomicForcefield

#irgnore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------
# Physical constants and distance cutoffs (Å)
# ------------------------------------------------------------
EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636

COVALENT_CUTOFF = 2.1
CUTOFF_13 = 3.6
CUTOFF_14 = 4.2

# ------------------------------------------------------------
# Force-field directory
# Can be overridden with the environment variable FF_DIR
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FF_DIR = SCRIPT_DIR / "forcefield"


if not os.path.isdir(FF_DIR):
    raise RuntimeError(f"force-field directory not found: {FF_DIR}")
else:
    #print(f"using force field at {FF_DIR}")
    pass


class Atom(NamedTuple):
    """Minimal atom record used for contact calculations."""
    chain: str
    res_name: str              # e.g., "ARG"
    res_seq: str                # e.g., "45"
    atom_name: str             # e.g., "CA"
    all_atom_names: List[str]  # all atom names in the residue
    pos: np.ndarray            # 3D coordinates


def compute_nonbonded(atoms: List[Atom], ff: AtomicForcefield):
    """
    Build NxN matrices of:
      - distances (Å)
      - electrostatic energy (Coulomb)
      - van der Waals energy (Lennard-Jones)

    Args:
        atoms: list of Atom
        ff: initialized AtomicForcefield

    Returns:
        D      : np.ndarray (N, N)
        E_elec : np.ndarray (N, N)
        E_vdw  : np.ndarray (N, N)
    """
    # distance matrix (diagonal set to inf to avoid self-interactions)
    coords = np.stack([a.pos for a in atoms], axis=0)
    D = distance_matrix(coords, coords)
    np.fill_diagonal(D, np.inf)

    # Cache force field lookups - same (res_name, atom_name) pairs appear many times
    charge_cache = {}
    vdw_cache = {}

    def get_charge_cached(res_name, atom_name, all_atom_names):
        key = (res_name, atom_name)
        if key not in charge_cache:
            charge_cache[key] = ff.get_charge(res_name, atom_name, all_atom_names)
        return charge_cache[key]

    def get_vdw_cached(res_name, atom_name, all_atom_names):
        key = (res_name, atom_name)
        if key not in vdw_cache:
            vdw_cache[key] = ff.get_vanderwaals(res_name, atom_name, all_atom_names)
        return vdw_cache[key]

    # electrostatics: q_i q_j / (epsilon * r_ij)
    charges = [get_charge_cached(a.res_name, a.atom_name, a.all_atom_names) for a in atoms]
    q = np.array(charges, dtype=float)
    E_elec = np.outer(q, q) * COULOMB_CONSTANT / (EPSILON0 * D)

    # LJ main terms via Lorentz-Berthelot mixing
    vdw_params = [get_vdw_cached(a.res_name, a.atom_name, a.all_atom_names) for a in atoms]
    sig_main = np.array([v.sigma_main for v in vdw_params], dtype=float)
    eps_main = np.array([v.epsilon_main for v in vdw_params], dtype=float)
    mean_sig = 0.5 * (sig_main[:, None] + sig_main[None, :])          # arithmetic mean σ
    geom_eps = np.sqrt(eps_main[:, None] * eps_main[None, :])          # geometric mean ε
    r6 = (mean_sig / D) ** 6
    E_vdw = 4 * geom_eps * (r6**2 - r6)

    # LJ 1–4 parameters (reuse cached vdw_params)
    sig_14 = np.array([v.sigma_14 for v in vdw_params], dtype=float)
    eps_14 = np.array([v.epsilon_14 for v in vdw_params], dtype=float)
    mean_sig14 = 0.5 * (sig_14[:, None] + sig_14[None, :])
    geom_eps14 = np.sqrt(eps_14[:, None] * eps_14[None, :])
    r614 = (mean_sig14 / D) ** 6
    E_vdw14 = 4 * geom_eps14 * (r614**2 - r614)

    # same-chain 1–4 and 1–3 handling (simple distance-based masks)
    chain_ids = [a.chain for a in atoms]
    same_chain = np.equal.outer(chain_ids, chain_ids)
    idx14 = (D < CUTOFF_14) & same_chain
    idx13 = (D < CUTOFF_13) & same_chain

    # apply 1–4 LJ; zero out 1–3 terms for both LJ and Coulomb
    E_vdw[idx14] = E_vdw14[idx14]
    E_vdw[idx13] = 0.0
    E_elec[idx13] = 0.0

    return D, E_elec, E_vdw


def add_residue_contacts(graph, db):
    """
    Add residue–residue edge features to a NetworkX graph:
      - dist     : minimum interatomic distance
      - elec     : sum of electrostatic energies
      - vdw      : sum of van der Waals energies
      - covalent : 1 if dist < COVALENT_CUTOFF, else 0

    Args:
        graph: object exposing graph.nx (NetworkX) with nodes (chainID, resSeq, resName)
        db   : pdb2sql interface to fetch atoms
    """
    # force field
    ff = AtomicForcefield(FF_DIR)

    # collect per-atom data for each residue node
    all_atoms: List[Atom] = []
    res_to_idxs = {}
    for node in graph.nx.nodes:
        chainID, resSeq, resName = node
        raw = db.get("name,x,y,z", chainID=chainID, resSeq=int(resSeq))

        # decode atom names once per residue for charge/LJ lookup
        atom_names = [
            r[0].decode() if isinstance(r[0], (bytes, bytearray)) else r[0]
            for r in raw
        ]

        idxs = []
        for name, x, y, z in raw:
            # minimal atom record
            atom_name = name.decode() if isinstance(name, (bytes, bytearray)) else name
            pos = np.array([x, y, z], dtype=float)
            all_atoms.append(Atom(chainID, resName, atom_name, atom_names, pos))
            idxs.append(len(all_atoms) - 1)

        res_to_idxs[node] = idxs

    # nonbonded matrices once for all atoms
    D, E_elec, E_vdw = compute_nonbonded(all_atoms, ff)

    # aggregate to residue–residue edges
    for u, v in graph.nx.edges:
        idx1, idx2 = res_to_idxs[u], res_to_idxs[v]
        d_min = float(D[np.ix_(idx1, idx2)].min())
        e_sum = float(E_elec[np.ix_(idx1, idx2)].sum())
        v_sum = float(E_vdw[np.ix_(idx1, idx2)].sum())
        cov = float(d_min < COVALENT_CUTOFF)

        graph.nx.edges[u, v].update({
            "dist": d_min,
            "elec": e_sum,
            "vdw": v_sum,
            "covalent": cov,
        })



# end of contacts.py


def add_residue_contacts_atomgraph(graph, db):
    """
    Add residue–residue edge features to a NetworkX graph:
      - dist     : minimum interatomic distance
      - elec     : sum of electrostatic energies
      - vdw      : sum of van der Waals energies
      - covalent : 1 if dist < COVALENT_CUTOFF, else 0

    Args:
        graph: object exposing graph.nx (NetworkX) with nodes (chainID, resSeq, resName)
        db   : pdb2sql interface to fetch atoms
    """
    # force field
    ff = AtomicForcefield(FF_DIR)

    # collect per-atom data for each residue node
    all_atoms: List[Atom] = []
    # Build lookup dict for O(1) access instead of O(N) linear search per edge
    atom_to_idx: dict = {}

    for node in graph.nx.nodes:
        chainID, resSeq, resName, atomid, atomname = node
        raw = db.get("name,x,y,z", chainID=chainID, resSeq=int(resSeq))

        # decode atom names once per residue for charge/LJ lookup
        atom_names = [
            r[0].decode() if isinstance(r[0], (bytes, bytearray)) else r[0]
            for r in raw
        ]

        for name, x, y, z in raw:
            # minimal atom record
            atom_name = name.decode() if isinstance(name, (bytes, bytearray)) else name
            pos = np.array([x, y, z], dtype=float)
            idx = len(all_atoms)
            all_atoms.append(Atom(chainID, resName, resSeq, atom_name, atom_names, pos))
            # Key: (chain, resName, resSeq, atomName) for O(1) lookup
            atom_to_idx[(chainID, resName, resSeq, atom_name)] = idx

    # nonbonded matrices once for all atoms
    D, E_elec, E_vdw = compute_nonbonded(all_atoms, ff)

    # Process edges with O(1) lookup instead of O(N) linear search
    for u, v in graph.nx.edges:
        # node format: (chainID, resSeq, resName, atomID, atomName)
        # lookup key:  (chain, resName, resSeq, atomName)
        key1 = (u[0], u[2], u[1], u[4])
        key2 = (v[0], v[2], v[1], v[4])

        idx1 = atom_to_idx.get(key1)
        idx2 = atom_to_idx.get(key2)

        if idx1 is None or idx2 is None or idx1 == idx2:
            continue

        d = float(D[idx1, idx2])
        e_sum = float(E_elec[idx1, idx2])
        v_sum = float(E_vdw[idx1, idx2])
        cov = float(d < COVALENT_CUTOFF)

        graph.nx.edges[u, v]["dist"] = d
        graph.nx.edges[u, v]["elec"] = e_sum
        graph.nx.edges[u, v]["vdw"] = v_sum
        graph.nx.edges[u, v]["covalent"] = cov


