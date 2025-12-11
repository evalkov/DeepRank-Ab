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



class Atom(NamedTuple):
    """Minimal atom record used for contact calculations."""
    chain: str
    res_name: str              # e.g., "ARG"
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

    # electrostatics: q_i q_j / (epsilon * r_ij)
    charges = [ff.get_charge(a.res_name, a.atom_name, a.all_atom_names) for a in atoms]
    q = np.array(charges, dtype=float)
    E_elec = np.outer(q, q) * COULOMB_CONSTANT / (EPSILON0 * D)

    # LJ main terms via Lorentz-Berthelot mixing
    sig_main = np.array([
        ff.get_vanderwaals(a.res_name, a.atom_name, a.all_atom_names).sigma_main
        for a in atoms
    ], dtype=float)
    eps_main = np.array([
        ff.get_vanderwaals(a.res_name, a.atom_name, a.all_atom_names).epsilon_main
        for a in atoms
    ], dtype=float)
    mean_sig = 0.5 * (sig_main[:, None] + sig_main[None, :])          # arithmetic mean σ
    geom_eps = np.sqrt(eps_main[:, None] * eps_main[None, :])          # geometric mean ε
    r6 = (mean_sig / D) ** 6
    E_vdw = 4 * geom_eps * (r6**2 - r6)

    # LJ 1–4 parameters
    sig_14 = np.array([
        ff.get_vanderwaals(a.res_name, a.atom_name, a.all_atom_names).sigma_14
        for a in atoms
    ], dtype=float)
    eps_14 = np.array([
        ff.get_vanderwaals(a.res_name, a.atom_name, a.all_atom_names).epsilon_14
        for a in atoms
    ], dtype=float)
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
    ff = AtomicForcefield(FF_DIR)

    all_atoms: List[Atom] = []
    res_to_idxs = {}

    # extract atoms
    for node in graph.nx.nodes:
        chainID, resSeq, resName = node
        raw = db.get("name,x,y,z", chainID=chainID, resSeq=int(resSeq))

        atom_names = [
            r[0].decode() if isinstance(r[0], (bytes, bytearray)) else r[0]
            for r in raw
        ]

        idxs = []
        for name, x, y, z in raw:
            atom_name = name.decode() if isinstance(name, (bytes, bytearray)) else name
            pos = np.array([x, y, z], dtype=float)
            all_atoms.append(Atom(chainID, resName, atom_name, atom_names, pos))
            idxs.append(len(all_atoms) - 1)

        res_to_idxs[node] = idxs

    # compute all-vs-all atom interactions
    D, E_elec, E_vdw = compute_nonbonded(all_atoms, ff)

    # fix edge-ordering
    edges = list(graph.nx.edges)

    dists = []
    elevs = []
    vdws = []
    covalents = []

    # compute features
    for u, v in edges:
        idx1, idx2 = res_to_idxs[u], res_to_idxs[v]
        d_min = float(D[np.ix_(idx1, idx2)].min())
        e_sum = float(E_elec[np.ix_(idx1, idx2)].sum())
        v_sum = float(E_vdw[np.ix_(idx1, idx2)].sum())
        cov = float(d_min < COVALENT_CUTOFF)

        dists.append(d_min)
        elevs.append(e_sum)
        vdws.append(v_sum)
        covalents.append(cov)

    # assign features
    for i, (u, v) in enumerate(edges):
        graph.nx[u][v]['dist'] = dists[i]
        graph.nx[u][v]['elec'] = elevs[i]
        graph.nx[u][v]['vdw'] = vdws[i]
        graph.nx[u][v]['covalent'] = covalents[i]
