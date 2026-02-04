import os
import numpy as np
import shutil
import torch
from time import time
import networkx as nx
from pdb2sql import interface

from tools import BioWrappers, BSA
from Graph import Graph
from tools.edge_orientation import compute_edge_orientation
from tools.VoroArea import VoronotaAreas 



class ResidueGraph(Graph):
    """
    Build a residue-level interface graph from a PDB and decorate it with features.
    Nodes are residues; edges include interface and intra-chain contacts (both directions).
    Optional extras:region labels, edge orientations, contact features.
    """
    def __init__(
        self,
        pdb=None,
        contact_distance=5,
        internal_contact_distance=3,
        biopython=False,
        region_map=None,
        add_orientation=False,
        contact_features: bool = False,
        antigen_chainid="B",
        use_regions=True,
        use_voro: bool = False,
    ):
        super().__init__()
        # basic ids
        self.type = "residue"
        self.pdb = pdb
        self.name = os.path.splitext(os.path.basename(pdb))[0]

        # region mapping (disable if empty/None)
        self.region_map = region_map or {}
        self.antigen_chainid = antigen_chainid
        self.use_voro = use_voro 

        # region labels and indexing
        self.region_labels = [
            "FR", "L1", "L2", "L3",
            "H1", "H2", "H3",
            "CONST", "AG",
        ]
        self.region2idx = {lab: i for i, lab in enumerate(self.region_labels)}
        if self.use_voro:
            self.voro = VoronotaAreas(self.pdb)


        self.add_orientation = add_orientation
        self.contact_distance = contact_distance
        self.internal_contact_distance = internal_contact_distance
        self.biopython = biopython
        self.contact_features = contact_features
        self.use_regions = use_regions 

        # residue charge (scaled)
        self.residue_charge = {
            "CYS": 2.5,  "HIS": -3.20, "ASN": -3.50, "GLN": -3.50, "SER": -0.80,
            "THR": -0.70, "TYR": -1.30, "TRP": -0.90, "ALA": 1.8,  "PHE": 2.8,
            "GLY": -0.4, "ILE": 4.50, "VAL": 4.20, "MET": 1.9,   "PRO": -1.60,
            "LEU": 3.80, "GLU": -3.50, "ASP": -3.50, "LYS": -3.90, "ARG": -4.50,
        }

        # residue index mapping
        self.residue_names = {
            "CYS": 0,  "HIS": 1,  "ASN": 2,  "GLN": 3,  "SER": 4,
            "THR": 5,  "TYR": 6,  "TRP": 7,  "ALA": 8,  "PHE": 9,
            "GLY": 10, "ILE": 11, "VAL": 12, "MET": 13, "PRO": 14,
            "LEU": 15, "GLU": 16, "ASP": 17, "LYS": 18, "ARG": 19,
        }

        # residue polarity category
        self.residue_polarity = {
            "CYS": "polar", "HIS": "polar", "ASN": "polar", "GLN": "polar",
            "SER": "polar", "THR": "polar", "TYR": "polar", "TRP": "polar",
            "ALA": "apolar", "PHE": "apolar", "GLY": "apolar", "ILE": "apolar",
            "VAL": "apolar", "MET": "apolar", "PRO": "apolar", "LEU": "apolar",
            "GLU": "neg_charged", "ASP": "neg_charged",
            "LYS": "pos_charged", "ARG": "pos_charged",
        }



        self.polarity_encoding = {
            "apolar": 0, "polar": 1, "neg_charged": 2, "pos_charged": 3,
        }

        # create database handle
        db = interface(self.pdb)

        # build graph
        t0 = time()
        self.get_graph(db)

        # node features
        t0 = time()
        self.get_node_features(db)

        # edge features
        t0 = time()
        self.get_edge_features(db)

        # close db
        db._close()

    def check_execs(self):
        """
        Check presence of external executables needed by optional features.
        """
        execs = {"msms": "http://mgltools.scripps.edu/downloads#msms"}
        for e, inst in execs.items():
            if shutil.which(e) is None:
                print(e, "is not installed. see", inst, "for details")

    def get_graph(self, db):
        self.nx = nx.DiGraph()

        self.res_contact_pairs = db.get_contact_residues(
            cutoff=self.contact_distance, return_contact_pairs=True
        )
        all_nodes = self._get_all_valid_nodes(self.res_contact_pairs)

        for key, neighbors in self.res_contact_pairs.items():
            if key not in all_nodes:
                continue
            for v in neighbors:
                if v not in all_nodes:
                    continue
                d = self._get_edge_distance(key, v, db)
                self.nx.add_edge(key, v, dist=d, type=bytes("interface", encoding="utf-8"))
                self.nx.add_edge(v, key, dist=d, type=bytes("interface", encoding="utf-8"))

        edges, dist_list = self.get_internal_edges(db)
        for (u, v), d in zip(edges, dist_list):
            if u in all_nodes and v in all_nodes:
                self.nx.add_edge(u, v, dist=d, type=bytes("internal", encoding="utf-8"))
                self.nx.add_edge(v, u, dist=d, type=bytes("internal", encoding="utf-8"))


    @staticmethod
    def _get_all_valid_nodes(res_contact_pairs, verbose=False):
        valid_res = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "ASX", "SEC", "GLX",
        ]

        keys_to_pop = []
        for res in res_contact_pairs.keys():
            if res[2] not in valid_res:
                keys_to_pop.append(res)
                Warning("--> Residue ", res, " not valid")

        for res in keys_to_pop:
            if res in res_contact_pairs:
                res_contact_pairs.pop(res, None)

        nodesB = []
        for _, reslist in list(res_contact_pairs.items()):
            for res in reslist:
                if res[2] in valid_res:
                    nodesB.append(res)
                else:
                    if verbose:
                        print("removed node", res)
        nodesB = sorted(set(nodesB))

        return list(res_contact_pairs.keys()) + nodesB

    def get_node_features(self, db):
        """
        Compute per-node features: chain id, mean 3D position, type, charge,
        polarity, BSA, optional region label, and optional BioPython features.
        """
        t0 = time()
        bsa_calc = BSA.BSA(self.pdb, db)
        bsa_calc.get_structure()
        bsa_calc.get_contact_residue_sasa(cutoff=self.contact_distance)
        bsa_data = bsa_calc.bsa_data

        model = BioWrappers.get_bio_model(db.pdbfile)
        if self.biopython:
            ResDepth = BioWrappers.get_depth_contact_res(model, self.nx.nodes)
            HSE = BioWrappers.get_hse(model)

        for node_key in self.nx.nodes:
            chainID = node_key[0]
            resName = node_key[2]
            resSeq = int(node_key[1])

            self.nx.nodes[node_key]["chain"] = {"A": 0, "B": 1}[chainID]
            self.nx.nodes[node_key]["pos"] = np.mean(
                db.get("x,y,z", chainID=chainID, resSeq=resSeq), axis=0
            )
            self.nx.nodes[node_key]["type"] = self.onehot(
                self.residue_names[resName], len(self.residue_names)
            )
            self.nx.nodes[node_key]["charge"] = self.residue_charge[resName]
            self.nx.nodes[node_key]["polarity"] = self.onehot(
                self.polarity_encoding[self.residue_polarity[resName]],
                len(self.polarity_encoding),
            )
            self.nx.nodes[node_key]["bsa"] = bsa_data[node_key]


            # region feature: CDRs on chain A from mapping, AG for chain B
            if self.use_regions:
                if node_key[0] == self.antigen_chainid:
                    ag_idx = self.region2idx["AG"]
                    self.nx.nodes[node_key]["region"] = self.onehot(
                        ag_idx, len(self.region_labels)
                    )
                else:
                    key = (f'{self.name}', resSeq)
                    if key not in self.region_map:
                        raise KeyError(f"missing region mapping for {key}")
                    region_label = self.region_map[key]
                    try:
                        idx = self.region2idx[region_label]
                    except KeyError:
                        raise KeyError(f"unknown region label '{region_label}' for {key}")
                    self.nx.nodes[node_key]["region"] = self.onehot(
                        idx, len(self.region_labels)
                    )

            if self.biopython:
                self.nx.nodes[node_key]["depth"] = (
                    ResDepth[node_key] if node_key in ResDepth else 0
                )
                bio_key = (chainID, resSeq)
                self.nx.nodes[node_key]["hse"] = (
                    HSE[bio_key] if bio_key in HSE else (0, 0, 0)
                )

    def get_edge_features(self, db):
        """
        Add edge-level features:
        - directed edge_index
        - rij (vector from source to target)
        - optional local-frame orientation per edge
        - optional residue contact features
        """
        # edge index (directed) - use dict for O(1) lookup instead of O(n) list.index()
        node_keys = list(self.nx.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_keys)}
        directed_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in self.nx.edges]
        self.nx.edge_index = directed_idx

        # rij = pos[j] - pos[i]
        # pos = np.array([self.nx.nodes[n]["pos"] for n in node_keys])
        # ei = np.array(directed_idx)
        # rij = pos[ei[:, 1]] - pos[ei[:, 0]]
        # for idx, (u, v) in enumerate(self.nx.edges):
        #     self.nx.edges[u, v]["rij"] = rij[idx]
        for idx, (u, v) in enumerate(self.nx.edges):
            # self.nx.edges[u, v]["rij"] = rij[idx][0]
            if self.use_voro:
                area = self._get_voro_contact_area(u, v)
                self.nx.edges[u, v]["voro_area"] = area

        # orientation (optional)
        if self.add_orientation:
            # Batch fetch all backbone atoms once instead of 3 queries per residue
            backbone_coords = {}
            for name in ("CA", "C", "N"):
                for row in db.get("chainID,resSeq,x,y,z", name=name):
                    chain, resseq, x, y, z = row
                    backbone_coords[(chain, int(resseq), name)] = (x, y, z)

            pos_CA, pos_C, pos_N = [], [], []
            for (chain, resseq, _), _ in self.nx.nodes(data=True):
                resseq_int = int(resseq)
                pos_CA.append(backbone_coords.get((chain, resseq_int, "CA"), (0, 0, 0)))
                pos_C.append(backbone_coords.get((chain, resseq_int, "C"), (0, 0, 0)))
                pos_N.append(backbone_coords.get((chain, resseq_int, "N"), (0, 0, 0)))
            pos_CA = torch.tensor(pos_CA, dtype=torch.float)
            pos_C = torch.tensor(pos_C, dtype=torch.float)
            pos_N = torch.tensor(pos_N, dtype=torch.float)
            ei_t = torch.tensor(directed_idx, dtype=torch.long).t()

            orient = compute_edge_orientation(pos_CA, pos_C, pos_N, ei_t).cpu().numpy()
            for idx, (u, v) in enumerate(self.nx.edges):
                self.nx.edges[u, v]["orientation"] = orient[idx]

        # optional residue-level contact features
        if self.contact_features:
            from tools.contacts_dr2_res import add_residue_contacts
            add_residue_contacts(self, db)

    def get_internal_edges(self, db):
        """
        Collect intra-chain edges for both chains using the atom-level cutoff.
        Returns edge tuples and their minimal atom-atom distances.
        """
        nodesA, nodesB = [], []
        for n in self.nx.nodes:
            if n[0] == "A":
                nodesA.append(n)
            elif n[0] == "B":
                nodesB.append(n)

        edgesA, distA = self._get_internal_edges_chain(nodesA, db, self.internal_contact_distance)
        edgesB, distB = self._get_internal_edges_chain(nodesB, db, self.internal_contact_distance)
        return edgesA + edgesB, distA + distB

    def _get_internal_edges_chain(self, nodes, db, cutoff):
        """
        Find intra-chain edges and minimal atom-atom distance between residue pairs.
        """
        nn = len(nodes)
        if nn == 0:
            return [], []

        # Pre-fetch all coordinates for all nodes in this chain (O(n) queries instead of O(n²))
        node_coords = {}
        for node in nodes:
            xyz = np.array(db.get("x,y,z", chainID=node[0], resSeq=node[1]))
            node_coords[node] = xyz

        edges, dist = [], []
        cutoff_sq = cutoff ** 2
        for i1 in range(nn):
            xyz1 = node_coords[nodes[i1]]
            xyz1_sq = np.sum(xyz1**2, axis=1)[:, None]
            for i2 in range(i1 + 1, nn):
                xyz2 = node_coords[nodes[i2]]
                d2 = (
                    -2 * np.dot(xyz1, xyz2.T)
                    + xyz1_sq
                    + np.sum(xyz2**2, axis=1)
                )
                if np.any(d2 < cutoff_sq):
                    edges.append((nodes[i1], nodes[i2]))
                    dist.append(np.sqrt(np.min(d2)))
        return edges, dist

    def _get_edge_distance(self, node1, node2, db):
        """
        Minimal atom-atom distance between two residues.
        """
        xyz1 = np.array(db.get("x,y,z", chainID=node1[0], resSeq=node1[1]))
        xyz2 = np.array(db.get("x,y,z", chainID=node2[0], resSeq=node2[1]))
        d2 = (
            -2 * np.dot(xyz1, xyz2.T)
            + np.sum(xyz1**2, axis=1)[:, None]
            + np.sum(xyz2**2, axis=1)
        )
        return np.sqrt(np.min(d2))

    def onehot(self, idx, size):
        """One-hot encode an integer index."""
        onehot = torch.zeros(size)
        onehot[idx] = 1
        return np.array(onehot)
    

    def _get_voro_contact_area_residue(self, node1, node2):
        """
        Compute Voronota contact area between two residues.

        node format: (chainID, resSeq, resName)
        """
        try:
            chain1, res1, _ = node1
            chain2, res2, _ = node2

            # Collect all atoms belonging to each residue
            atoms1 = [a for a in self.atom_list
                    if a.chain == chain1 and a.residue == res1]

            atoms2 = [a for a in self.atom_list
                    if a.chain == chain2 and a.residue == res2]

            area = 0.0
            for at1 in atoms1:
                at1k = (str(at1.atom_id),
                        str(at1.chain),
                        str(at1.residue),
                        str(at1.resname),
                        str(at1.atom_name))

                for at2 in atoms2:
                    # skip self–pairs
                    if at1.residue == at2.residue:
                        continue

                    at2k = (str(at2.atom_id),
                            str(at2.chain),
                            str(at2.residue),
                            str(at2.resname),
                            str(at2.atom_name))

                    area += (
                        self.voro.contact_areas.get(at1k, {}).get(at2k, 0.0)
                        or self.voro.contact_areas.get(at2k, {}).get(at1k, 0.0)
                        or 0.0
                    )

            return float(area)

        except Exception:
            return 0.0
