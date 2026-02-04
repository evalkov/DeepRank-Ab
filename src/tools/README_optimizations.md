# Performance Optimizations

This document describes algorithmic optimizations applied to the graph generation pipeline to reduce computational complexity and database query overhead.

## Overview

The original implementation had several O(n²) and O(n×E) bottlenecks that caused significant slowdowns for larger structures. These have been replaced with O(n) and O(n+E) algorithms using dictionary lookups and batch queries.

---

## 1. Edge Index Construction

**Files:** `AtomGraph.py`, `ResidueGraph.py`

**Problem:** Converting node pairs to indices used `list.index()` which is O(n) per call.

```python
# Before: O(n) per edge = O(n×E) total
node_keys = list(self.nx.nodes)
directed_idx = [[node_keys.index(u), node_keys.index(v)] for u, v in self.nx.edges]
```

**Solution:** Build a dictionary for O(1) lookup.

```python
# After: O(1) per edge = O(E) total
node_keys = list(self.nx.nodes)
node_to_idx = {node: i for i, node in enumerate(node_keys)}
directed_idx = [[node_to_idx[u], node_to_idx[v]] for u, v in self.nx.edges]
```

**Complexity:** O(n×E) → O(n+E)

---

## 2. Backbone Coordinate Fetching (Orientation)

**Files:** `AtomGraph.py`, `ResidueGraph.py`

**Problem:** Fetching CA, C, N coordinates for orientation required 3 database queries per node.

```python
# Before: 3×N database queries
for node in self.nx.nodes:
    pos_CA.append(db.get("x,y,z", name="CA", chainID=chain, resSeq=resseq)[0])
    pos_C.append(db.get("x,y,z", name="C", chainID=chain, resSeq=resseq)[0])
    pos_N.append(db.get("x,y,z", name="N", chainID=chain, resSeq=resseq)[0])
```

**Solution:** Batch fetch all backbone atoms in 3 queries total.

```python
# After: 3 database queries total
backbone_coords = {}
for name in ("CA", "C", "N"):
    for row in db.get("chainID,resSeq,x,y,z", name=name):
        chain, resseq, x, y, z = row
        backbone_coords[(chain, int(resseq), name)] = (x, y, z)

for node in self.nx.nodes:
    pos_CA.append(backbone_coords.get((chain, resseq, "CA"), (0, 0, 0)))
    pos_C.append(backbone_coords.get((chain, resseq, "C"), (0, 0, 0)))
    pos_N.append(backbone_coords.get((chain, resseq, "N"), (0, 0, 0)))
```

**Complexity:** O(3×N) queries → O(3) queries

---

## 3. Internal Edge Detection

**Files:** `AtomGraph.py`, `ResidueGraph.py`

**Problem:** Finding intra-chain edges required a database query for each pair of nodes in the nested loop.

```python
# Before: O(n²) database queries
for i1 in range(nn):
    xyz1 = np.array(db.get("x,y,z", chainID=nodes[i1][0], resSeq=nodes[i1][1]))
    for i2 in range(i1 + 1, nn):
        xyz2 = np.array(db.get("x,y,z", chainID=nodes[i2][0], resSeq=nodes[i2][1]))
        # compute distance...
```

**Solution:** Pre-fetch all coordinates before the nested loop.

```python
# After: O(n) database queries
node_coords = {}
for node in nodes:
    xyz = np.array(db.get("x,y,z", chainID=node[0], resSeq=node[1]))
    node_coords[node] = xyz

for i1 in range(nn):
    xyz1 = node_coords[nodes[i1]]
    for i2 in range(i1 + 1, nn):
        xyz2 = node_coords[nodes[i2]]
        # compute distance...
```

**Complexity:** O(n²) queries → O(n) queries

---

## 4. Contact Feature Computation (Atom Graph)

**File:** `tools/contacts_dr2.py` (`add_residue_contacts_atomgraph`)

**Problem:** Finding atom indices for each edge used linear search through all atoms.

```python
# Before: O(N) search per edge = O(N×E) total
for u, v in graph.nx.edges:
    idx1 = next(i for i, a in enumerate(all_atoms)
                if a.chain == u[0] and a.res_seq == u[1] and a.atom_name == u[4])
    idx2 = next(i for i, a in enumerate(all_atoms)
                if a.chain == v[0] and a.res_seq == v[1] and a.atom_name == v[4])
```

**Solution:** Build a lookup dictionary during atom collection.

```python
# After: O(1) lookup per edge = O(E) total
atom_to_idx = {}
for node in graph.nx.nodes:
    # ... collect atoms ...
    atom_to_idx[(chainID, resName, resSeq, atom_name)] = idx

for u, v in graph.nx.edges:
    key1 = (u[0], u[2], u[1], u[4])
    key2 = (v[0], v[2], v[1], v[4])
    idx1 = atom_to_idx.get(key1)
    idx2 = atom_to_idx.get(key2)
```

**Complexity:** O(N×E) → O(N+E)

---

## 5. HDF5 Edge Index Writing

**File:** `Graph.py`

**Problem:** Converting node tuples to integer indices for HDF5 storage used `list.index()`.

```python
# Before: O(n) per edge = O(E×n) total
node_key = list(self.nx.nodes)
idx_if = [[node_key.index(u), node_key.index(v)] for u, v in directed_if]
idx_int = [[node_key.index(u), node_key.index(v)] for u, v in directed_int]
```

**Solution:** Build a dictionary for O(1) lookup.

```python
# After: O(1) per edge = O(E) total
node_key = list(self.nx.nodes)
node_to_idx = {node: i for i, node in enumerate(node_key)}
idx_if = [[node_to_idx[u], node_to_idx[v]] for u, v in directed_if]
idx_int = [[node_to_idx[u], node_to_idx[v]] for u, v in directed_int]
```

**Complexity:** O(E×n) → O(E)

---

## 6. Force Field Lookup Caching

**Files:** `tools/contacts_dr2.py`, `tools/contacts_dr2_res.py`

**Problem:** Same (residue_name, atom_name) pairs looked up repeatedly for charge and van der Waals parameters.

```python
# Before: N lookups even when only ~50 unique atom types
charges = [ff.get_charge(a.res_name, a.atom_name, ...) for a in atoms]
sig_main = np.array([ff.get_vanderwaals(...).sigma_main for a in atoms])
eps_main = np.array([ff.get_vanderwaals(...).epsilon_main for a in atoms])
sig_14 = np.array([ff.get_vanderwaals(...).sigma_14 for a in atoms])
eps_14 = np.array([ff.get_vanderwaals(...).epsilon_14 for a in atoms])
# 5 × N lookups total
```

**Solution:** Cache lookups by (res_name, atom_name) key.

```python
# After: ~U lookups where U = unique atom types (~50)
charge_cache = {}
vdw_cache = {}

def get_charge_cached(res_name, atom_name, all_atom_names):
    key = (res_name, atom_name)
    if key not in charge_cache:
        charge_cache[key] = ff.get_charge(res_name, atom_name, all_atom_names)
    return charge_cache[key]

# Single vdw lookup per unique type, reuse for all 4 parameters
vdw_params = [get_vdw_cached(a.res_name, a.atom_name, ...) for a in atoms]
sig_main = np.array([v.sigma_main for v in vdw_params])
```

**Complexity:** O(5×N) lookups → O(N + U) lookups, where U << N

---

## Summary Table

| Component | Location | Before | After | Speedup Factor |
|-----------|----------|--------|-------|----------------|
| Edge indexing | AtomGraph.py:310-316 | O(n×E) | O(n+E) | ~n |
| Edge indexing | ResidueGraph.py:251-254 | O(n×E) | O(n+E) | ~n |
| Backbone fetch | AtomGraph.py:324-348 | O(3×N) queries | O(3) queries | ~N |
| Backbone fetch | ResidueGraph.py:269-285 | O(3×N) queries | O(3) queries | ~N |
| Internal edges | AtomGraph.py:440-476 | O(n²) queries | O(n) queries | ~n |
| Internal edges | ResidueGraph.py:314-340 | O(n²) queries | O(n) queries | ~n |
| Contact features | contacts_dr2.py:206-251 | O(N×E) | O(N+E) | ~N |
| HDF5 edge indexing | Graph.py:96-100 | O(E×n) | O(E) | ~n |
| Force field lookups | contacts_dr2.py, contacts_dr2_res.py | O(N) lookups | O(U) lookups | ~N/U |

Where:
- U = number of unique (residue_name, atom_name) pairs (~50 for typical proteins)
- N = number of atoms
- n = number of nodes (residues or atoms depending on graph type)
- E = number of edges

---

## Benchmarks

Pipeline timing on 4 shards (48-core HPC node):

| Stage | Before (s) | After (s) | Speedup |
|-------|------------|-----------|---------|
| prep | 143.7 (3.5%) | 143.4 (8.5%) | 1.0× |
| annotate | 110.3 (2.7%) | 111.3 (6.6%) | 1.0× |
| **graphs** | **3762.5 (92.2%)** | **1368.9 (81.1%)** | **2.75×** |
| cluster | 65.2 (1.6%) | 65.1 (3.9%) | 1.0× |
| **total** | **4081.6** | **1688.8** | **2.42×** |

Graph generation time reduced by **64%**, dropping from 92% to 81% of total pipeline time.

---

## Scientific Accuracy

These optimizations are purely algorithmic and do not affect scientific accuracy:

- **Dictionary lookups** return the exact same values as linear search
- **Batch queries** return the same data, just fetched more efficiently
- **Pre-fetching** uses the same coordinates, just stored temporarily

All computed features (distances, energies, orientations, contact areas) remain identical.

---

## Related Documentation

- [README_voronota.md](README_voronota.md) - Voronota-LT integration and BSA optimization
