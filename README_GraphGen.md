# GraphGen

This document explains the GraphGen implementation in `src/GraphGenMP.py` and the producer-consumer variant in `src/GraphGenMP.py.claude`. Both versions generate identical HDF5 outputs; the Claude variant improves throughput and scalability for large batches.

**Files**
- `src/GraphGenMP.py`: baseline implementation.
- `src/GraphGenMP.py.claude`: producer-consumer implementation.

**Common Behavior**
- Builds residue- or atom-level graphs from a directory of PDB files.
- Optionally uses region annotations and adds orientation and contact features.
- Writes graphs to a single HDF5 file.
- Optionally appends ESM embeddings after graph generation.

**Baseline Pipeline (Pickle Merge)**
1. Workers build graphs in parallel.
2. Each worker pickles its graph to a temporary directory.
3. A single process loads pickles and writes to HDF5.

This works but introduces a serialized merge stage and extra disk I/O.

**Claude Pipeline (Producer-Consumer)**
1. The main process prepares a work list, optionally sorted by PDB size.
2. A worker pool computes graphs and pushes them into a bounded queue.
3. A dedicated writer process drains the queue and writes directly to HDF5.

This removes the pickle intermediate and allows compute and I/O to overlap.

**Why the Claude Version Scales Better**
- Dynamic load balancing via `imap_unordered` and per-item scheduling.
- No serialize-deserialize-serialize loop.
- Continuous writing instead of a single merge phase.
- Bounded queue provides backpressure and controls memory usage.
- Worker initialization happens once, reducing repeated import overhead.

**Behavioral Differences That Can Affect Results**
- `_xray_tidy` filtering:
  - `GraphGenMP.py` mutates the list while iterating, which can skip entries and yield a non-deterministic subset.
  - `GraphGenMP.py.claude` uses a list comprehension, so filtering is deterministic.
- Reference scoring path (`ref_path`):
  - `GraphGenMP.py` uses an undefined `base` when constructing `ref`, which can break reference scoring when `ref_path` is provided.
  - `GraphGenMP.py.claude` defines `base` from the first PDB filename before using it.
- `use_regions=False` robustness:
  - `GraphGenMP.py` does not initialize `self.region_map` unless regions are enabled, which can cause missing-attribute issues in some call paths.
  - `GraphGenMP.py.claude` always initializes `self.region_map` and passes `None` to workers when regions are disabled.

**What Does Not Change (When Inputs Match)**
- Both versions call `ResidueGraph(...)` and `AtomGraph(...)` with the same feature flags.
- Both write graphs through `g.nx2h5(f5)`, so the HDF5 schema is consistent.
- If the same PDBs are processed with the same settings, the graph content matches.

**Output Parity**
The HDF5 outputs are identical between the two versions. This has been verified on a 10,000-PDB test run.

**Quick Usage**
Use either version by importing the desired class and instantiating `GraphHDF5`.

```python
from src.GraphGenMP import GraphHDF5

GraphHDF5(
    pdb_path="/path/to/pdbs",
    graph_type="atom",
    outfile="graphs.hdf5",
    nproc=32,
    use_regions=True,
    region_json="/path/to/annotations.json",
    antigen_chainid="B",
    use_voro=True,
)
```
