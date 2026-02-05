# Pipeline Optimization TODO

## Medium Priority

### 1. Parallel Graph Prep
**File**: `scripts/split_stageA_cpu.py` (lines 352-376)

**Current**: Sequential PDB merging + FASTA writing in `build_merged_structure()` loop

**Proposed**: Use `ProcessPoolExecutor` to parallelize across cores

```python
from concurrent.futures import ProcessPoolExecutor

def prep_one_pdb(args):
    pdb_path, heavy, light, antigen, merged_dir, fasta_dir = args
    seqH, seqL, seqAg = build_merged_structure(...)
    # Write FASTA
    return seqH, seqL, seqAg, merged_pdb

with ProcessPoolExecutor(max_workers=num_cores) as ex:
    results = ex.map(prep_one_pdb, task_args)
```

**Expected gain**: 5-10%

---

### 2. Dynamic ESM Load Balancing
**File**: `scripts/split_stageB_gpu.py` (lines 385-455)

**Current**: Fixed load balancing at shard creation time, blocking wait on all GPUs

**Proposed**: Monitor GPU completion and redistribute remaining sequences to idle GPUs

**Expected gain**: 5-20% (depends on hardware heterogeneity)

---

### 3. Batch Annotation by Model
**File**: `src/tools/annotate.py` (lines 393-486)

**Current**: All records batched into single ANARCI call

**Proposed**: Micro-batch by model to improve HMMER cache locality

**Expected gain**: 5%

---

### 4. DataLoader Tuning
**File**: `scripts/split_stageB_gpu.py`

**Current**: `num_workers=8`, `prefetch_factor=2` (PyTorch default)

**Proposed**:
- Increase `prefetch_factor` to 4 for better GPU utilization
- Auto-tune `batch_size` based on GPU memory

**Expected gain**: 5-15%

---

## Low Priority / Advanced

### 5. Async I/O for Network Filesystems
**Files**: Multiple

**Proposed**: Use `aiofiles` for async I/O during staging operations

**Expected gain**: 20-50% (network-bound operations only)

---

### 6. Caching Layer for Expensive Computations
**Proposed**: Track PDB -> graph mappings across runs with checksum-based cache

**Expected gain**: 50-80% (when reprocessing same PDBs)

---

### 7. Distributed Multi-Node Inference
**File**: `scripts/split_stageB_gpu.py`

**Current**: Single-GPU inference per task

**Proposed**: Use Ray or Dask for distributed inference across nodes

**Expected gain**: 2-8x (multi-node scaling)

**Effort**: High (architecture change)

---

## Summary Table

| Priority | Improvement | Effort | Expected Gain |
|----------|-------------|--------|---------------|
| Medium | Parallel graph prep | Low | 5-10% |
| Medium | Dynamic ESM load balancing | Medium | 5-20% |
| Medium | Batch annotation by model | Low | 5% |
| Medium | DataLoader tuning | Low | 5-15% |
| Low | Async I/O | Medium | 20-50% |
| Low | Caching layer | High | 50-80% |
| Low | Distributed inference | High | 2-8x |
