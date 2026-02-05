# Pipeline Optimization TODO

## Completed

### 1. Parallel Graph Prep - DONE
See `docs/improvements.md` for details.

### 4. DataLoader Tuning - DONE
See `docs/improvements.md` for details.

### 5. Per-subprocess OpenMP Control (VORO_OMP_THREADS) - DONE
See `docs/improvements.md` for details.

### 6. Dynamic Resource Allocation - DONE
Pipeline launcher now analyzes input PDBs and automatically sizes SLURM arrays:
- `--analyze` flag to preview resource estimates
- Two-phase Stage A (sharding then processing)
- Dynamic array sizing based on estimated/actual shard count
- Configurable concurrency limits (`max_concurrent_a`, `max_concurrent_b`)
See `docs/howto.md` for usage.

---

## Medium Priority

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

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|-------------|--------|---------------|--------|
| ~~Medium~~ | ~~Parallel graph prep~~ | ~~Low~~ | ~~5-10%~~ | **DONE** |
| Medium | Dynamic ESM load balancing | Medium | 5-20% | Pending |
| Medium | Batch annotation by model | Low | 5% | Pending |
| ~~Medium~~ | ~~DataLoader tuning~~ | ~~Low~~ | ~~5-15%~~ | **DONE** |
| ~~Medium~~ | ~~VORO_OMP_THREADS~~ | ~~Low~~ | ~~Variable~~ | **DONE** |
| ~~High~~ | ~~Dynamic resource allocation~~ | ~~Medium~~ | ~~Usability~~ | **DONE** |
| Low | Async I/O | Medium | 20-50% | Pending |
| Low | Caching layer | High | 50-80% | Pending |
| Low | Distributed inference | High | 2-8x | Pending |
