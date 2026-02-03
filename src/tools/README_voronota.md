# Voronota-LT Integration

This document describes the migration from legacy `voronota` and `freesasa` to the unified `voronota-lt` implementation for computing contact areas and buried surface areas (BSA).

## Overview

The original implementation used:
- **voronota** (two-step pipeline) for atom-atom contact areas
- **freesasa** (Python library) for solvent-accessible surface / BSA

The new implementation uses:
- **voronota-lt** for both contact areas and BSA computation

### Benefits

| Metric | Before | After |
|--------|--------|-------|
| Subprocess calls per structure | 4 | 3 |
| Multi-threading | None | OpenMP (`--processors`) |
| Chain BSA calculations | Sequential | Parallel |
| Dependencies | voronota + freesasa | voronota-lt only |

---

## File Changes

### 1. `tools/VoroContacts.py` (NEW)

Unified voronota-lt wrapper that combines contact area and BSA computation.

**Key features:**
- Single voronota-lt call for contacts + complex SAS (using file output)
- Parallel chain SAS calculations via `ThreadPoolExecutor`
- Configurable `--processors` for OpenMP multi-threading
- Drop-in replacement classes for backwards compatibility

**Classes:**
- `VoroContacts` - Main unified class
- `VoronotaAreas` - Compatibility wrapper for VoroArea
- `BSA` - Compatibility wrapper for BSA

**Usage:**
```python
from tools.VoroContacts import VoroContacts

# Compute both contacts and BSA in optimized way
vc = VoroContacts(
    pdb_path,
    probe=1.4,
    processors=4,
    compute_contacts=True,
    compute_bsa=True,
)

# Access results
contact_areas = vc.contact_areas  # For edge features
complex_sas = vc.complex_sas      # Residue-level SAS
chain_sas = vc.chain_sas          # Per-chain SAS for BSA calculation
```

---

### 2. `tools/VoroArea.py` (MODIFIED)

Now re-exports optimized `VoronotaAreas` from `VoroContacts`.

**Before:**
```python
class VoronotaAreas:
    def __init__(self, pdb_path, probe=1.4, processors=1):
        # Single voronota-lt call, but processors=1 default
```

**After:**
```python
# Re-export from optimized module
from tools.VoroContacts import VoronotaAreas

# Legacy class kept as VoronotaAreasLegacy for fallback
```

**Key changes:**
- Default `processors=4` (was 1)
- Combined with BSA complex calculation when both are used

---

### 3. `tools/BSA.py` (MODIFIED)

Now re-exports optimized `BSA` from `VoroContacts`.

**Before (freesasa):**
```python
class BSA:
    def get_structure(self):
        # 3x freesasa.calc() calls
        self.result_complex = freesasa.calc(self.complex)
        self.result_chains['A'] = freesasa.calc(self.chains['A'])
        self.result_chains['B'] = freesasa.calc(self.chains['B'])

    def get_contact_residue_sasa(self):
        # 2n freesasa.selectArea() calls in loop
        for r in residues:
            freesasa.selectArea(..., self.complex, ...)
            freesasa.selectArea(..., self.chains[r[0]], ...)
```

**After (voronota-lt):**
```python
class BSA:
    def get_structure(self):
        # 1x voronota-lt for complex (combined with contacts if VoroArea used)
        # 2x voronota-lt for chains (parallel via ThreadPoolExecutor)

    def get_contact_residue_sasa(self):
        # Pure dict lookups, no subprocess calls
        for r in residues:
            bsa = chain_sas[key] - complex_sas[key]
```

**Key changes:**
- Replaced freesasa with voronota-lt
- Added `--processors` for multi-threading
- Parallel chain calculations
- Eliminated per-residue subprocess calls

---

### 4. `AtomGraph.py` (MODIFIED)

Updated `_get_voro_contact_area()` to use new key format.

**Before:**
```python
# Key format: (atomID, chainID, resSeq, resName, atomName)
at1k = (str(at1[3]), str(at1[0]), str(at1[1]), str(at1[2]), str(at1[4]))
```

**After:**
```python
# Key format: (chainID, resSeq, resName, atomName)
at1k = (str(node1[0]), str(node1[1]), str(node1[2]), str(node1[4]))
```

**Reason:** voronota-lt outputs chain/resSeq/resName/atomName directly, not PDB serial numbers.

---

## Build Configuration

### `expansion_lt/build_intel.sh` (MODIFIED)

Changed default architecture from `-xHost` (native) to `-xCORE-AVX2` (portable).

**Before:**
```bash
CXX_FLAGS="-std=c++14 -O3 -xHost -qopenmp -qopenmp-link=static"
```

**After:**
```bash
ARCH_TARGET="${3:-portable}"  # New parameter

case "${ARCH_TARGET}" in
    portable|avx2) ARCH_FLAG="-xCORE-AVX2" ;;   # Default - works on Haswell+
    avx512)        ARCH_FLAG="-xCORE-AVX512" ;; # Requires Skylake-X+
    native)        ARCH_FLAG="-xHost" ;;         # Current CPU only
esac

CXX_FLAGS="-std=c++14 -O3 ${ARCH_FLAG} -qopenmp -qopenmp-link=static"
```

**Usage:**
```bash
./build_intel.sh                          # Portable (AVX2) - recommended
./build_intel.sh 2023.1.0 direct avx512   # AVX512 (faster but less compatible)
./build_intel.sh 2023.1.0 direct native   # Current CPU only (not portable)
```

---

## Performance Tuning

### Processor Count

Adjust `processors` parameter based on your system:

```python
# In AtomGraph.py or ResidueGraph.py
self.voro = VoronotaAreas(self.pdb, processors=8)
bsa_calc = BSA.BSA(self.pdb, db, processors=8)
```

Or modify defaults in `VoroContacts.py`:
```python
def __init__(self, ..., processors=8, ...):  # Change default
```

### Benchmarking

To compare old vs new implementation:

```python
from tools.VoroArea import VoronotaAreas, VoronotaAreasLegacy
from tools.BSA import BSA, BSA_Freesasa
import time

# New (voronota-lt)
t0 = time.time()
voro = VoronotaAreas(pdb_path)
print(f"voronota-lt contacts: {time.time() - t0:.2f}s")

# Legacy (two-step voronota)
t0 = time.time()
voro_legacy = VoronotaAreasLegacy(pdb_path)
print(f"voronota legacy: {time.time() - t0:.2f}s")
```

---

## Binary Requirements

Place the voronota-lt binary at:
```
src/tools/voronota/voronota-lt
```

Build with portable flags:
```bash
cd /path/to/voronota/expansion_lt
./build_intel.sh 2023.1.0 direct portable

# Or with GCC (no Intel compiler needed)
g++ -std=c++14 -O3 -fopenmp -o voronota-lt src/voronota_lt.cpp

# Copy to DeepRank-Ab
cp voronota-lt /path/to/DeepRank-Ab/src/tools/voronota/
```

---

## Troubleshooting

### "AVX512 instructions not supported"

The binary was compiled with `-xHost` on a machine with AVX512. Rebuild with portable flags:
```bash
./build_intel.sh 2023.1.0 direct portable
```

### "voronota-lt not found"

Ensure binary exists at `src/tools/voronota/voronota-lt` and is executable:
```bash
chmod +x src/tools/voronota/voronota-lt
```

### Different results from legacy implementation

voronota-lt uses **radical tessellation** by default (faster), while legacy voronota uses **additively weighted** tessellation. Results are highly correlated (r > 0.98) but not identical.

For exact compatibility, add `--run-in-aw-diagram-regime` flag (slower):
```python
# In VoroContacts.py _run_combined():
cmd.append('--run-in-aw-diagram-regime')
```
