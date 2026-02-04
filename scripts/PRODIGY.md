# PRODIGY Batch Processing

High-performance batch processor for PRODIGY binding affinity predictions on HPC clusters.

## Overview

PRODIGY (PROtein binDIng enerGY prediction) predicts binding affinity from protein-protein complex structures. This implementation eliminates subprocess overhead by using PRODIGY as a Python library directly.

**Key optimizations:**
- Library-based execution (no subprocess per PDB)
- Direct filesystem reads (no file copying)
- Batched work units to reduce IPC overhead
- Multi-node parallelism via SLURM

**Expected throughput:** 50-200 structures/second depending on structure size and hardware.

## Quick Start

```bash
# Basic usage
sbatch scripts/prodigy.slurm /path/to/pdbs

# With custom output directory
sbatch scripts/prodigy.slurm /path/to/pdbs /path/to/output

# With custom chain selections
SELECTION_A=H,L SELECTION_B=A sbatch scripts/prodigy.slurm /path/to/pdbs
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SELECTION_A` | `H` | Chain(s) for partner A (comma-separated for multiple) |
| `SELECTION_B` | `T` | Chain(s) for partner B |
| `TEMPERATURE` | `25.0` | Temperature in Celsius for Kd calculation |
| `DISTANCE_CUTOFF` | `5.5` | Distance cutoff for contacts (Angstroms) |
| `ACC_THRESHOLD` | `0.05` | Accessibility threshold for NIS calculation |
| `WORKERS_PER_NODE` | `$SLURM_CPUS_PER_TASK` | Parallel workers per node |
| `BATCH_SIZE` | `100` | PDBs per work unit |
| `WORKER_SCRIPT` | `$SLURM_SUBMIT_DIR/prodigy_batch_worker.py` | Path to worker script |

### SLURM Resources

Default configuration in `prodigy.slurm`:

```bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
```

Adjust based on dataset size:
- ~1M structures: 16 nodes, 4 hours
- ~100K structures: 4 nodes, 1 hour
- ~10K structures: 1 node, 15 minutes

## Output Format

Results are written to a TSV file with the following columns:

| Column | Description |
|--------|-------------|
| `pdb` | Structure identifier (filename stem) |
| `n_chains` | Number of chains in structure |
| `n_residues` | Total residue count |
| `ic_total` | Total interfacial contacts |
| `charged_charged` | Charged-charged contacts |
| `charged_polar` | Charged-polar contacts |
| `charged_apolar` | Charged-apolar contacts |
| `polar_polar` | Polar-polar contacts |
| `apolar_polar` | Apolar-polar contacts |
| `apolar_apolar` | Apolar-apolar contacts |
| `pct_apolar_nis` | % apolar NIS residues |
| `pct_charged_nis` | % charged NIS residues |
| `pct_polar_nis` | % polar NIS residues |
| `dG_kcal_mol` | Predicted binding free energy (kcal/mol) |
| `Kd_M` | Predicted dissociation constant (M) |
| `temperature_C` | Temperature used for Kd calculation |
| `status` | `ok` or `fail` |
| `error` | Error message if failed |

## Chain Selection

### Single Chain vs Chain

```bash
# Heavy chain (H) vs Target (T)
SELECTION_A=H SELECTION_B=T

# Antibody (H+L) vs Antigen (A)
SELECTION_A=H,L SELECTION_B=A
```

### Finding Chain IDs

```bash
# List chains in a PDB
grep "^ATOM" structure.pdb | cut -c22 | sort -u

# Or with BioPython
python -c "
from Bio.PDB import PDBParser
s = PDBParser(QUIET=True).get_structure('x', 'structure.pdb')
print([c.id for c in s[0].get_chains()])
"
```

## Standalone Worker Usage

The worker script can be used independently of SLURM:

```bash
python scripts/prodigy_batch_worker.py \
    --input-dir /path/to/pdbs \
    --output-tsv results.tsv \
    --selection-a H \
    --selection-b T \
    --workers 16 \
    --batch-size 50
```

### Worker Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | (required) | Directory containing PDB/CIF files |
| `--output-tsv` | (required) | Output TSV path |
| `--file-list` | | Optional file with PDB paths (one per line) |
| `--selection-a` | `H` | Partner A chain selection |
| `--selection-b` | `T` | Partner B chain selection |
| `--temperature` | `25.0` | Temperature (Celsius) |
| `--distance-cutoff` | `5.5` | Contact distance cutoff |
| `--acc-threshold` | `0.05` | Accessibility threshold |
| `--workers` | auto | Number of worker processes |
| `--batch-size` | `50` | PDBs per work unit |
| `--start-index` | `0` | Start index in file list |
| `--count` | `0` | Number of files (0 = all) |

## Error Handling

- Individual structure failures don't stop processing
- Failed structures have `status=fail` and error message in output
- Job exits with error if failure rate exceeds 5%
- Common failures: missing chains, malformed PDB, chain selection mismatch

### Debugging Failures

```bash
# Extract failed structures
awk -F'\t' 'NR>1 && $17=="fail" {print $1, $18}' results.tsv

# Re-run single structure for detailed error
python -c "
from Bio.PDB import PDBParser
from prodigy_prot.modules.prodigy import Prodigy

s = PDBParser().get_structure('x', 'failed.pdb')
p = Prodigy(s[0], selection=['H', 'T'])
p.predict()
print(p.as_dict())
"
```

## Performance Tuning

### Batch Size

Higher batch size = less IPC overhead, but less granular progress reporting.

```bash
# Small structures (< 500 residues): use larger batches
BATCH_SIZE=200

# Large structures (> 2000 residues): use smaller batches
BATCH_SIZE=25
```

### Workers Per Node

PRODIGY is CPU-bound but lightweight. Using all available cores is usually optimal.

```bash
# Match SLURM allocation
WORKERS_PER_NODE=${SLURM_CPUS_PER_TASK}

# Leave headroom for I/O
WORKERS_PER_NODE=$((SLURM_CPUS_PER_TASK - 2))
```

### Memory

Memory usage scales with workers and structure size. 2GB per worker is typically sufficient.

```bash
# 32 workers * 2GB = 64GB
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
```

## Dependencies

- Python 3.8+
- BioPython
- prodigy-prot (`pip install prodigy-prot`)

Ensure PRODIGY is available:
```bash
python -c "from prodigy_prot.modules.prodigy import Prodigy; print('OK')"
```
