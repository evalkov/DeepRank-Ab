# Stage A Process/Compute Optimization Changes (codex-test)

Date: 2026-02-07
Branch: `codex-test`

## Scope

Implemented process and compute-architecture optimizations only.
No scientific algorithms, feature formulas, or model logic were changed.

## Goals Implemented

1. Reduce risk of accidental single-core execution by validating effective CPU affinity and improving task launch binding.
2. Add split-phase Stage A execution mode to separate `prep/annotate/graphs` from `cluster`, improving orchestration flexibility and throughput.
3. Add non-invasive pipeline telemetry and queue controls to identify graph writer bottlenecks.
4. Add opt-in parallel ANARCI batching for large annotation workloads.
5. Preserve backward compatibility with feature flags and default-safe behavior.

## File-by-File Changes

## 1) `scripts/drab-A.slurm`

### Added runtime/launch controls

- New env vars:
  - `STAGEA_PHASE` (`full|prep_graphs|cluster_only`, default `full`)
  - `STAGEA_STRICT_AFFINITY` (default `0`)
  - `STAGEA_USE_SRUN_BIND` (default `1`)
  - `STAGEA_SRUN_CPU_BIND` (default `cores`)

### Added affinity verification

- New helper functions:
  - `_detect_affinity_cpus()` (uses `os.sched_getaffinity(0)` via Python)
  - `check_effective_affinity()`
- Logs effective affinity and compares against `NUM_CORES`.
- If `STAGEA_STRICT_AFFINITY=1`, exits when effective cpuset is smaller than requested cores.

### Added runtime diagnostics

- Logs key variables before execution:
  - `SLURM_CPUS_PER_TASK`, `SLURM_CPU_BIND`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VORO_OMP_THREADS`, phase.

### Added phase-driven Python invocation

- Builds `PY_ARGS` and adds:
  - `--prep-graphs-only` for `STAGEA_PHASE=prep_graphs`
  - `--cluster-only` for `STAGEA_PHASE=cluster_only`

### Added `srun`-bound payload launch

- If enabled/available, uses:
  - `srun --ntasks=1 --cpus-per-task=${NUM_CORES} --cpu-bind=${STAGEA_SRUN_CPU_BIND}`
- Falls back to direct `python3` launch when disabled/unavailable.

## 2) `scripts/split_stageA_cpu.py`

### Added split-phase/idempotent Stage A flow

- New marker behavior:
  - `STAGEA_GRAPHS_DONE` indicates prep+graphs publish complete.
  - `STAGEA_DONE` indicates full Stage A complete.

### Added metadata helper

- `_result_from_meta_dict()` safely reconstructs `StageAResult` from existing metadata.

### Added cluster-only execution entry point

- `run_stageA_cluster_only()`:
  - Requires existing `graphs.h5` and `STAGEA_GRAPHS_DONE`.
  - Runs clustering (unless `--no-cluster`).
  - Updates `meta_stageA.json` with `stagea_phase=cluster_only` and `cluster_s`.
  - Writes `STAGEA_DONE`.

### Extended main Stage A shard runner

- `run_stageA_one_shard()` new parameters:
  - `publish_stagea_done` (bool)
  - `stagea_phase` (str)
- Behavior changes:
  - If full run sees `STAGEA_GRAPHS_DONE` and published graphs, it short-circuits to cluster-only finalization.
  - Prep-only mode writes `STAGEA_GRAPHS_DONE` without `STAGEA_DONE`.
  - Writes `stagea_phase` into metadata.
- Added `try/finally` cleanup for local temp directory.

### CLI additions

- New flags:
  - `--prep-graphs-only`
  - `--cluster-only`
- Mutual exclusion validation enforced.
- Main execution now dispatches to:
  - full mode (legacy behavior)
  - prep-graphs-only mode
  - cluster-only mode

## 3) `scripts/run_pipeline.py`

### Added Stage A env plumbing

Stage A env now also sets:

- `STAGEA_SPLIT_MODE`
- `STAGEA_STRICT_AFFINITY`
- `STAGEA_USE_SRUN_BIND`
- `STAGEA_SRUN_CPU_BIND`
- `GRAPH_PIPELINE_TELEMETRY`
- `GRAPH_RESULT_QUEUE_MAXSIZE`
- `GRAPH_WRITER_LOG_EVERY_S`
- `ANARCI_PARALLEL_BATCHES`

### Enhanced Stage A submission function

- `run_stage_a_processing()` now accepts:
  - `env_overrides`
  - `stage_label`
- Enables phase-specific array submissions without duplicating submission logic.

### Added split-mode orchestration

When `stage_a.split_mode: true`:

1. Submit Stage A phase `prep_graphs` array (`A1`).
2. Submit Stage A phase `cluster_only` array (`A2`) dependent on `A1`.
3. Downstream Stage B depends on `A2` as the final Stage A completion point.

Legacy single-array Stage A path remains unchanged when split mode is disabled.

## 4) `src/GraphGenMP.py`

### Added safe env parsers

- `_env_int`, `_env_float`, `_env_flag` for telemetry/queue configuration.

### Added writer/queue telemetry controls

- New env-driven controls:
  - `GRAPH_PIPELINE_TELEMETRY` (default off)
  - `GRAPH_WRITER_LOG_EVERY_S` (default `30`)
  - `GRAPH_RESULT_QUEUE_MAXSIZE` (default `100`)

### Writer process telemetry

- `_writer_process()` signature extended with telemetry parameters.
- Periodically logs processed count, successful writes, errors, and effective processing rate.

### Main pipeline telemetry

- `_parallel_pipeline()` now logs queue settings when telemetry is enabled.
- Periodically logs enqueue rate from producer side.

No graph construction algorithm or feature computation was altered.

## 5) `src/tools/annotate.py`

### Added opt-in parallel ANARCI batching

- New helper: `_run_anarci_batch(records)`.
- `annotate_folder_one_by_one_mp()` now supports env-controlled batching:
  - `ANARCI_PARALLEL_BATCHES` (default `1`)
- If set >1 (and sufficient record volume), splits records into chunks and runs ANARCI in parallel process batches.
- Output ordering is preserved by ordered chunk assembly and length validation.

No ANARCI scheme/parameters were changed (`imgt`, `assign_germline=True`).

## 6) Configuration/Docs

### `scripts/pipeline.yaml.example`

Added Stage A knobs:

- `split_mode`
- `strict_affinity`
- `use_srun_bind`
- `srun_cpu_bind`
- `graph_pipeline_telemetry`
- `graph_result_queue_maxsize`
- `graph_writer_log_every_s`
- `anarci_parallel_batches`

### `docs/howto.md`

Updated Stage A config example with the new runtime/process controls.

### `scripts/PIPELINE.md`

Updated Stage A variable table with new env knobs and documented split-mode two-phase array behavior.

## Backward Compatibility / Rollback

Defaults keep behavior close to prior execution:

- Split mode disabled by default (`split_mode: false`)
- Strict affinity check disabled by default (`strict_affinity: 0`)
- Graph telemetry off by default (`graph_pipeline_telemetry: 0`)
- ANARCI parallel batching disabled by default (`anarci_parallel_batches: 1`)

To fully revert to previous Stage A runtime behavior via config:

1. `split_mode: false`
2. `strict_affinity: 0`
3. `use_srun_bind: 0`
4. `graph_pipeline_telemetry: 0`
5. `anarci_parallel_batches: 1`

## Validation Performed

Static/syntax checks passed:

- `python3 -m py_compile scripts/run_pipeline.py scripts/split_stageA_cpu.py src/GraphGenMP.py src/tools/annotate.py`
- `bash -n scripts/drab-A.slurm`

Runtime checks in this environment were limited because required dependencies are missing:

- `PyYAML` missing for `scripts/run_pipeline.py --help`
- `h5py` missing for `scripts/split_stageA_cpu.py --help`

## Notes

- No training/inference scientific kernels were modified.
- No residue/contact/BSA/Voronota formulas were changed.
- Changes focus on scheduling, execution topology, idempotency markers, and observability.

---

## Follow-Up Optimization Pass (2026-02-07, Later)

This follow-up focused specifically on the serial bottlenecks observed in Stage A telemetry.
Algorithms and scientific outputs were preserved.

### 1) `src/GraphGenMP.py` (IPC and prepass bottlenecks)

- Removed the extra worker -> parent -> writer relay for graph objects.
  - Workers now push graph results directly to the writer queue.
  - Parent process tracks lightweight completion tokens only.
- Parallelized the region-map prepass (previously serial) with a bounded thread pool.

Expected effect:
- Less parent-process serialization overhead.
- Better overlap between worker compute and writer throughput.
- Lower startup overhead before graph workers fully engage.

### 2) `src/AtomGraph.py` (repeated DB lookup bottlenecks)

- Added per-node coordinate caching in `get_graph()`.
- Reused cached coordinates for interface edge distance calculations.
- Reused cached coordinates for node `pos` assignment in `get_node_features()`.

Expected effect:
- Fewer repeated `db.get(...)` calls in hot loops.
- Lower Python/db overhead during edge and node feature construction.

### 3) `src/tools/contacts_dr2.py` (atomgraph contact expansion bottleneck)

- Refactored `add_residue_contacts_atomgraph()` to cache residue atom data once.
- Removed repeated full-residue expansion for every atom node.
- Built atom-node lookup indices once per unique atom.

Expected effect:
- Reduced redundant data expansion and lookup work prior to nonbonded matrix build.
- Lower memory/CPU overhead in edge feature computation.

### 4) `src/tools/BSA.py` (repeated centroid query bottleneck)

- Prefetched contact-residue centroids once per unique residue in `get_contact_residue_sasa()`.

Expected effect:
- Reduced repeated SQL coordinate fetches in BSA bookkeeping.

### 5) `src/tools/annotate.py` (small-shard ANARCI overhead)

- Raised ANARCI parallel-batch activation threshold:
  - now requires at least `max(64, ANARCI_PARALLEL_BATCHES * 32)` records.

Expected effect:
- Avoid process/chunk overhead dominating small shards.

### 6) `scripts/drab-A.slurm` + `scripts/run_pipeline.py` (thread knob cleanup + telemetry)

- Removed Stage A wiring for `VORO_OMP_THREADS` in launch/config path.
- Removed Stage A-specific OMP/VORO runtime knob logging in `drab-A.slurm`.
- Added process-scoped metrics attachment (best effort):
  - Stage A payload is launched in background.
  - monitor is started with `--pid <payload_pid>` where available.
  - enables `proc_metrics_*` output when the PID path is valid.

### 7) Documentation updates

- Updated `scripts/pipeline.yaml.example` to remove `voro_omp_threads`.
- Updated `docs/howto.md` tuning guidance to emphasize shard/process tuning.
