# Deprecated Scripts

This folder contains legacy helpers that are no longer part of the active
`run_pipeline.py` production path.

## Moved Here

- `compare_voronota.py`
- `inference.py`
- `large_scale_infer_vhh.py`
- `large_scale_infer_vhh_esm_opt.py`
- `stageA_progress.sh`
- `stageA_progress_live.py`
- `stageA_timing_breakdown.sh`
- `stageB_progress.sh`
- `stageB_progress_live.py`
- `watch_progress.sh`

## Notes

- Active live monitoring is now `scripts/progress_live.py`.
- `prodigy.slurm` and `prodigy_batch_worker.py` were intentionally left outside
  this folder and can be relocated separately.
