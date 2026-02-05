#!/usr/bin/env python3
"""
DeepRank-Ab Pipeline Launcher

Submits Stage A, B, and C jobs to SLURM with proper dependencies.
Reads configuration from a YAML file.

Usage:
    python run_pipeline.py pipeline.yaml              # Run all stages
    python run_pipeline.py pipeline.yaml --stage a    # Run only Stage A
    python run_pipeline.py pipeline.yaml --stage b    # Run only Stage B (requires A done)
    python run_pipeline.py pipeline.yaml --stage c    # Run only Stage C (requires B done)
    python run_pipeline.py pipeline.yaml --dry-run    # Show what would be submitted
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


@dataclass
class JobResult:
    stage: str
    job_id: Optional[str]
    command: str
    success: bool
    message: str = ""


@dataclass
class PipelineConfig:
    """Parsed and validated pipeline configuration."""
    run_root: Path
    pdb_root: Path
    deeprank_root: Path
    model_path: Path

    heavy: str
    light: str
    antigen: str

    stage_a: Dict = field(default_factory=dict)
    stage_b: Dict = field(default_factory=dict)
    stage_c: Dict = field(default_factory=dict)

    collect_metrics: bool = True
    metrics_interval: int = 2

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load and validate configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Required fields
        required = ["run_root", "pdb_root", "deeprank_root", "model_path", "chains"]
        for key in required:
            if key not in raw:
                raise ValueError(f"Missing required config key: {key}")

        chains = raw["chains"]
        for key in ["heavy", "antigen"]:
            if key not in chains:
                raise ValueError(f"Missing required chains.{key}")

        return cls(
            run_root=Path(raw["run_root"]).expanduser().resolve(),
            pdb_root=Path(raw["pdb_root"]).expanduser().resolve(),
            deeprank_root=Path(raw["deeprank_root"]).expanduser().resolve(),
            model_path=Path(raw["model_path"]).expanduser().resolve(),
            heavy=chains["heavy"],
            light=chains.get("light", "-"),
            antigen=chains["antigen"],
            stage_a=raw.get("stage_a", {}),
            stage_b=raw.get("stage_b", {}),
            stage_c=raw.get("stage_c", {}),
            collect_metrics=raw.get("collect_metrics", True),
            metrics_interval=raw.get("metrics_interval", 2),
        )

    def validate(self) -> List[str]:
        """Validate paths exist. Returns list of errors."""
        errors = []

        if not self.pdb_root.is_dir():
            errors.append(f"pdb_root not found: {self.pdb_root}")
        if not self.deeprank_root.is_dir():
            errors.append(f"deeprank_root not found: {self.deeprank_root}")
        if not self.model_path.is_file():
            errors.append(f"model_path not found: {self.model_path}")

        # Check SLURM scripts exist
        for stage in ["A", "B", "C"]:
            script = self.deeprank_root / "scripts" / f"drab-{stage}.slurm"
            if not script.is_file():
                errors.append(f"SLURM script not found: {script}")

        return errors


def build_env(cfg: PipelineConfig, stage: str) -> Dict[str, str]:
    """Build environment variables for a stage."""
    env = os.environ.copy()

    # Common variables
    env["RUN_ROOT"] = str(cfg.run_root)
    env["PDB_ROOT"] = str(cfg.pdb_root)
    env["DEEPRANK_ROOT"] = str(cfg.deeprank_root)
    env["HEAVY"] = cfg.heavy
    env["LIGHT"] = cfg.light
    env["ANTIGEN"] = cfg.antigen

    # Metrics
    env["COLLECT_COMPUTE_METRICS"] = "1" if cfg.collect_metrics else "0"
    env["COMPUTE_METRICS_INTERVAL"] = str(cfg.metrics_interval)

    if stage == "a":
        sa = cfg.stage_a
        env["NUM_CORES"] = str(sa.get("cores", 32))
        env["TARGET_SHARD_GB"] = str(sa.get("target_shard_gb", 0.1))
        env["MIN_PER_SHARD"] = str(sa.get("min_per_shard", 10))
        env["MAX_PER_SHARD"] = str(sa.get("max_per_shard", 100))
        env["GLOB_PAT"] = sa.get("glob", "**/*.pdb")
        env["VORONOTA_BINARY"] = sa.get("voronota_binary", "voronota_129")
        env["USE_FREESASA"] = str(sa.get("use_freesasa", 1))
        env["VORO_OMP_THREADS"] = str(sa.get("voro_omp_threads", 1))

    elif stage == "b":
        sb = cfg.stage_b
        env["MODEL_PATH"] = str(cfg.model_path)
        env["NUM_CORES"] = str(sb.get("cores", 32))
        env["ESM_GPUS"] = str(sb.get("gpus", 4))
        env["ESM_MODEL"] = sb.get("esm_model", "esm2_t33_650M_UR50D")
        env["ESM_TOKS_PER_BATCH"] = str(sb.get("esm_toks_per_batch", 12288))
        env["ESM_SCALAR_DTYPE"] = sb.get("esm_scalar_dtype", "float16")
        env["BATCH_SIZE"] = str(sb.get("batch_size", 64))
        env["DL_WORKERS"] = str(sb.get("dl_workers", 8))
        env["PREFETCH_FACTOR"] = str(sb.get("prefetch_factor", 4))

    elif stage == "c":
        sc = cfg.stage_c
        env["NUM_CORES"] = str(sc.get("cores", 4))

    return env


def build_sbatch_args(cfg: PipelineConfig, stage: str) -> List[str]:
    """Build sbatch arguments for a stage."""
    args = ["sbatch", "--parsable"]

    if stage == "a":
        sa = cfg.stage_a
        args.extend(["--partition", sa.get("partition", "norm")])
        args.extend(["--cpus-per-task", str(sa.get("cores", 32))])
        args.extend(["--mem", f"{sa.get('mem_gb', 64)}G"])
        args.extend(["--time", sa.get("time", "01:00:00")])
        if "array" in sa:
            args.extend(["--array", sa["array"]])

    elif stage == "b":
        sb = cfg.stage_b
        args.extend(["--partition", sb.get("partition", "gpu")])
        args.extend(["--cpus-per-task", str(sb.get("cores", 32))])
        args.extend(["--mem", f"{sb.get('mem_gb', 128)}G"])
        args.extend(["--time", sb.get("time", "04:00:00")])
        args.extend(["--gres", f"gpu:{sb.get('gpus', 4)}"])
        if "array" in sb:
            args.extend(["--array", sb["array"]])

    elif stage == "c":
        sc = cfg.stage_c
        args.extend(["--partition", sc.get("partition", "norm")])
        args.extend(["--cpus-per-task", str(sc.get("cores", 4))])
        args.extend(["--mem", f"{sc.get('mem_gb', 16)}G"])
        args.extend(["--time", sc.get("time", "02:00:00")])

    return args


def submit_job(
    cfg: PipelineConfig,
    stage: str,
    dependency: Optional[str] = None,
    dry_run: bool = False,
) -> JobResult:
    """Submit a SLURM job for a stage."""
    script = cfg.deeprank_root / "scripts" / f"drab-{stage.upper()}.slurm"

    args = build_sbatch_args(cfg, stage)
    if dependency:
        args.extend(["--dependency", f"afterok:{dependency}"])
    args.append(str(script))

    env = build_env(cfg, stage)
    cmd_str = " ".join(args)

    if dry_run:
        # Show environment variables that would be set
        env_diff = {k: v for k, v in env.items() if k not in os.environ or os.environ[k] != v}
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_diff.items()) if not k.startswith("_"))
        return JobResult(
            stage=stage.upper(),
            job_id=None,
            command=f"{env_str[:200]}... {cmd_str}" if len(env_str) > 200 else f"{env_str} {cmd_str}",
            success=True,
            message="[DRY RUN]",
        )

    try:
        result = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip()
        # Handle array jobs: "12345678_[0-9]" -> "12345678"
        if "_" in job_id:
            job_id = job_id.split("_")[0]
        return JobResult(
            stage=stage.upper(),
            job_id=job_id,
            command=cmd_str,
            success=True,
            message=f"Submitted job {job_id}",
        )
    except subprocess.CalledProcessError as e:
        return JobResult(
            stage=stage.upper(),
            job_id=None,
            command=cmd_str,
            success=False,
            message=f"sbatch failed: {e.stderr}",
        )


def run_pipeline(
    cfg: PipelineConfig,
    stages: List[str],
    dry_run: bool = False,
) -> List[JobResult]:
    """Run the pipeline for specified stages."""
    results = []
    prev_job_id: Optional[str] = None

    # Create run_root if it doesn't exist
    if not dry_run:
        cfg.run_root.mkdir(parents=True, exist_ok=True)
        # Save config to run_root for reproducibility
        config_copy = cfg.run_root / "pipeline_config.yaml"
        if not config_copy.exists():
            import shutil
            # We don't have the original path here, so skip this for now

    for stage in stages:
        # Determine dependency
        dependency = None
        if stage == "b" and "a" in stages:
            dependency = prev_job_id
        elif stage == "c" and "b" in stages:
            dependency = prev_job_id
        elif stage == "c" and "a" in stages and "b" not in stages:
            dependency = prev_job_id

        result = submit_job(cfg, stage, dependency, dry_run)
        results.append(result)

        if result.success and result.job_id:
            prev_job_id = result.job_id
        elif not result.success:
            break  # Stop on failure

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepRank-Ab Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to pipeline YAML config file",
    )
    parser.add_argument(
        "--stage", "-s",
        choices=["a", "b", "c", "all"],
        default="all",
        help="Stage to run (default: all)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate config, don't submit jobs",
    )

    args = parser.parse_args()

    # Load config
    if not args.config.is_file():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        cfg = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        return 1

    # Validate
    errors = cfg.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.validate_only:
        print("Configuration valid!")
        print(f"  run_root: {cfg.run_root}")
        print(f"  pdb_root: {cfg.pdb_root}")
        print(f"  model_path: {cfg.model_path}")
        print(f"  chains: H={cfg.heavy} L={cfg.light} Ag={cfg.antigen}")
        return 0

    # Determine stages to run
    if args.stage == "all":
        stages = ["a", "b", "c"]
    else:
        stages = [args.stage]

    # Print header
    print("=" * 60)
    print("DeepRank-Ab Pipeline")
    print("=" * 60)
    print(f"Config:    {args.config}")
    print(f"Run root:  {cfg.run_root}")
    print(f"PDB root:  {cfg.pdb_root}")
    print(f"Model:     {cfg.model_path}")
    print(f"Chains:    H={cfg.heavy} L={cfg.light} Ag={cfg.antigen}")
    print(f"Stages:    {' -> '.join(s.upper() for s in stages)}")
    print(f"Metrics:   {'ON' if cfg.collect_metrics else 'OFF'}")
    if args.dry_run:
        print(f"Mode:      DRY RUN")
    print("=" * 60)

    # Run pipeline
    results = run_pipeline(cfg, stages, args.dry_run)

    # Print results
    print("\nJob submission results:")
    all_ok = True
    job_chain = []
    for r in results:
        status = "OK" if r.success else "FAILED"
        if r.success:
            if r.job_id:
                print(f"  Stage {r.stage}: {status} (job {r.job_id})")
                job_chain.append(f"{r.stage}:{r.job_id}")
            else:
                print(f"  Stage {r.stage}: {r.message}")
                print(f"    {r.command}")
        else:
            print(f"  Stage {r.stage}: {status}")
            print(f"    {r.message}")
            all_ok = False

    if job_chain and not args.dry_run:
        print(f"\nJob chain: {' -> '.join(job_chain)}")
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Progress:    {cfg.deeprank_root}/scripts/watch_progress.sh {cfg.run_root}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
