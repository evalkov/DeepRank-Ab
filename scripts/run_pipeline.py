#!/usr/bin/env python3
"""
DeepRank-Ab Pipeline Launcher

Submits Stage A, B, and C jobs to SLURM with proper dependencies.
Dynamically sizes job arrays based on input data.

Usage:
    python run_pipeline.py pipeline.yaml              # Run all stages
    python run_pipeline.py pipeline.yaml --stage a    # Run only Stage A
    python run_pipeline.py pipeline.yaml --stage b    # Run only Stage B (requires A done)
    python run_pipeline.py pipeline.yaml --stage c    # Run only Stage C (requires B done)
    python run_pipeline.py pipeline.yaml --dry-run    # Show what would be submitted
    python run_pipeline.py pipeline.yaml --analyze    # Analyze input and show estimates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InputAnalysis:
    """Results of analyzing input PDB files."""
    pdb_count: int
    total_bytes: int
    estimated_shards: int
    recommended_cores: int
    avg_pdb_size: float
    glob_pattern: str

    def __str__(self) -> str:
        return (
            f"PDBs: {self.pdb_count:,} files ({self.total_bytes / 1e9:.2f} GB)\n"
            f"Estimated shards: {self.estimated_shards}\n"
            f"Recommended cores: {self.recommended_cores}\n"
            f"Avg PDB size: {self.avg_pdb_size / 1e3:.1f} KB"
        )


@dataclass
class JobResult:
    stage: str
    job_id: Optional[str]
    command: str
    success: bool
    message: str = ""


@dataclass
class InputHygieneState:
    enabled: bool
    mode: str
    effective_pdb_root: Path
    preflight_ok: bool
    preflight_json: Optional[Path] = None
    preflight_tsv: Optional[Path] = None
    preflight_summary: Optional[Dict[str, Any]] = None
    preflight_issue_counts: Dict[str, int] = field(default_factory=dict)
    preflight_recommend_cure: bool = False
    preflight_reasons: List[str] = field(default_factory=list)
    cure_requested: bool = False
    cure_job_id: Optional[str] = None


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
    input_hygiene: Dict = field(default_factory=dict)

    collect_metrics: bool = True
    metrics_interval: int = 2
    max_concurrent_a: int = 20  # Max concurrent Stage A jobs
    max_concurrent_b: int = 10  # Max concurrent Stage B jobs

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
            input_hygiene=raw.get("input_hygiene", {}),
            collect_metrics=raw.get("collect_metrics", True),
            metrics_interval=raw.get("metrics_interval", 2),
            max_concurrent_a=raw.get("max_concurrent_a", 20),
            max_concurrent_b=raw.get("max_concurrent_b", 10),
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

        # Input hygiene scripts (optional)
        mode = str((self.input_hygiene or {}).get("mode", "off")).strip().lower()
        valid_modes = {"off", "recommend", "required", "auto"}
        if mode not in valid_modes:
            errors.append(
                f"input_hygiene.mode must be one of {sorted(valid_modes)} (got: {mode})"
            )
        if mode != "off":
            checker = self.deeprank_root / "scripts" / "check_pdb_preflight.py"
            if not checker.is_file():
                errors.append(f"Input hygiene checker not found: {checker}")
            if mode == "auto":
                cure = self.deeprank_root / "scripts" / "cure_pdbs.sh"
                if not cure.is_file():
                    errors.append(f"Input hygiene cure script not found: {cure}")

        return errors


# =============================================================================
# Pre-flight Analysis
# =============================================================================

def analyze_input(cfg: PipelineConfig) -> InputAnalysis:
    """Analyze input PDBs to estimate resource requirements."""
    glob_pat = cfg.stage_a.get("glob", "**/*.pdb")
    target_shard_gb = cfg.stage_a.get("target_shard_gb", 0.1)
    min_per_shard = cfg.stage_a.get("min_per_shard", 10)
    max_per_shard = cfg.stage_a.get("max_per_shard", 100)

    # Find all PDBs
    pdbs = list(cfg.pdb_root.glob(glob_pat))
    if not pdbs:
        # Try non-recursive if recursive found nothing
        pdbs = list(cfg.pdb_root.glob("*.pdb"))

    pdb_count = len(pdbs)
    if pdb_count == 0:
        return InputAnalysis(
            pdb_count=0,
            total_bytes=0,
            estimated_shards=0,
            recommended_cores=1,
            avg_pdb_size=0,
            glob_pattern=glob_pat,
        )

    # Calculate sizes
    total_bytes = sum(p.stat().st_size for p in pdbs)
    avg_pdb_size = total_bytes / pdb_count

    # Estimate shards (mimics split_stageA_cpu.py logic)
    target_bytes = int(target_shard_gb * (1024**3))
    estimated_shards = 0
    cur_bytes = 0
    cur_count = 0

    for p in sorted(pdbs, key=lambda x: x.stat().st_size, reverse=True):
        sz = p.stat().st_size
        if cur_count > 0 and (cur_count >= max_per_shard or cur_bytes + sz > target_bytes) and cur_count >= min_per_shard:
            estimated_shards += 1
            cur_bytes = 0
            cur_count = 0
        cur_bytes += sz
        cur_count += 1

    if cur_count > 0:
        estimated_shards += 1

    # Recommend cores based on PDBs per shard
    pdbs_per_shard = pdb_count / max(1, estimated_shards)
    recommended_cores = min(32, max(8, int(pdbs_per_shard / 5)))

    return InputAnalysis(
        pdb_count=pdb_count,
        total_bytes=total_bytes,
        estimated_shards=estimated_shards,
        recommended_cores=recommended_cores,
        avg_pdb_size=avg_pdb_size,
        glob_pattern=glob_pat,
    )


def count_existing_shards(cfg: PipelineConfig) -> int:
    """Count existing shard list files."""
    shard_lists_dir = cfg.run_root / "shard_lists"
    if not shard_lists_dir.is_dir():
        return 0
    return len(list(shard_lists_dir.glob("shard_*.lst")))


def count_completed_stage_a(cfg: PipelineConfig) -> int:
    """Count completed Stage A shards."""
    shards_dir = cfg.run_root / "shards"
    if not shards_dir.is_dir():
        return 0
    return len(list(shards_dir.glob("shard_*/STAGEA_DONE")))


def count_completed_stage_b(cfg: PipelineConfig) -> int:
    """Count completed Stage B predictions."""
    preds_dir = cfg.run_root / "preds"
    if not preds_dir.is_dir():
        return 0
    return len(list(preds_dir.glob("DONE_shard_*.ok")))


def _int_or_default(v: Any, default: int, min_value: Optional[int] = None) -> int:
    try:
        out = int(v)
    except Exception:
        out = default
    if min_value is not None:
        out = max(min_value, out)
    return out


def _float_or_default(v: Any, default: float, min_value: Optional[float] = None) -> float:
    try:
        out = float(v)
    except Exception:
        out = default
    if min_value is not None:
        out = max(min_value, out)
    return out


def _resolve_input_hygiene(cfg: PipelineConfig) -> Dict[str, Any]:
    raw = cfg.input_hygiene or {}
    cure_cfg = raw.get("cure", {}) if isinstance(raw.get("cure", {}), dict) else {}

    mode = str(raw.get("mode", "off")).strip().lower()
    if mode not in {"off", "recommend", "required", "auto"}:
        mode = "off"

    out_dir_raw = raw.get("cure_output_dir", "pdb_cured")
    out_dir = Path(out_dir_raw).expanduser()
    if not out_dir.is_absolute():
        out_dir = (cfg.run_root / out_dir).resolve()

    return {
        "mode": mode,
        "sample_n": _int_or_default(raw.get("sample_n", 50), 50, min_value=1),
        "sample_mode": str(raw.get("sample_mode", "first")).strip().lower(),
        "sample_seed": raw.get("sample_seed", None),
        "glob": str(raw.get("glob", cfg.stage_a.get("glob", "**/*.pdb"))),
        "auto_cure_min_fraction": _float_or_default(
            raw.get("auto_cure_min_fraction", 0.20), 0.20, min_value=0.0
        ),
        "fail_on_preflight_error": bool(
            raw.get("fail_on_preflight_error", mode in {"required", "auto"})
        ),
        "cure_output_dir": out_dir,
        "cure_partition": str(cure_cfg.get("partition", "norm")),
        "cure_cores": _int_or_default(cure_cfg.get("cores", 32), 32, min_value=1),
        "cure_time": str(cure_cfg.get("time", "08:00:00")),
        "cure_files_per_task": _int_or_default(
            cure_cfg.get("files_per_task", 2000), 2000, min_value=1
        ),
        "cure_jobs_per_task": _int_or_default(
            cure_cfg.get("jobs_per_task", 0), 0, min_value=0
        ),
        "cure_skip_existing": _int_or_default(
            cure_cfg.get("skip_existing", 1), 1, min_value=0
        ),
    }


def _parse_submitted_job_id(text: str) -> Optional[str]:
    if not text:
        return None
    # sbatch default output: "Submitted batch job <id>"
    matches = re.findall(r"Submitted batch job\s+(\d+)", text)
    if matches:
        return matches[-1]
    # Fallback for parsable-like output.
    m = re.search(r"\b(\d{5,})\b", text)
    return m.group(1) if m else None


def _write_input_hygiene_summary(cfg: PipelineConfig, state: InputHygieneState) -> None:
    if not cfg.run_root.exists():
        return
    summary_path = cfg.run_root / "input_hygiene_summary.json"
    obj: Dict[str, Any] = {
        "enabled": state.enabled,
        "mode": state.mode,
        "effective_pdb_root": str(state.effective_pdb_root),
        "preflight_ok": state.preflight_ok,
        "preflight_json": str(state.preflight_json) if state.preflight_json else None,
        "preflight_tsv": str(state.preflight_tsv) if state.preflight_tsv else None,
        "preflight_summary": state.preflight_summary,
        "preflight_issue_counts": state.preflight_issue_counts,
        "preflight_recommend_cure": state.preflight_recommend_cure,
        "preflight_reasons": state.preflight_reasons,
        "cure_requested": state.cure_requested,
        "cure_job_id": state.cure_job_id,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(summary_path, "w") as f:
        json.dump(obj, f, indent=2)


def run_input_hygiene(
    cfg: PipelineConfig,
    dry_run: bool = False,
) -> Tuple[InputHygieneState, List[JobResult]]:
    """
    Run optional preflight/cure integration before Stage A.
    Returns (state, pre_stage_results). If state.preflight_ok is False and mode
    requires hygiene, caller should abort pipeline.
    """
    ih = _resolve_input_hygiene(cfg)
    mode = ih["mode"]
    results: List[JobResult] = []

    if mode == "off":
        state = InputHygieneState(
            enabled=False,
            mode=mode,
            effective_pdb_root=cfg.pdb_root,
            preflight_ok=True,
        )
        return state, results

    checker = cfg.deeprank_root / "scripts" / "check_pdb_preflight.py"
    pre_json = cfg.run_root / "preflight_report_pre.json"
    pre_tsv = cfg.run_root / "preflight_report_pre.tsv"

    cmd = [
        sys.executable,
        str(checker),
        str(cfg.pdb_root),
        "--sample-n",
        str(ih["sample_n"]),
        "--glob",
        ih["glob"],
        "--sample-mode",
        ih["sample_mode"] if ih["sample_mode"] in {"first", "random"} else "first",
        "--heavy",
        cfg.heavy,
        "--light",
        cfg.light,
        "--antigen",
        cfg.antigen,
        "--json-out",
        str(pre_json),
        "--tsv-out",
        str(pre_tsv),
        "--fail-on",
        "never",
    ]
    if ih["sample_seed"] is not None:
        cmd.extend(["--seed", str(ih["sample_seed"])])

    if dry_run:
        results.append(
            JobResult(
                stage="PREFLIGHT",
                job_id=None,
                command=" ".join(cmd),
                success=True,
                message="[DRY RUN] would run input preflight checker",
            )
        )
        state = InputHygieneState(
            enabled=True,
            mode=mode,
            effective_pdb_root=cfg.pdb_root,
            preflight_ok=True,
            preflight_json=pre_json,
            preflight_tsv=pre_tsv,
        )
        return state, results

    proc = subprocess.run(cmd, capture_output=True, text=True)
    preflight_ok = proc.returncode == 0 and pre_json.is_file()

    pre_data: Dict[str, Any] = {}
    pre_summary: Dict[str, Any] = {}
    issue_counts: Dict[str, int] = {}
    rec_cure = False
    cure_reasons: List[str] = []
    cure_signal_frac = 0.0
    total_sampled = 0
    fail_count = 0
    warn_count = 0

    if preflight_ok:
        try:
            pre_data = json.loads(pre_json.read_text())
            pre_summary = dict(pre_data.get("summary", {}) or {})
            issue_counts = {
                str(k): int(v)
                for k, v in dict(pre_data.get("issue_counts", {}) or {}).items()
            }
            recs = dict(pre_data.get("recommendations", {}) or {})
            rec_cure = bool(recs.get("recommend_cure_pdbs", False))
            cure_reasons = [str(x) for x in (recs.get("cure_reasons", []) or [])]
            total_sampled = _int_or_default(pre_summary.get("total", 0), 0, min_value=0)
            fail_count = _int_or_default(pre_summary.get("fail", 0), 0, min_value=0)
            warn_count = _int_or_default(pre_summary.get("warn", 0), 0, min_value=0)
            hydrogen_files = _int_or_default(
                issue_counts.get("high_hydrogen_fraction", 0), 0, min_value=0
            )
            oxt_files = _int_or_default(issue_counts.get("has_oxt", 0), 0, min_value=0)
            signal_files = max(hydrogen_files, oxt_files)
            cure_signal_frac = signal_files / max(1, total_sampled)
        except Exception:
            preflight_ok = False

    if preflight_ok:
        msg = (
            f"PASS={pre_summary.get('pass', 0)} WARN={pre_summary.get('warn', 0)} "
            f"FAIL={pre_summary.get('fail', 0)} sample={pre_summary.get('total', 0)}"
        )
        results.append(
            JobResult(
                stage="PREFLIGHT",
                job_id=None,
                command=" ".join(cmd),
                success=True,
                message=msg,
            )
        )
    else:
        err_excerpt = (proc.stderr or proc.stdout or "").strip()
        if len(err_excerpt) > 240:
            err_excerpt = err_excerpt[:240] + "..."
        results.append(
            JobResult(
                stage="PREFLIGHT",
                job_id=None,
                command=" ".join(cmd),
                success=False,
                message=f"checker failed (rc={proc.returncode}): {err_excerpt}",
            )
        )
        should_abort = bool(ih["fail_on_preflight_error"])
        state = InputHygieneState(
            enabled=True,
            mode=mode,
            effective_pdb_root=cfg.pdb_root,
            preflight_ok=not should_abort,
            preflight_json=pre_json if pre_json.exists() else None,
            preflight_tsv=pre_tsv if pre_tsv.exists() else None,
        )
        return state, results

    # Enforce preflight failures for required mode.
    if mode == "required" and fail_count > 0:
        state = InputHygieneState(
            enabled=True,
            mode=mode,
            effective_pdb_root=cfg.pdb_root,
            preflight_ok=False,
            preflight_json=pre_json,
            preflight_tsv=pre_tsv,
            preflight_summary=pre_summary,
            preflight_issue_counts=issue_counts,
            preflight_recommend_cure=rec_cure,
            preflight_reasons=cure_reasons,
        )
        return state, results

    # Optional auto-cure submission.
    cure_requested = (
        mode == "auto"
        and rec_cure
        and cure_signal_frac >= float(ih["auto_cure_min_fraction"])
    )
    cure_job_id: Optional[str] = None
    effective_root = cfg.pdb_root

    if cure_requested:
        cure_script = cfg.deeprank_root / "scripts" / "cure_pdbs.sh"
        cure_out = Path(ih["cure_output_dir"]).resolve()
        cure_out.mkdir(parents=True, exist_ok=True)

        cure_env = os.environ.copy()
        cure_env["PARTITION"] = ih["cure_partition"]
        cure_env["CPUS"] = str(ih["cure_cores"])
        cure_env["TIME"] = ih["cure_time"]
        cure_env["FILES_PER_TASK"] = str(ih["cure_files_per_task"])
        cure_env["JOBS_PER_TASK"] = str(ih["cure_jobs_per_task"])
        cure_env["SKIP_EXISTING"] = str(ih["cure_skip_existing"])

        cure_cmd = [str(cure_script), str(cfg.pdb_root), str(cure_out)]
        cure_proc = subprocess.run(cure_cmd, env=cure_env, capture_output=True, text=True)
        parsed_job = _parse_submitted_job_id(
            (cure_proc.stdout or "") + "\n" + (cure_proc.stderr or "")
        )

        if cure_proc.returncode == 0 and parsed_job:
            cure_job_id = parsed_job
            effective_root = cure_out
            results.append(
                JobResult(
                    stage="CURE",
                    job_id=parsed_job,
                    command=" ".join(cure_cmd),
                    success=True,
                    message=f"Submitted cure job {parsed_job}",
                )
            )
        else:
            err_excerpt = (cure_proc.stderr or cure_proc.stdout or "").strip()
            if len(err_excerpt) > 240:
                err_excerpt = err_excerpt[:240] + "..."
            results.append(
                JobResult(
                    stage="CURE",
                    job_id=None,
                    command=" ".join(cure_cmd),
                    success=False,
                    message=f"cure submission failed (rc={cure_proc.returncode}): {err_excerpt}",
                )
            )
            state = InputHygieneState(
                enabled=True,
                mode=mode,
                effective_pdb_root=cfg.pdb_root,
                preflight_ok=False,
                preflight_json=pre_json,
                preflight_tsv=pre_tsv,
                preflight_summary=pre_summary,
                preflight_issue_counts=issue_counts,
                preflight_recommend_cure=rec_cure,
                preflight_reasons=cure_reasons,
                cure_requested=True,
            )
            return state, results

    state = InputHygieneState(
        enabled=True,
        mode=mode,
        effective_pdb_root=effective_root,
        preflight_ok=True,
        preflight_json=pre_json,
        preflight_tsv=pre_tsv,
        preflight_summary=pre_summary,
        preflight_issue_counts=issue_counts,
        preflight_recommend_cure=rec_cure,
        preflight_reasons=cure_reasons,
        cure_requested=cure_requested,
        cure_job_id=cure_job_id,
    )
    return state, results


# =============================================================================
# Environment and SBATCH Building
# =============================================================================

def build_env(cfg: PipelineConfig, stage: str, **overrides) -> Dict[str, str]:
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

    if stage == "a" or stage == "a_shard":
        sa = cfg.stage_a
        # NUM_CORES intentionally not set — drab-A.slurm auto-detects from
        # SLURM_CPUS_PER_TASK so it uses all allocated cores even when the
        # node has more than the requested minimum.
        env["TARGET_SHARD_GB"] = str(sa.get("target_shard_gb", 0.1))
        env["MIN_PER_SHARD"] = str(sa.get("min_per_shard", 10))
        env["MAX_PER_SHARD"] = str(sa.get("max_per_shard", 100))
        env["GLOB_PAT"] = sa.get("glob", "**/*.pdb")
        env["STAGEA_SPLIT_MODE"] = "1" if sa.get("split_mode", False) else "0"
        env["STAGEA_STRICT_AFFINITY"] = str(sa.get("strict_affinity", 0))
        env["STAGEA_USE_SRUN_BIND"] = str(sa.get("use_srun_bind", 1))
        env["STAGEA_SRUN_CPU_BIND"] = str(sa.get("srun_cpu_bind", "cores"))
        env["GRAPH_PIPELINE_TELEMETRY"] = str(sa.get("graph_pipeline_telemetry", 0))
        env["GRAPH_RESULT_QUEUE_MAXSIZE"] = str(sa.get("graph_result_queue_maxsize", 100))
        env["GRAPH_WRITER_LOG_EVERY_S"] = str(sa.get("graph_writer_log_every_s", 30))
        env["ANARCI_PARALLEL_BATCHES"] = str(sa.get("anarci_parallel_batches", 1))

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
        env["SHARDS_PER_TASK"] = str(sb.get("shards_per_task", 20))

    elif stage == "c":
        sc = cfg.stage_c
        env["NUM_CORES"] = str(sc.get("cores", 4))

    # Apply overrides
    for k, v in overrides.items():
        env[k] = str(v)

    return env


def build_sbatch_args(
    cfg: PipelineConfig,
    stage: str,
    array_spec: Optional[str] = None,
    dependency: Optional[str] = None,
    job_name: Optional[str] = None,
) -> List[str]:
    """Build sbatch arguments for a stage.

    All SBATCH directives are passed here on the CLI so that the .slurm
    scripts contain no #SBATCH headers and cannot conflict.
    """
    args = ["sbatch", "--parsable"]
    args.extend(["--nodes", "1"])
    args.extend(["--ntasks", "1"])

    if job_name:
        args.extend(["--job-name", job_name])

    if dependency:
        dep_type = "afterany" if stage == "c" else "afterok"
        args.extend(["--dependency", f"{dep_type}:{dependency}"])

    # Log files go into {run_root}/logs/
    log_dir = cfg.run_root / "logs"
    if array_spec:
        log_pat = str(log_dir / "%x_%A_%a")
    else:
        log_pat = str(log_dir / "%x_%j")
    args.extend(["--output", f"{log_pat}.log"])
    args.extend(["--error", f"{log_pat}.err"])

    if stage == "a" or stage == "a_shard":
        sa = cfg.stage_a
        args.extend(["--partition", sa.get("partition", "norm")])
        args.extend(["--cpus-per-task", str(sa.get("cores", 32))])
        args.extend(["--mem", f"{sa.get('mem_gb', 64)}G"])
        args.extend(["--time", sa.get("time", "01:00:00")])
        if array_spec:
            args.extend(["--array", array_spec])

    elif stage == "b":
        sb = cfg.stage_b
        args.extend(["--partition", sb.get("partition", "gpu")])
        args.extend(["--cpus-per-task", str(sb.get("cores", 32))])
        args.extend(["--mem", f"{sb.get('mem_gb', 128)}G"])
        args.extend(["--time", sb.get("time", "04:00:00")])
        args.extend(["--gres", f"gpu:{sb.get('gpus', 4)}"])
        if array_spec:
            args.extend(["--array", array_spec])

    elif stage == "c":
        sc = cfg.stage_c
        args.extend(["--partition", sc.get("partition", "norm")])
        args.extend(["--cpus-per-task", str(sc.get("cores", 4))])
        args.extend(["--mem", f"{sc.get('mem_gb', 16)}G"])
        args.extend(["--time", sc.get("time", "02:00:00")])

    return args


# =============================================================================
# Job Submission
# =============================================================================

def submit_job(
    args: List[str],
    script: Path,
    env: Dict[str, str],
    dry_run: bool = False,
    stage_name: str = "",
) -> JobResult:
    """Submit a SLURM job."""
    full_args = args + [str(script)]
    cmd_str = " ".join(full_args)

    if dry_run:
        env_diff = {k: v for k, v in env.items() if k not in os.environ or os.environ[k] != v}
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_diff.items()) if not k.startswith("_"))
        display = f"{env_str[:150]}..." if len(env_str) > 150 else env_str
        return JobResult(
            stage=stage_name,
            job_id=None,
            command=f"{display} {cmd_str}",
            success=True,
            message="[DRY RUN]",
        )

    try:
        result = subprocess.run(
            full_args,
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
            stage=stage_name,
            job_id=job_id,
            command=cmd_str,
            success=True,
            message=f"Submitted job {job_id}",
        )
    except subprocess.CalledProcessError as e:
        return JobResult(
            stage=stage_name,
            job_id=None,
            command=cmd_str,
            success=False,
            message=f"sbatch failed: {e.stderr}",
        )


def run_sharding_only(
    cfg: PipelineConfig,
    dry_run: bool = False,
    dependency: Optional[str] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[JobResult, int]:
    """
    Run Stage A sharding only (task 0 with MAKE_SHARDS_ONLY=1).
    Returns (result, estimated_shards).
    """
    analysis = analyze_input(cfg)

    if analysis.pdb_count == 0:
        return JobResult(
            stage="A-SHARD",
            job_id=None,
            command="",
            success=False,
            message="No PDB files found",
        ), 0

    script = cfg.deeprank_root / "scripts" / "drab-A.slurm"
    env = build_env(cfg, "a_shard", MAKE_SHARDS_ONLY="1", **(env_overrides or {}))

    # Single-element array so SLURM_ARRAY_TASK_ID=0 is set (script requires it)
    args = build_sbatch_args(
        cfg,
        "a_shard",
        array_spec="0-0",
        dependency=dependency,
        job_name="drab-A-shard",
    )
    # Override to short time — sharding is fast
    filtered = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--time":
            skip_next = True
            continue
        filtered.append(a)
    filtered.extend(["--time", "00:10:00"])
    args = filtered

    result = submit_job(args, script, env, dry_run, "A-SHARD")
    return result, analysis.estimated_shards


def run_stage_a_processing(
    cfg: PipelineConfig,
    n_shards: int,
    dependency: Optional[str] = None,
    dry_run: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
    stage_label: str = "A",
) -> JobResult:
    """Run Stage A processing for all shards."""
    if n_shards <= 0:
        return JobResult(
            stage="A",
            job_id=None,
            command="",
            success=False,
            message="No shards to process",
        )

    script = cfg.deeprank_root / "scripts" / "drab-A.slurm"
    env = build_env(cfg, "a", **(env_overrides or {}))

    # Dynamic array sizing
    max_concurrent = cfg.max_concurrent_a
    array_spec = f"0-{n_shards - 1}%{max_concurrent}"

    args = build_sbatch_args(
        cfg, "a",
        array_spec=array_spec,
        dependency=dependency,
        job_name="drab-A",
    )

    return submit_job(args, script, env, dry_run, f"{stage_label}[0-{n_shards-1}]")


def run_stage_b(
    cfg: PipelineConfig,
    n_shards: int,
    dependency: Optional[str] = None,
    dry_run: bool = False,
) -> JobResult:
    """Run Stage B for all shards."""
    if n_shards <= 0:
        return JobResult(
            stage="B",
            job_id=None,
            command="",
            success=False,
            message="No shards to process",
        )

    script = cfg.deeprank_root / "scripts" / "drab-B.slurm"
    env = build_env(cfg, "b")

    # Each GPU task processes SHARDS_PER_TASK CPU shards
    shards_per_task = cfg.stage_b.get("shards_per_task", 20)
    n_tasks = math.ceil(n_shards / shards_per_task)

    # Dynamic array sizing
    max_concurrent = cfg.max_concurrent_b
    array_spec = f"0-{n_tasks - 1}%{max_concurrent}"

    args = build_sbatch_args(
        cfg, "b",
        array_spec=array_spec,
        dependency=dependency,
        job_name="drab-B",
    )

    return submit_job(args, script, env, dry_run, f"B[0-{n_tasks-1}]")


def run_stage_c(
    cfg: PipelineConfig,
    dependency: Optional[str] = None,
    dry_run: bool = False,
) -> JobResult:
    """Run Stage C (merge)."""
    script = cfg.deeprank_root / "scripts" / "drab-C.slurm"
    env = build_env(cfg, "c")

    args = build_sbatch_args(
        cfg, "c",
        dependency=dependency,
        job_name="drab-C",
    )

    return submit_job(args, script, env, dry_run, "C")


# =============================================================================
# Pipeline Orchestration
# =============================================================================

def run_pipeline_dynamic(
    cfg: PipelineConfig,
    stages: List[str],
    dry_run: bool = False,
) -> List[JobResult]:
    """
    Run pipeline with dynamic resource allocation.

    For Stage A: Two-phase (shard first, then process)
    For Stage B: Sizes array based on completed Stage A shards
    """
    results = []
    prev_job_id: Optional[str] = None
    stage_a_env_overrides: Dict[str, str] = {}
    stage_a_dependency: Optional[str] = None

    # Create run_root and logs directory
    if not dry_run:
        cfg.run_root.mkdir(parents=True, exist_ok=True)
        (cfg.run_root / "logs").mkdir(exist_ok=True)

    # Check for existing state
    existing_shards = count_existing_shards(cfg)
    completed_a = count_completed_stage_a(cfg)
    completed_b = count_completed_stage_b(cfg)

    print(f"\nExisting state:")
    print(f"  Shard lists: {existing_shards}")
    print(f"  Stage A done: {completed_a}")
    print(f"  Stage B done: {completed_b}")

    # Stage A
    if "a" in stages:
        stage_a_dependency = prev_job_id

        # Optional input hygiene phase (preflight + optional auto-cure submit).
        hygiene_state, hygiene_results = run_input_hygiene(cfg, dry_run=dry_run)
        results.extend(hygiene_results)
        if not hygiene_state.preflight_ok:
            if not dry_run:
                _write_input_hygiene_summary(cfg, hygiene_state)
            return results
        if not dry_run:
            _write_input_hygiene_summary(cfg, hygiene_state)
        if hygiene_state.effective_pdb_root != cfg.pdb_root:
            stage_a_env_overrides["PDB_ROOT"] = str(hygiene_state.effective_pdb_root)
            print(f"\n  Input hygiene effective PDB root: {hygiene_state.effective_pdb_root}")
        if hygiene_state.cure_job_id:
            stage_a_dependency = hygiene_state.cure_job_id
        if hygiene_state.cure_requested:
            stage_a_env_overrides["FORCE_RESHARD"] = "1"
            if (existing_shards > 0 or completed_a > 0 or completed_b > 0) and not dry_run:
                results.append(
                    JobResult(
                        stage="INPUT-HYGIENE",
                        job_id=None,
                        command="",
                        success=False,
                        message=(
                            "Auto-cure requested but run_root already has existing shard/pred outputs. "
                            "Use a fresh run_root (recommended) or disable input_hygiene auto mode."
                        ),
                    )
                )
                return results
            if existing_shards > 0:
                print("\n  Auto-cure enabled: forcing shard list rebuild.")
                existing_shards = 0

        analysis = analyze_input(cfg)
        split_mode = bool(cfg.stage_a.get("split_mode", False))
        print(f"\nInput analysis:")
        print(f"  {analysis}")

        if analysis.pdb_count == 0:
            results.append(JobResult("A", None, "", False, "No PDB files found"))
            return results

        if existing_shards > 0 and not dry_run:
            print(f"\n  Using existing {existing_shards} shard lists")
            n_shards = existing_shards
        else:
            # Phase 1: Create shards
            print(f"\n  Phase 1: Creating shard lists...")
            shard_result, estimated = run_sharding_only(
                cfg,
                dry_run,
                dependency=stage_a_dependency,
                env_overrides=stage_a_env_overrides if stage_a_env_overrides else None,
            )
            results.append(shard_result)

            if not shard_result.success:
                return results

            stage_a_dependency = shard_result.job_id
            n_shards = estimated

            if not dry_run:
                # Wait briefly for sharding to complete (it's fast)
                # Or use the estimated count
                print(f"  Estimated shards: {n_shards}")

        # Phase 2: Process shards
        if split_mode:
            print(f"\n  Phase 2a: Processing {n_shards} shards (prep+graphs)...")
            prep_result = run_stage_a_processing(
                cfg,
                n_shards,
                dependency=stage_a_dependency,
                dry_run=dry_run,
                env_overrides={**stage_a_env_overrides, "STAGEA_PHASE": "prep_graphs"},
                stage_label="A1",
            )
            results.append(prep_result)
            if not prep_result.success:
                return results

            print(f"\n  Phase 2b: Processing {n_shards} shards (cluster-only)...")
            cluster_result = run_stage_a_processing(
                cfg,
                n_shards,
                dependency=prep_result.job_id,
                dry_run=dry_run,
                env_overrides={**stage_a_env_overrides, "STAGEA_PHASE": "cluster_only"},
                stage_label="A2",
            )
            results.append(cluster_result)
            if not cluster_result.success:
                return results
            prev_job_id = cluster_result.job_id
        else:
            print(f"\n  Phase 2: Processing {n_shards} shards...")
            proc_result = run_stage_a_processing(
                cfg,
                n_shards,
                stage_a_dependency,
                dry_run,
                env_overrides=stage_a_env_overrides if stage_a_env_overrides else None,
            )
            results.append(proc_result)

            if not proc_result.success:
                return results

            prev_job_id = proc_result.job_id

    # Stage B
    if "b" in stages:
        # Determine number of shards
        if "a" in stages:
            # Use same count as Stage A
            n_shards = existing_shards if existing_shards > 0 else analyze_input(cfg).estimated_shards
        else:
            # Count completed Stage A shards
            n_shards = count_completed_stage_a(cfg)
            if n_shards == 0:
                # Fall back to shard lists
                n_shards = count_existing_shards(cfg)

        if n_shards == 0:
            results.append(JobResult("B", None, "", False, "No shards found for Stage B"))
            return results

        print(f"\n  Stage B: Processing {n_shards} shards...")
        b_result = run_stage_b(cfg, n_shards, prev_job_id, dry_run)
        results.append(b_result)

        if not b_result.success:
            return results

        prev_job_id = b_result.job_id

    # Stage C
    if "c" in stages:
        print(f"\n  Stage C: Merging predictions...")
        c_result = run_stage_c(cfg, prev_job_id, dry_run)
        results.append(c_result)

    return results


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    pipeline_invoked_at = datetime.now().isoformat(timespec="seconds")
    parser = argparse.ArgumentParser(
        description="DeepRank-Ab Pipeline Launcher (Dynamic Resource Allocation)",
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
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze input and show resource estimates, don't submit",
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

    # Analyze only
    if args.analyze:
        print("=" * 60)
        print("Input Analysis")
        print("=" * 60)
        analysis = analyze_input(cfg)
        ih_cfg = _resolve_input_hygiene(cfg)
        print(f"\nPDB Root: {cfg.pdb_root}")
        print(f"Glob: {analysis.glob_pattern}")
        print(f"Input hygiene mode: {ih_cfg['mode']}")
        print(f"\n{analysis}")

        if analysis.estimated_shards > 0:
            print(f"\nRecommended settings:")
            print(f"  stage_a.cores: {analysis.recommended_cores}")
            print(f"  Stage A array: 0-{analysis.estimated_shards - 1}%{cfg.max_concurrent_a}")
            print(f"  Stage B array: 0-{analysis.estimated_shards - 1}%{cfg.max_concurrent_b}")

        # Show existing state
        existing_shards = count_existing_shards(cfg)
        completed_a = count_completed_stage_a(cfg)
        completed_b = count_completed_stage_b(cfg)
        if existing_shards > 0 or completed_a > 0 or completed_b > 0:
            print(f"\nExisting run state:")
            print(f"  Shard lists: {existing_shards}")
            print(f"  Stage A completed: {completed_a}")
            print(f"  Stage B completed: {completed_b}")

        return 0

    if args.validate_only:
        print("Configuration valid!")
        print(f"  run_root: {cfg.run_root}")
        print(f"  pdb_root: {cfg.pdb_root}")
        print(f"  model_path: {cfg.model_path}")
        print(f"  chains: H={cfg.heavy} L={cfg.light} Ag={cfg.antigen}")
        return 0

    # Determine stages
    if args.stage == "all":
        stages = ["a", "b", "c"]
    else:
        stages = [args.stage]

    # Print header
    print("=" * 60)
    print("DeepRank-Ab Pipeline (Dynamic Resource Allocation)")
    print("=" * 60)
    print(f"Config:    {args.config}")
    print(f"Run root:  {cfg.run_root}")
    print(f"PDB root:  {cfg.pdb_root}")
    print(f"Model:     {cfg.model_path}")
    print(f"Chains:    H={cfg.heavy} L={cfg.light} Ag={cfg.antigen}")
    print(f"Stages:    {' -> '.join(s.upper() for s in stages)}")
    print(f"Input hygiene: {_resolve_input_hygiene(cfg)['mode'].upper()}")
    print(f"Metrics:   {'ON' if cfg.collect_metrics else 'OFF'}")
    if args.dry_run:
        print(f"Mode:      DRY RUN")
    print("=" * 60)

    # Run pipeline
    results = run_pipeline_dynamic(cfg, stages, args.dry_run)

    # Print results
    print("\n" + "=" * 60)
    print("Job Submission Results")
    print("=" * 60)

    all_ok = True
    job_chain = []
    for r in results:
        status = "OK" if r.success else "FAILED"
        if r.success:
            if r.job_id:
                print(f"  {r.stage}: {status} (job {r.job_id})")
                job_chain.append(f"{r.stage}:{r.job_id}")
            else:
                print(f"  {r.stage}: {r.message}")
                if r.command:
                    print(f"    {r.command}")
        else:
            print(f"  {r.stage}: {status}")
            print(f"    {r.message}")
            all_ok = False

    if job_chain and not args.dry_run:
        print(f"\nJob chain: {' -> '.join(job_chain)}")
        print(f"\nMonitor with:")
        print(f"  squeue -u $USER")
        print(f"  {cfg.deeprank_root}/scripts/watch_progress.sh {cfg.run_root}")

    # Save submission info
    if not args.dry_run and job_chain:
        info_file = cfg.run_root / "pipeline_jobs.json"
        ih_cfg = _resolve_input_hygiene(cfg)
        ih_summary_path = cfg.run_root / "input_hygiene_summary.json"
        info = {
            "config": str(args.config),
            "stages": stages,
            "jobs": {r.stage: r.job_id for r in results if r.job_id},
            "input_hygiene_mode": ih_cfg["mode"],
            "input_hygiene_summary": str(ih_summary_path) if ih_summary_path.exists() else None,
            "pipeline_invoked_at": pipeline_invoked_at,
            "jobs_recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nJob info saved to: {info_file}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
