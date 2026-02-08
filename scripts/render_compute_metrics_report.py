#!/usr/bin/env python3
"""
render_compute_metrics_report.py

Render a human-readable HTML report from summarize_compute_metrics aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fmt(x: Optional[float], nd: int = 2, suffix: str = "") -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}{suffix}"


def _fmt_pct(x: Optional[float], nd: int = 1) -> str:
    return _fmt(x, nd=nd, suffix="%")


def _fmt_bytes_mib_to_gib(x_mib: Optional[float], nd: int = 2) -> str:
    if x_mib is None:
        return "—"
    return f"{(x_mib / 1024.0):.{nd}f} GiB"


def _fmt_duration(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_dt(x: Any) -> Optional[datetime]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _as_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "unknown"


def _infer_logical_cpus(prefix: str, metrics_dir: Optional[Path]) -> Optional[int]:
    if metrics_dir is None:
        return None
    p = metrics_dir / f"cpu_percore_{prefix}.csv"
    if not p.is_file():
        return None
    try:
        header = p.open("r").readline().strip().split(",")
    except Exception:
        return None
    if not header:
        return None
    ncpu = sum(1 for h in header if re.fullmatch(r"cpu\d+_pct", h))
    return ncpu if ncpu > 0 else None


def _load_reports(path_or_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    p = path_or_dir.expanduser().resolve()
    if p.is_dir():
        cands = sorted(p.glob("summary_*_ALL.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit(f"ERROR: no summary_*_ALL.json found in {p}")
        src = cands[0]
    else:
        src = p
    if not src.is_file():
        raise SystemExit(f"ERROR: summary JSON not found: {src}")

    obj = json.loads(src.read_text())
    if isinstance(obj, dict):
        reports = [obj]
    elif isinstance(obj, list):
        reports = obj
    else:
        raise SystemExit(f"ERROR: unsupported JSON shape in {src}")
    if not reports:
        raise SystemExit(f"ERROR: empty report list in {src}")
    return reports, src


def _parse_stage_job_task_host(prefix: str) -> Tuple[str, str, str, str]:
    # Expected prefix shape: stageA_<jobid>_<task>_<host>
    stage = "UNKNOWN"
    job = "—"
    task = "—"
    host = "—"

    m_stage = re.match(r"^(stage[A-Za-z0-9]+)_", prefix)
    if m_stage:
        s = m_stage.group(1).lower()
        if s == "stagea":
            stage = "A"
        elif s == "stageb":
            stage = "B"
        elif s == "stagec":
            stage = "C"
        else:
            stage = m_stage.group(1)

    m_job = re.search(r"_(\d{5,})_(\d+)(?:_|$)", prefix)
    if m_job:
        job = m_job.group(1)
        task = m_job.group(2)

    if "_" in prefix:
        host = prefix.rsplit("_", 1)[-1]

    return stage, job, task, host


def _series_row(metric: str, unit: str, series: Optional[Dict[str, Any]]) -> str:
    if not series:
        return (
            "<tr>"
            f"<td>{escape(metric)}</td><td>{escape(unit)}</td>"
            "<td colspan='10'>not collected</td>"
            "</tr>"
        )
    n = _as_float(series.get("n"))
    mean = _as_float(series.get("mean"))
    med = _as_float(series.get("median"))
    p95 = _as_float(series.get("p95"))
    vmax = _as_float(series.get("max"))
    vmin = _as_float(series.get("min"))
    std = _as_float(series.get("std"))
    f0 = _as_float(series.get("frac_gt0"))
    f10 = _as_float(series.get("frac_ge10"))
    f50 = _as_float(series.get("frac_ge50"))
    f80 = _as_float(series.get("frac_ge80"))
    return (
        "<tr>"
        f"<td>{escape(metric)}</td>"
        f"<td>{escape(unit)}</td>"
        f"<td>{_fmt(n, nd=0)}</td>"
        f"<td>{_fmt(mean)}</td>"
        f"<td>{_fmt(med)}</td>"
        f"<td>{_fmt(p95)}</td>"
        f"<td>{_fmt(vmax)}</td>"
        f"<td>{_fmt(vmin)}</td>"
        f"<td>{_fmt(std)}</td>"
        f"<td>{_fmt_pct(None if f0 is None else 100.0 * f0)}</td>"
        f"<td>{_fmt_pct(None if f10 is None else 100.0 * f10)}</td>"
        f"<td>{_fmt_pct(None if f50 is None else 100.0 * f50)}</td>"
        f"<td>{_fmt_pct(None if f80 is None else 100.0 * f80)}</td>"
        "</tr>"
    )


def _bar(value: Optional[float], cap: float = 100.0) -> str:
    if value is None:
        return "<div class='bar'><span style='width:0%'></span></div>"
    pct = max(0.0, min(100.0, 100.0 * value / max(cap, 1e-9)))
    return f"<div class='bar'><span style='width:{pct:.1f}%'></span></div>"


def _top_findings(rep: Dict[str, Any], stage: str) -> List[str]:
    out: List[str] = []
    sys = rep.get("sys")
    gpus = rep.get("gpus") or []

    if sys:
        cpu_mean = _as_float((sys.get("cpu_total") or {}).get("mean"))
        load1_mean = _as_float((sys.get("load1") or {}).get("mean"))
        iow_p95 = _as_float((sys.get("cpu_iowait") or {}).get("p95"))
        disk_w_mean = _as_float((sys.get("disk_w_MBps") or {}).get("mean"))
        disk_w_p95 = _as_float((sys.get("disk_w_MBps") or {}).get("p95"))
        if cpu_mean is not None and load1_mean is not None and cpu_mean < 8.0 and load1_mean > 4.0:
            out.append("single-core/scheduler bottleneck pattern")
        if iow_p95 is not None and iow_p95 >= 10.0:
            out.append("high iowait (storage pressure)")
        if disk_w_mean is not None and disk_w_p95 is not None and disk_w_mean > 0 and (disk_w_p95 / disk_w_mean) >= 5.0:
            out.append("bursty disk writes")

    if stage == "B" and gpus:
        util_means = [_as_float((g.get("util") or {}).get("mean")) for g in gpus]
        util_means = [u for u in util_means if u is not None]
        if util_means and max(util_means) < 20.0:
            out.append("GPU underutilized")

    for n in rep.get("notes") or []:
        s = str(n).strip()
        if not s:
            continue
        # Stage A is CPU-only; hide noisy GPU-missing notes.
        if stage in {"A", "C"} and (
            s.startswith("missing/empty gpu_metrics_")
            or s.startswith("missing/empty gpu_pmon_")
            or "gpu_metrics missing; using pmon-only" in s
        ):
            continue
        out.append(s)
    # Keep report concise.
    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:6]


def render_html(reports: List[Dict[str, Any]], src_path: Path, run_root: str, metrics_dir: Optional[Path]) -> str:
    normalized: List[Dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    total_duration = 0.0
    bottleneck_prefixes = 0

    cpu_means: List[float] = []
    gpu_util_maxes: List[float] = []
    logical_cpu_counts: List[int] = []
    busy_cores_peak: List[float] = []
    gpu_slots_per_prefix: List[int] = []
    active_gpus_per_prefix: List[int] = []
    disk_read_gib_total = 0.0
    disk_write_gib_total = 0.0
    net_rx_gib_total = 0.0
    net_tx_gib_total = 0.0
    disk_total_mean_mbps: List[float] = []
    net_total_mean_mbps: List[float] = []
    disk_write_burst_ratios: List[float] = []
    gpu_duty_samples_pct: List[float] = []
    iowait_p95_samples_pct: List[float] = []
    mem_pressure_samples_pct: List[float] = []
    mem_used_peak_samples_mib: List[float] = []
    gpu_mem_peak_samples_mib: List[float] = []
    pipeline_start: Optional[datetime] = None
    pipeline_end: Optional[datetime] = None

    for rep in reports:
        prefix = str(rep.get("prefix", "unknown"))
        stage, job, task, host = _parse_stage_job_task_host(prefix)
        stage_counts[stage] += 1

        dur = _as_float(rep.get("duration_s")) or 0.0
        total_duration += dur
        rep_start = _parse_dt(rep.get("start"))
        rep_end = _parse_dt(rep.get("end"))
        if rep_start is not None and (pipeline_start is None or rep_start < pipeline_start):
            pipeline_start = rep_start
        if rep_end is not None and (pipeline_end is None or rep_end > pipeline_end):
            pipeline_end = rep_end

        sys = rep.get("sys")
        logical_cpus = _infer_logical_cpus(prefix, metrics_dir)
        if logical_cpus is not None:
            logical_cpu_counts.append(logical_cpus)

        if sys:
            cmean = _as_float((sys.get("cpu_total") or {}).get("mean"))
            if cmean is not None:
                cpu_means.append(cmean)
                if logical_cpus is not None:
                    busy_cores_peak.append((cmean / 100.0) * logical_cpus)

            cp95 = _as_float((sys.get("cpu_total") or {}).get("p95"))
            if cp95 is not None and logical_cpus is not None:
                busy_cores_peak.append((cp95 / 100.0) * logical_cpus)

            iow_p95 = _as_float((sys.get("cpu_iowait") or {}).get("p95"))
            if iow_p95 is not None:
                iowait_p95_samples_pct.append(iow_p95)

            mem_used_max = _as_float((sys.get("mem_used_mib") or {}).get("max"))
            mem_avail_min = _as_float((sys.get("mem_avail_mib") or {}).get("min"))
            if mem_used_max is not None:
                mem_used_peak_samples_mib.append(mem_used_max)
            if (
                mem_used_max is not None
                and mem_avail_min is not None
                and mem_used_max >= 0
                and mem_avail_min >= 0
                and (mem_used_max + mem_avail_min) > 0
            ):
                mem_pressure_samples_pct.append(100.0 * mem_used_max / (mem_used_max + mem_avail_min))

            dwr_mean = _as_float((sys.get("disk_w_MBps") or {}).get("mean"))
            dwr_p95 = _as_float((sys.get("disk_w_MBps") or {}).get("p95"))
            drd_mean = _as_float((sys.get("disk_r_MBps") or {}).get("mean"))
            if dwr_mean is not None or drd_mean is not None:
                disk_total_mean_mbps.append((dwr_mean or 0.0) + (drd_mean or 0.0))
            if dwr_mean is not None and dwr_p95 is not None and dwr_mean > 0:
                disk_write_burst_ratios.append(dwr_p95 / dwr_mean)
            dur_s = _as_float(rep.get("duration_s")) or 0.0
            if dwr_mean is not None and dur_s > 0:
                disk_write_gib_total += (dwr_mean * dur_s) / 1024.0
            if drd_mean is not None and dur_s > 0:
                disk_read_gib_total += (drd_mean * dur_s) / 1024.0

            nrx_mean = _as_float(((sys.get("net_rx_MBps") or {}) if sys.get("net_rx_MBps") else {}).get("mean"))
            ntx_mean = _as_float(((sys.get("net_tx_MBps") or {}) if sys.get("net_tx_MBps") else {}).get("mean"))
            if nrx_mean is not None or ntx_mean is not None:
                net_total_mean_mbps.append((nrx_mean or 0.0) + (ntx_mean or 0.0))
            if dur_s > 0:
                if nrx_mean is not None:
                    net_rx_gib_total += (nrx_mean * dur_s) / 1024.0
                if ntx_mean is not None:
                    net_tx_gib_total += (ntx_mean * dur_s) / 1024.0

        gmax = 0.0
        g_active = 0
        for g in rep.get("gpus") or []:
            umax = _as_float((g.get("util") or {}).get("max"))
            if umax is not None:
                gmax = max(gmax, umax)
            if bool(g.get("active")):
                g_active += 1
            gmem_max = _as_float((g.get("mem_mib") or {}).get("max"))
            if gmem_max is not None:
                gpu_mem_peak_samples_mib.append(gmem_max)
            if stage == "B":
                f50 = _as_float((g.get("util") or {}).get("frac_ge50"))
                if f50 is not None:
                    gpu_duty_samples_pct.append(100.0 * f50)
        gpu_util_maxes.append(gmax)
        gpu_slots_per_prefix.append(len(rep.get("gpus") or []))
        active_gpus_per_prefix.append(g_active)

        findings = _top_findings(rep, stage)
        if findings:
            bottleneck_prefixes += 1

        job_num = _as_int_or_none(job)
        task_num = _as_int_or_none(task)
        rep["_meta"] = {
            "stage": stage,
            "job": job,
            "task": task,
            "host": host,
            "logical_cpus": logical_cpus,
            "job_num": job_num,
            "task_num": task_num,
        }
        rep["_findings"] = findings
        normalized.append(rep)

    normalized.sort(
        key=lambda r: (
            r["_meta"]["stage"],
            r["_meta"].get("job_num") is None,
            r["_meta"].get("job_num") if r["_meta"].get("job_num") is not None else 10**18,
            str(r["_meta"].get("job", "")),
            r["_meta"].get("task_num") is None,
            r["_meta"].get("task_num") if r["_meta"].get("task_num") is not None else 10**18,
            str(r["_meta"].get("task", "")),
            str(r["_meta"].get("host", "")),
        )
    )
    stage_groups: Dict[str, List[Dict[str, Any]]] = {}
    for rep in normalized:
        stg = str((rep.get("_meta") or {}).get("stage", "UNKNOWN"))
        if stg not in stage_groups:
            stage_groups[stg] = []
        stage_groups[stg].append(rep)
    preferred_stage_order = ["A", "B", "C"]
    stage_sequence = [s for s in preferred_stage_order if s in stage_groups]
    stage_sequence.extend(sorted([s for s in stage_groups.keys() if s not in preferred_stage_order]))

    kpi_total = len(normalized)
    kpi_cpu_mean = sum(cpu_means) / len(cpu_means) if cpu_means else None
    kpi_gpu_peak = max(gpu_util_maxes) if gpu_util_maxes else None
    kpi_logical_cpus = max(logical_cpu_counts) if logical_cpu_counts else None
    kpi_busy_cores_peak = max(busy_cores_peak) if busy_cores_peak else None
    kpi_gpu_slots = max(gpu_slots_per_prefix) if gpu_slots_per_prefix else 0
    kpi_gpu_active_peak = max(active_gpus_per_prefix) if active_gpus_per_prefix else 0
    kpi_disk_total_mean = sum(disk_total_mean_mbps) / len(disk_total_mean_mbps) if disk_total_mean_mbps else None
    kpi_net_total_mean = sum(net_total_mean_mbps) / len(net_total_mean_mbps) if net_total_mean_mbps else None
    kpi_gpu_duty_cycle = sum(gpu_duty_samples_pct) / len(gpu_duty_samples_pct) if gpu_duty_samples_pct else None
    kpi_bottleneck_prefixes = (100.0 * bottleneck_prefixes / kpi_total) if kpi_total > 0 else None
    kpi_disk_burstiness = None
    if disk_write_burst_ratios:
        s = sorted(disk_write_burst_ratios)
        n = len(s)
        mid = n // 2
        kpi_disk_burstiness = s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    pipeline_wall_s = None
    if pipeline_start is not None and pipeline_end is not None and pipeline_end >= pipeline_start:
        pipeline_wall_s = (pipeline_end - pipeline_start).total_seconds()
    kpi_effective_parallelism = None
    if pipeline_wall_s is not None and pipeline_wall_s > 0:
        kpi_effective_parallelism = total_duration / pipeline_wall_s
    kpi_busy_core_saturation = None
    if kpi_busy_cores_peak is not None and kpi_logical_cpus is not None and kpi_logical_cpus > 0:
        kpi_busy_core_saturation = 100.0 * (kpi_busy_cores_peak / kpi_logical_cpus)
    kpi_idle_overhead_pct = None
    if kpi_busy_core_saturation is not None:
        kpi_idle_overhead_pct = max(0.0, 100.0 - min(100.0, kpi_busy_core_saturation))
    kpi_iowait_p95_peak = max(iowait_p95_samples_pct) if iowait_p95_samples_pct else None
    kpi_mem_pressure_peak = max(mem_pressure_samples_pct) if mem_pressure_samples_pct else None
    kpi_mem_used_peak_mib = max(mem_used_peak_samples_mib) if mem_used_peak_samples_mib else None
    kpi_gpu_mem_peak_mib = max(gpu_mem_peak_samples_mib) if gpu_mem_peak_samples_mib else None
    kpi_gpu_mem_peak_gib = None if kpi_gpu_mem_peak_mib is None else (kpi_gpu_mem_peak_mib / 1024.0)
    if kpi_gpu_mem_peak_gib is None:
        gpu_mem_cap_gib = 48.0
    elif kpi_gpu_mem_peak_gib <= 48.0:
        gpu_mem_cap_gib = 48.0
    elif kpi_gpu_mem_peak_gib <= 80.0:
        gpu_mem_cap_gib = 80.0
    else:
        gpu_mem_cap_gib = max(100.0, kpi_gpu_mem_peak_gib * 1.1)

    css = """
    :root{
      --bg:#f4f7fb; --card:#ffffff; --ink:#10213a; --muted:#4c5d78;
      --line:#d8e0ee; --accent:#0d6efd; --ok:#1f9d55; --warn:#d9822b; --bad:#c0392b;
      --chip:#eef3ff;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    .wrap{max-width:1520px;margin:24px auto;padding:0 20px}
    h1{margin:0 0 8px;font-size:30px}
    h2{margin:24px 0 10px;font-size:20px}
    h3{margin:0;font-size:17px}
    p.meta{margin:0 0 18px;color:var(--muted)}
    .top-grid{display:flex;flex-direction:column;gap:12px}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
    .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
    .kpi .label{color:var(--muted);font-size:11px}
    .kpi .val{font-size:21px;font-weight:700;line-height:1.2;margin-top:3px;word-break:break-word}
    .kpi .sub{font-size:11px;color:var(--muted);line-height:1.3}
    .util-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
    .util-title{font-size:18px;font-weight:700;margin:0 0 4px}
    .util-meta{font-size:11px;color:var(--muted);margin:0 0 8px;overflow-wrap:anywhere}
    .util-row{margin-top:7px}
    .util-head{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:var(--muted)}
    .util-head .v{color:var(--ink);font-weight:700}
    .stage-nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .stage-link{display:inline-block;background:var(--chip);border:1px solid #cfe0ff;border-radius:999px;padding:5px 11px;font-weight:700;font-size:12px;color:#1d4ed8;text-decoration:none}
    .stage-link:hover{background:#e4edff}
    .stage-block{margin-top:14px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 12px}
    .stage-block[open]{padding-bottom:12px}
    .stage-block>summary{list-style:none;cursor:pointer;color:var(--ink);font-weight:700;display:flex;justify-content:space-between;align-items:center;padding:4px 0}
    .stage-block>summary::-webkit-details-marker{display:none}
    .stage-meta{color:var(--muted);font-size:12px;font-weight:600}
    .prefix{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:14px}
    .head{display:flex;justify-content:flex-start;gap:10px;align-items:flex-start}
    .badges{display:flex;gap:8px;flex-wrap:wrap}
    .badge{padding:3px 9px;border-radius:999px;font-weight:700;font-size:12px;border:1px solid}
    .bA{background:#e9f2ff;color:#1d4ed8;border-color:#bfd7ff}
    .bB{background:#ecfff3;color:#15803d;border-color:#c7f0d7}
    .bC{background:#fff7e8;color:#b45309;border-color:#f4ddaf}
    .bX{background:#f2f4f7;color:#344054;border-color:#d0d5dd}
    .grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px;margin-top:10px}
    .mini{border:1px solid var(--line);border-radius:10px;padding:9px;background:#fcfdff}
    .mini .k{font-size:12px;color:var(--muted)}
    .mini .v{font-size:17px;font-weight:700;margin-top:2px}
    .bar{margin-top:6px;height:7px;background:#ebf0fa;border-radius:999px;overflow:hidden}
    .bar>span{display:block;height:100%;background:linear-gradient(90deg,#66a4ff,#0d6efd)}
    details{margin-top:12px}
    summary{cursor:pointer;color:#1d4ed8;font-weight:600}
    .table-wrap{overflow:auto;margin-top:8px}
    table{border-collapse:collapse;width:100%;font-size:12px}
    th,td{border:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}
    th:first-child,td:first-child{text-align:left}
    th:nth-child(2),td:nth-child(2){text-align:left}
    thead th{background:#f7faff}
    .foot{margin:30px 0 10px;color:var(--muted);font-size:12px}
    @media (max-width:1200px){.kpis{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}.grid{grid-template-columns:repeat(3,minmax(200px,1fr))}}
    @media (max-width:760px){.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}.grid{grid-template-columns:1fr}}
    """

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Compute Metrics Report</title>")
    parts.append(f"<style>{css}</style></head><body><div class='wrap'>")
    parts.append("<h1>Compute Metrics Report</h1>")
    parts.append(
        f"<p class='meta'>Generated {escape(datetime.now().isoformat(timespec='seconds'))} | "
        f"Source: <code>{escape(str(src_path))}</code> | Run root: <code>{escape(run_root)}</code></p>"
    )

    # Top summary and utilization snapshot
    parts.append("<div class='top-grid'>")
    parts.append("<div class='kpis'>")
    parts.append(f"<div class='kpi'><div class='label'>Pipeline Walltime</div><div class='val'>{escape(_fmt_duration(pipeline_wall_s))}</div><div class='sub'>from earliest metric start to latest metric end</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>Summed Observed Walltime</div><div class='val'>{escape(_fmt_duration(total_duration))}</div><div class='sub'>sum of per-prefix durations</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>Effective Parallelism</div><div class='val'>{escape(_fmt(kpi_effective_parallelism, nd=2, suffix='x'))}</div><div class='sub'>summed observed walltime / pipeline walltime</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>Bottlenecked Prefixes</div><div class='val'>{escape(_fmt_pct(kpi_bottleneck_prefixes))}</div><div class='sub'>{bottleneck_prefixes} of {kpi_total} prefixes flagged</div></div>")
    parts.append(
        f"<div class='kpi'><div class='label'>CPU Capacity / Peak Busy</div>"
        f"<div class='val'>{escape(_fmt(kpi_logical_cpus, nd=0))} / {escape(_fmt(kpi_busy_cores_peak, nd=2))}</div>"
        "<div class='sub'>logical CPUs observed / estimated peak busy cores</div></div>"
    )
    parts.append(f"<div class='kpi'><div class='label'>CPU Mean (avg prefixes)</div><div class='val'>{escape(_fmt_pct(kpi_cpu_mean))}</div><div class='sub'>node-level cpu_total_pct</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>GPU Slots / Active Peak</div><div class='val'>{kpi_gpu_slots} / {kpi_gpu_active_peak}</div><div class='sub'>observed slots / max active GPUs</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>Peak GPU Util</div><div class='val'>{escape(_fmt_pct(kpi_gpu_peak))}</div><div class='sub'>max util_gpu_pct observed</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>GPU Duty Cycle (Stage B)</div><div class='val'>{escape(_fmt_pct(kpi_gpu_duty_cycle))}</div><div class='sub'>mean fraction of time with util >= 50%</div></div>")
    parts.append(f"<div class='kpi'><div class='label'>Disk Burstiness (P95/Mean)</div><div class='val'>{escape(_fmt(kpi_disk_burstiness, nd=2, suffix='x'))}</div><div class='sub'>median per-prefix write burst ratio</div></div>")
    parts.append(
        f"<div class='kpi'><div class='label'>Estimated Data Generated / Read</div>"
        f"<div class='val'>W {escape(_fmt(disk_write_gib_total, nd=2, suffix=' GiB'))} / R {escape(_fmt(disk_read_gib_total, nd=2, suffix=' GiB'))}</div>"
        "<div class='sub'>disk write/read volumes from mean MB/s x duration</div></div>"
    )
    parts.append(
        f"<div class='kpi'><div class='label'>Estimated Copied To / From</div>"
        f"<div class='val'>TX {escape(_fmt(net_tx_gib_total, nd=2, suffix=' GiB'))} / RX {escape(_fmt(net_rx_gib_total, nd=2, suffix=' GiB'))}</div>"
        "<div class='sub'>network tx/rx volumes from mean MB/s x duration</div></div>"
    )
    parts.append("</div>")
    parts.append("<div class='util-card'>")
    parts.append("<div class='util-title'>Utilization Snapshot</div>")
    parts.append(
        f"<div class='util-meta'>Pipeline start {escape(_fmt_dt(pipeline_start))} | "
        f"end {escape(_fmt_dt(pipeline_end))} | wall {escape(_fmt_duration(pipeline_wall_s))}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>CPU mean utilization</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_cpu_mean))}</span></div>{_bar(kpi_cpu_mean, cap=100.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>CPU saturation proxy (busy/logical)</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_busy_core_saturation))}</span></div>{_bar(kpi_busy_core_saturation, cap=100.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>CPU iowait p95 (peak)</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_iowait_p95_peak))}</span></div>{_bar(kpi_iowait_p95_peak, cap=30.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>Memory pressure proxy (peak)</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_mem_pressure_peak))} | used max {escape(_fmt_bytes_mib_to_gib(kpi_mem_used_peak_mib))}</span></div>{_bar(kpi_mem_pressure_peak, cap=100.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>GPU peak utilization</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_gpu_peak))}</span></div>{_bar(kpi_gpu_peak, cap=100.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>GPU memory peak</span>"
        f"<span class='v'>{escape(_fmt(kpi_gpu_mem_peak_gib, nd=2, suffix=' GiB'))}</span></div>{_bar(kpi_gpu_mem_peak_gib, cap=gpu_mem_cap_gib)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>Disk throughput mean (R+W)</span>"
        f"<span class='v'>{escape(_fmt(kpi_disk_total_mean, nd=2, suffix=' MB/s'))}</span></div>{_bar(kpi_disk_total_mean, cap=250.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>Network throughput mean (TX+RX)</span>"
        f"<span class='v'>{escape(_fmt(kpi_net_total_mean, nd=2, suffix=' MB/s'))}</span></div>{_bar(kpi_net_total_mean, cap=25.0)}</div>"
    )
    parts.append(
        "<div class='util-row'><div class='util-head'><span>Queue/idle overhead proxy</span>"
        f"<span class='v'>{escape(_fmt_pct(kpi_idle_overhead_pct))}</span></div>{_bar(kpi_idle_overhead_pct, cap=100.0)}</div>"
    )
    parts.append("</div>")
    parts.append("</div>")

    parts.append("<div class='stage-nav'>")
    for stg in stage_sequence:
        sid = _safe_slug(stg)
        parts.append(f"<a class='stage-link' href='#stage-{sid}'>Stage {escape(stg)} ({len(stage_groups.get(stg, []))})</a>")
    parts.append("</div>")

    for i, stg in enumerate(stage_sequence):
        sid = _safe_slug(stg)
        stage_reps = stage_groups.get(stg, [])
        open_attr = " open" if i == 0 else ""
        parts.append(f"<details class='stage-block'{open_attr}>")
        parts.append(
            f"<summary id='stage-{sid}'>"
            f"<span>Stage {escape(stg)}</span>"
            f"<span class='stage-meta'>{len(stage_reps)} panels</span>"
            "</summary>"
        )

        for rep in stage_reps:
            meta = rep.get("_meta", {})
            stage = meta.get("stage", "UNKNOWN")
            job = meta.get("job", "—")
            task = meta.get("task", "—")
            host = meta.get("host", "—")
            logical_cpus = _as_float(meta.get("logical_cpus"))
            badge_cls = "bA" if stage == "A" else ("bB" if stage == "B" else ("bC" if stage == "C" else "bX"))

            sys = rep.get("sys")
            gpus = rep.get("gpus") or []
            proc = rep.get("proc")

            cpu_mean = _as_float(((sys or {}).get("cpu_total") or {}).get("mean"))
            cpu_p95 = _as_float(((sys or {}).get("cpu_total") or {}).get("p95"))
            busy_cores_mean = None if (cpu_mean is None or logical_cpus is None) else (cpu_mean / 100.0) * logical_cpus
            busy_cores_p95 = None if (cpu_p95 is None or logical_cpus is None) else (cpu_p95 / 100.0) * logical_cpus
            mem_max = _as_float(((sys or {}).get("mem_used_mib") or {}).get("max"))
            disk_w_p95 = _as_float(((sys or {}).get("disk_w_MBps") or {}).get("p95"))
            net_tx_p95 = _as_float((((sys or {}).get("net_tx_MBps") or {}) if sys and (sys.get("net_tx_MBps") is not None) else {}).get("p95"))

            gpu_util_peak = 0.0
            active_gpu_count = 0
            for g in gpus:
                if bool(g.get("active")):
                    active_gpu_count += 1
                umax = _as_float((g.get("util") or {}).get("max"))
                if umax is not None:
                    gpu_util_peak = max(gpu_util_peak, umax)

            parts.append("<div class='prefix'>")
            parts.append("<div class='head'>")
            parts.append("<div class='badges'>")
            parts.append(f"<span class='badge {badge_cls}'>Stage {escape(stage)}</span>")
            parts.append(f"<span class='badge bX'>Job {escape(str(job))}_{escape(str(task))}</span>")
            parts.append(f"<span class='badge bX'>{escape(str(host))}</span>")
            parts.append(f"<span class='badge bX'>Duration {escape(_fmt_duration(_as_float(rep.get('duration_s'))))}</span>")
            parts.append("</div></div>")

            parts.append("<div class='grid'>")
            parts.append(
                "<div class='mini'><div class='k'>CPU total mean / p95</div>"
                f"<div class='v'>{escape(_fmt_pct(cpu_mean))} / {escape(_fmt_pct(cpu_p95))}</div>"
                f"{_bar(cpu_mean, cap=100.0)}</div>"
            )
            parts.append(
                "<div class='mini'><div class='k'>Busy cores est (mean / p95)</div>"
                f"<div class='v'>{escape(_fmt(busy_cores_mean))} / {escape(_fmt(busy_cores_p95))}</div>"
                f"{_bar(busy_cores_p95, cap=max(1.0, logical_cpus or 1.0))}</div>"
            )
            if stage != "A":
                parts.append(
                    "<div class='mini'><div class='k'>GPU active / peak util</div>"
                    f"<div class='v'>{active_gpu_count} / {escape(_fmt_pct(gpu_util_peak))}</div>"
                    f"{_bar(gpu_util_peak, cap=100.0)}</div>"
                )
            parts.append(
                "<div class='mini'><div class='k'>Disk write p95</div>"
                f"<div class='v'>{escape(_fmt(disk_w_p95, nd=2, suffix=' MB/s'))}</div>"
                f"{_bar(disk_w_p95, cap=max(50.0, (disk_w_p95 or 0.0) * 1.2))}</div>"
            )
            parts.append(
                "<div class='mini'><div class='k'>Net tx p95 / Mem used max</div>"
                f"<div class='v'>{escape(_fmt(net_tx_p95, nd=2, suffix=' MB/s'))} / {escape(_fmt_bytes_mib_to_gib(mem_max))}</div>"
                f"{_bar(net_tx_p95, cap=max(10.0, (net_tx_p95 or 0.0) * 1.2))}</div>"
            )
            parts.append("</div>")

            parts.append("<details><summary>Full metrics (CPU, GPU, disk, network, process)</summary>")
            parts.append("<div class='table-wrap'><table><thead><tr>")
            parts.append(
                "<th>Metric</th><th>Unit</th><th>N</th><th>Mean</th><th>Median</th><th>P95</th><th>Max</th><th>Min</th><th>Std</th>"
                "<th>Frac&gt;0</th><th>Frac&gt;=10</th><th>Frac&gt;=50</th><th>Frac&gt;=80</th>"
            )
            parts.append("</tr></thead><tbody>")

            if sys:
                parts.append(_series_row("cpu_total", "%", (sys.get("cpu_total") or None)))
                parts.append(_series_row("cpu_iowait", "%", (sys.get("cpu_iowait") or None)))
                parts.append(_series_row("load1", "load", (sys.get("load1") or None)))
                parts.append(_series_row("mem_used", "MiB", (sys.get("mem_used_mib") or None)))
                parts.append(_series_row("mem_avail", "MiB", (sys.get("mem_avail_mib") or None)))
                parts.append(_series_row("disk_read", "MB/s", (sys.get("disk_r_MBps") or None)))
                parts.append(_series_row("disk_write", "MB/s", (sys.get("disk_w_MBps") or None)))
                parts.append(_series_row("disk_read_iops", "IOPS", (sys.get("disk_r_iops") or None)))
                parts.append(_series_row("disk_write_iops", "IOPS", (sys.get("disk_w_iops") or None)))
                parts.append(_series_row("net_rx", "MB/s", (sys.get("net_rx_MBps") or None)))
                parts.append(_series_row("net_tx", "MB/s", (sys.get("net_tx_MBps") or None)))

            if proc:
                parts.append(_series_row("proc_cpu", "%", (proc.get("proc_cpu") or None)))
                parts.append(_series_row("proc_rss", "MiB", (proc.get("rss_mib") or None)))
                parts.append(_series_row("proc_vms", "MiB", (proc.get("vms_mib") or None)))
                parts.append(_series_row("proc_nprocs", "count", (proc.get("nprocs") or None)))

            for g in gpus:
                gidx = g.get("gpu")
                parts.append(_series_row(f"gpu{gidx}_util", "%", (g.get("util") or None)))
                parts.append(_series_row(f"gpu{gidx}_mem_used", "MiB", (g.get("mem_mib") or None)))
                parts.append(_series_row(f"gpu{gidx}_power", "W", (g.get("power_w") or None)))
                parts.append(_series_row(f"gpu{gidx}_temp", "C", (g.get("temp_c") or None)))
                parts.append(_series_row(f"gpu{gidx}_clock_sm", "MHz", (g.get("clk_sm_mhz") or None)))

            parts.append("</tbody></table></div>")

            pmon = rep.get("pmon")
            if pmon:
                parts.append("<div class='table-wrap'><table><thead><tr><th>PMON GPU</th><th>SM mean</th><th>SM max</th><th>Frac SM&gt;0</th><th>Unique PIDs</th></tr></thead><tbody>")
                for p in pmon:
                    parts.append(
                        "<tr>"
                        f"<td>GPU{escape(str(p.get('gpu')))}</td>"
                        f"<td>{escape(_fmt(_as_float(p.get('sm_mean'))))}%</td>"
                        f"<td>{escape(_fmt(_as_float(p.get('sm_max'))))}%</td>"
                        f"<td>{escape(_fmt_pct(None if _as_float(p.get('frac_sm_gt0')) is None else 100.0 * _as_float(p.get('frac_sm_gt0'))))}</td>"
                        f"<td>{escape(_fmt(_as_float(p.get('unique_pids')), nd=0))}</td>"
                        "</tr>"
                    )
                parts.append("</tbody></table></div>")

            parts.append("</details>")
            parts.append("</div>")

        parts.append("</details>")

    parts.append(
        "<div class='foot'>"
        "Report generated by <code>scripts/render_compute_metrics_report.py</code>. "
        "Metrics source: summarize_compute_metrics aggregate JSON."
        "</div>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to summary_*_ALL.json or compute_metrics directory.")
    ap.add_argument("--out", default="", help="Output HTML path (default: compute_metrics_report.html near run root)")
    ap.add_argument("--run-root", default="", help="Run root path shown in report header.")
    args = ap.parse_args()

    reports, src = _load_reports(Path(args.input))

    run_root = args.run_root.strip()
    if not run_root:
        # Heuristic: metrics dir parent when input is summary json in compute_metrics/
        if src.parent.name == "compute_metrics":
            run_root = str(src.parent.parent)
        else:
            run_root = str(src.parent)

    metrics_dir: Optional[Path] = None
    if src.parent.name == "compute_metrics":
        metrics_dir = src.parent.resolve()
    else:
        candidate = Path(run_root) / "compute_metrics"
        if candidate.is_dir():
            metrics_dir = candidate.resolve()

    if args.out:
        out_html = Path(args.out).expanduser().resolve()
    else:
        if src.parent.name == "compute_metrics":
            out_html = (src.parent.parent / "compute_metrics_report.html").resolve()
        else:
            out_html = (src.parent / f"compute_metrics_report_{_safe_slug(src.stem)}.html").resolve()

    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(reports=reports, src_path=src, run_root=run_root, metrics_dir=metrics_dir)
    out_html.write_text(html)
    print(f"Wrote HTML report: {out_html}")


if __name__ == "__main__":
    main()
