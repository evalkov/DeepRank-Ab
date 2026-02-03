#!/usr/bin/env python3
"""
summarize_compute_metrics.py

Summarize compute_metrics collected by collect_compute_metrics.py.

Supports directories that may NOT contain GPU CSV logs. If GPU logs exist,
they are processed; otherwise, CPU/system summaries still work.

Inputs (per prefix) from <dir>:
  gpu_metrics_<prefix>.csv        (optional)
  gpu_pmon_<prefix>.log           (optional)
  sys_metrics_<prefix>.csv        (optional but recommended)
  disk_io_<prefix>.csv            (optional but recommended)
  net_io_<prefix>.csv             (optional)
  proc_metrics_<prefix>.csv       (optional)

Always writes, per prefix:
  summary_<jobid>_<n>.tsv
  summary_<jobid>_<n>.json

where <n> is the number immediately after the jobid in the prefix, e.g.
  stageA_50879909_0_fsitgl-hpc047p  -> summary_50879909_0.*
Stdlib-only (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Generic helpers
# -----------------------------

def parse_time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def mean(v: List[float]) -> float:
    return statistics.fmean(v) if v else 0.0


def median(v: List[float]) -> float:
    return statistics.median(v) if v else 0.0


def quantile(v: List[float], q: float) -> float:
    if not v:
        return 0.0
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def pstdev(v: List[float]) -> float:
    return statistics.pstdev(v) if len(v) > 1 else 0.0


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def median_sampling(times: List[datetime]) -> float:
    if len(times) < 3:
        return 0.0
    ds = []
    for i in range(1, len(times)):
        dt_s = (times[i] - times[i - 1]).total_seconds()
        if dt_s > 0:
            ds.append(dt_s)
    return median(ds)


def time_range_from_csv(path: str, ts_field: str = "timestamp") -> Tuple[str, str, float]:
    """
    Return (start_iso, end_iso, duration_s) from a CSV with a timestamp column.
    If missing/unparseable, returns ("","",0.0).
    """
    if not path or not os.path.exists(path):
        return "", "", 0.0
    try:
        rows = read_csv(path)
    except Exception:
        return "", "", 0.0
    if not rows:
        return "", "", 0.0

    times: List[datetime] = []
    for r in rows:
        t = parse_time(r.get(ts_field, ""))
        if t:
            times.append(t)

    if not times:
        return "", "", 0.0

    times.sort()
    start = times[0].isoformat(timespec="milliseconds")
    end = times[-1].isoformat(timespec="milliseconds")
    dur = (times[-1] - times[0]).total_seconds() if len(times) >= 2 else 0.0
    return start, end, dur


# -----------------------------
# Stats structures
# -----------------------------

@dataclass
class SeriesStats:
    n: int
    mean: float
    median: float
    p95: float
    max: float
    min: float
    std: float
    frac_gt0: float
    frac_ge10: float
    frac_ge50: float
    frac_ge80: float


def series_stats(v: List[float]) -> SeriesStats:
    n = len(v)
    if n == 0:
        return SeriesStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    gt0 = sum(1 for x in v if x > 0)
    ge10 = sum(1 for x in v if x >= 10)
    ge50 = sum(1 for x in v if x >= 50)
    ge80 = sum(1 for x in v if x >= 80)
    return SeriesStats(
        n=n,
        mean=mean(v),
        median=median(v),
        p95=quantile(v, 0.95),
        max=max(v),
        min=min(v),
        std=pstdev(v),
        frac_gt0=gt0 / n,
        frac_ge10=ge10 / n,
        frac_ge50=ge50 / n,
        frac_ge80=ge80 / n,
    )


@dataclass
class GPUReport:
    gpu: int
    util: SeriesStats
    mem_mib: SeriesStats
    power_w: SeriesStats
    temp_c: SeriesStats
    clk_sm_mhz: SeriesStats
    active: bool


@dataclass
class SysReport:
    duration_s: float
    median_sampling_s: float
    cpu_total: SeriesStats
    cpu_iowait: SeriesStats
    load1: SeriesStats
    mem_used_mib: SeriesStats
    mem_avail_mib: SeriesStats
    disk_r_MBps: SeriesStats
    disk_w_MBps: SeriesStats
    disk_r_iops: SeriesStats
    disk_w_iops: SeriesStats
    net_rx_MBps: Optional[SeriesStats]
    net_tx_MBps: Optional[SeriesStats]


@dataclass
class ProcReport:
    pid: int
    proc_cpu: SeriesStats
    rss_mib: SeriesStats
    vms_mib: SeriesStats


@dataclass
class PmonGPU:
    gpu: int
    sm_mean: float
    sm_max: float
    frac_sm_gt0: float
    unique_pids: int


@dataclass
class PrefixReport:
    prefix: str
    start: str
    end: str
    duration_s: float
    gpus: List[GPUReport]
    sys: Optional[SysReport]
    proc: Optional[ProcReport]
    pmon: Optional[List[PmonGPU]]
    notes: List[str]


# -----------------------------
# Prefix discovery & naming
# -----------------------------

_PREFIX_SPECS = [
    ("gpu_metrics_", ".csv"),
    ("gpu_pmon_", ".log"),
    ("sys_metrics_", ".csv"),
    ("disk_io_", ".csv"),
    ("net_io_", ".csv"),
    ("proc_metrics_", ".csv"),
]


def list_prefixes(d: str) -> List[str]:
    """
    Return prefixes found by scanning ANY supported metric filename pattern,
    not just gpu_metrics_*.csv.
    """
    prefixes: set[str] = set()
    for stem, ext in _PREFIX_SPECS:
        pat = os.path.join(d, f"{stem}*{ext}")
        for p in glob.glob(pat):
            b = os.path.basename(p)
            if not (b.startswith(stem) and b.endswith(ext)):
                continue
            pfx = b[len(stem) : -len(ext)]
            if pfx:
                prefixes.add(pfx)
    return sorted(prefixes)


def infer_expected_gpus_from_pmon(pmon_path: str) -> int:
    """
    If pmon exists, infer number of GPUs from the max observed gpu index.
    Returns 0 if no gpu indices found.
    """
    if not os.path.exists(pmon_path):
        return 0
    date_re = re.compile(r"^\d{8}$")
    time_re = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    max_gpu = -1
    try:
        with open(pmon_path, "r", errors="replace") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                if len(toks) < 4:
                    continue
                if not (date_re.match(toks[0]) and time_re.match(toks[1])):
                    continue
                try:
                    g = int(toks[2])
                except Exception:
                    continue
                if g > max_gpu:
                    max_gpu = g
    except Exception:
        return 0
    return (max_gpu + 1) if max_gpu >= 0 else 0


def parse_jobid_and_index(prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract jobid and the number immediately after jobid, e.g.
      stageA_50879909_0_fsitgl-hpc047p -> ("50879909", "0")
    """
    m = re.search(r"_(\d{5,})_(\d+)(?:_|$)", prefix)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


# -----------------------------
# Summarizers
# -----------------------------

def summarize_gpu(path: str, expected_gpus: int) -> Tuple[str, str, float, List[GPUReport], List[str]]:
    notes: List[str] = []
    rows = read_csv(path) if (path and os.path.exists(path)) else []

    if not rows:
        # If we were asked for 0 GPUs, return empty list cleanly.
        if expected_gpus <= 0:
            return ("", "", 0.0, [], [f"missing/empty {os.path.basename(path)}"] if path else ["no gpu_metrics collected"])
        return (
            "",
            "",
            0.0,
            [
                GPUReport(
                    g,
                    series_stats([]),
                    series_stats([]),
                    series_stats([]),
                    series_stats([]),
                    series_stats([]),
                    False,
                )
                for g in range(expected_gpus)
            ],
            [f"missing/empty {os.path.basename(path)}"],
        )

    times: List[datetime] = []
    per: Dict[int, Dict[str, List[float]]] = {}

    for r in rows:
        t = parse_time(r.get("timestamp", ""))
        if t:
            times.append(t)
        g = int(float(r.get("gpu", "0") or "0"))
        per.setdefault(g, {"u": [], "m": [], "p": [], "t": [], "c": []})
        per[g]["u"].append(safe_float(r.get("util_gpu_pct", 0)))
        per[g]["m"].append(safe_float(r.get("mem_used_mib", 0)))
        per[g]["p"].append(safe_float(r.get("power_w", 0)))
        per[g]["t"].append(safe_float(r.get("temp_c", 0)))
        per[g]["c"].append(safe_float(r.get("clock_sm_mhz", 0)))

    times.sort()
    start = times[0].isoformat(timespec="milliseconds") if times else ""
    end = times[-1].isoformat(timespec="milliseconds") if times else ""
    dur = (times[-1] - times[0]).total_seconds() if len(times) >= 2 else 0.0

    # If expected_gpus was 0 but data exists, infer range from file.
    if expected_gpus <= 0:
        expected_gpus = (max(per.keys()) + 1) if per else 0

    out: List[GPUReport] = []
    for g in range(expected_gpus):
        dct = per.get(g, {"u": [], "m": [], "p": [], "t": [], "c": []})
        u = series_stats(dct["u"])
        m = series_stats(dct["m"])
        p = series_stats(dct["p"])
        tt = series_stats(dct["t"])
        c = series_stats(dct["c"])
        active = (u.max > 0) or (m.max > 50) or (p.max > 45)
        out.append(GPUReport(g, u, m, p, tt, c, active))

    return start, end, dur, out, notes


def summarize_sys(sys_path: str, disk_path: str, net_path: str) -> Tuple[Optional[SysReport], List[str]]:
    notes: List[str] = []
    if not os.path.exists(sys_path) or not os.path.exists(disk_path):
        if not os.path.exists(sys_path):
            notes.append(f"missing {os.path.basename(sys_path)}")
        if not os.path.exists(disk_path):
            notes.append(f"missing {os.path.basename(disk_path)}")
        return None, notes

    sys_rows = read_csv(sys_path)
    disk_rows = read_csv(disk_path)
    if not sys_rows or not disk_rows:
        if not sys_rows:
            notes.append(f"empty {os.path.basename(sys_path)}")
        if not disk_rows:
            notes.append(f"empty {os.path.basename(disk_path)}")
        return None, notes

    times: List[datetime] = []
    cpu, iow, load1, memu, mema = [], [], [], [], []

    for r in sys_rows:
        t = parse_time(r.get("timestamp", ""))
        if t:
            times.append(t)
        cpu.append(safe_float(r.get("cpu_total_pct", 0)))
        iow.append(safe_float(r.get("cpu_iowait_pct", 0)))
        load1.append(safe_float(r.get("load1", 0)))
        memu.append(safe_float(r.get("mem_used_mib", 0)))
        mema.append(safe_float(r.get("mem_avail_mib", 0)))

    times.sort()
    dur = (times[-1] - times[0]).total_seconds() if len(times) >= 2 else 0.0
    samp = median_sampling(times)

    dr, dw, dri, dwi = [], [], [], []
    for r in disk_rows:
        dr.append(safe_float(r.get("disk_read_MBps", 0)))
        dw.append(safe_float(r.get("disk_write_MBps", 0)))
        dri.append(safe_float(r.get("disk_read_iops", 0)))
        dwi.append(safe_float(r.get("disk_write_iops", 0)))

    net_rx = None
    net_tx = None
    if os.path.exists(net_path):
        net_rows = read_csv(net_path)
        if net_rows:
            rx = [safe_float(r.get("net_rx_MBps", 0)) for r in net_rows]
            tx = [safe_float(r.get("net_tx_MBps", 0)) for r in net_rows]
            net_rx = series_stats(rx)
            net_tx = series_stats(tx)

    return (
        SysReport(
            duration_s=dur,
            median_sampling_s=samp,
            cpu_total=series_stats(cpu),
            cpu_iowait=series_stats(iow),
            load1=series_stats(load1),
            mem_used_mib=series_stats(memu),
            mem_avail_mib=series_stats(mema),
            disk_r_MBps=series_stats(dr),
            disk_w_MBps=series_stats(dw),
            disk_r_iops=series_stats(dri),
            disk_w_iops=series_stats(dwi),
            net_rx_MBps=net_rx,
            net_tx_MBps=net_tx,
        ),
        notes,
    )


def summarize_proc(proc_path: str) -> Optional[ProcReport]:
    if not os.path.exists(proc_path):
        return None
    rows = read_csv(proc_path)
    if not rows:
        return None
    pid = int(float(rows[0].get("pid", "0") or "0"))
    cpu = [safe_float(r.get("proc_cpu_pct", 0)) for r in rows]
    rss = [safe_float(r.get("proc_rss_mib", 0)) for r in rows]
    vms = [safe_float(r.get("proc_vms_mib", 0)) for r in rows]
    return ProcReport(pid=pid, proc_cpu=series_stats(cpu), rss_mib=series_stats(rss), vms_mib=series_stats(vms))


def summarize_pmon(pmon_path: str, expected_gpus: int) -> Optional[List[PmonGPU]]:
    """
    Parse nvidia-smi pmon -o DT output for your format.

    We ignore lines where pid is '-' (no active process).
    """
    if not os.path.exists(pmon_path):
        return None

    with open(pmon_path, "r", errors="replace") as f:
        lines = f.readlines()

    # If expected_gpus is 0, infer from file.
    if expected_gpus <= 0:
        expected_gpus = infer_expected_gpus_from_pmon(pmon_path)

    if expected_gpus <= 0:
        return None

    sm_vals: Dict[int, List[float]] = {g: [] for g in range(expected_gpus)}
    pid_sets: Dict[int, set] = {g: set() for g in range(expected_gpus)}

    date_re = re.compile(r"^\d{8}$")
    time_re = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < 6:
            continue
        if not (date_re.match(toks[0]) and time_re.match(toks[1])):
            continue

        try:
            g = int(toks[2])
        except Exception:
            continue
        if not (0 <= g < expected_gpus):
            continue

        pid_tok = toks[3]
        if pid_tok in ("-", "--"):
            continue
        try:
            pid = int(pid_tok)
        except Exception:
            continue

        # Column order: date time gpu pid type sm mem enc dec ...
        sm_tok = toks[5] if len(toks) > 5 else "-"
        sm = safe_float(sm_tok) if sm_tok != "-" else 0.0

        sm_vals[g].append(sm)
        pid_sets[g].add(pid)

    out: List[PmonGPU] = []
    for g in range(expected_gpus):
        vals = sm_vals[g]
        if not vals:
            out.append(PmonGPU(gpu=g, sm_mean=0.0, sm_max=0.0, frac_sm_gt0=0.0, unique_pids=0))
            continue
        gt0 = sum(1 for v in vals if v > 0)
        out.append(
            PmonGPU(
                gpu=g,
                sm_mean=mean(vals),
                sm_max=max(vals),
                frac_sm_gt0=gt0 / len(vals),
                unique_pids=len(pid_sets[g]),
            )
        )
    return out


# -----------------------------
# Output / flattening
# -----------------------------

def print_report(r: PrefixReport) -> None:
    print("=" * 88)
    print(f"PREFIX: {r.prefix}")
    print(f"Time:   {r.start}  →  {r.end}   (duration {fmt(r.duration_s,3)} s)")
    print()

    if r.gpus:
        print("GPU device metrics:")
        for g in r.gpus:
            u = g.util
            m = g.mem_mib
            p = g.power_w
            active = "ACTIVE" if g.active else "idle"
            print(
                f"  GPU{g.gpu} [{active}] "
                f"util mean {fmt(u.mean)}% p95 {fmt(u.p95)}% max {fmt(u.max,1)}% (>=10% {fmt(100*u.frac_ge10,1)}%) | "
                f"mem max {fmt(m.max,1)} MiB | pwr mean {fmt(p.mean,1)} W max {fmt(p.max,1)} W | "
                f"temp max {fmt(g.temp_c.max,1)} C"
            )
        act = [g.gpu for g in r.gpus if g.active]
        print(f"  GPUs active (heuristic): {act if act else 'none'}")
        print()
    else:
        print("GPU device metrics: (not collected)")
        print()

    if r.pmon:
        print("GPU per-process (pmon) SM activity:")
        for g in r.pmon:
            print(f"  GPU{g.gpu}: SM max {fmt(g.sm_max,1)}% (SM>0 {fmt(100*g.frac_sm_gt0,1)}%), unique PIDs {g.unique_pids}")
        print()

    if r.sys:
        s = r.sys
        print("CPU / system:")
        print(f"  sampling median: {fmt(s.median_sampling_s,3)} s")
        print(f"  cpu_total mean {fmt(s.cpu_total.mean)}% p95 {fmt(s.cpu_total.p95)}% max {fmt(s.cpu_total.max)}%")
        print(f"  cpu_iowait mean {fmt(s.cpu_iowait.mean)}% p95 {fmt(s.cpu_iowait.p95)}% max {fmt(s.cpu_iowait.max)}%")
        print(f"  load1 mean {fmt(s.load1.mean)} p95 {fmt(s.load1.p95)} max {fmt(s.load1.max)}")
        print(f"  mem_used max {fmt(s.mem_used_mib.max,1)} MiB | mem_avail min {fmt(s.mem_avail_mib.min,1)} MiB")
        print()
        print("Disk I/O (aggregated):")
        print(f"  read  mean {fmt(s.disk_r_MBps.mean)} MB/s p95 {fmt(s.disk_r_MBps.p95)} max {fmt(s.disk_r_MBps.max)} | IOPS max {fmt(s.disk_r_iops.max,1)}")
        print(f"  write mean {fmt(s.disk_w_MBps.mean)} MB/s p95 {fmt(s.disk_w_MBps.p95)} max {fmt(s.disk_w_MBps.max)} | IOPS max {fmt(s.disk_w_iops.max,1)}")
        print()
        if s.net_rx_MBps and s.net_tx_MBps:
            print("Network I/O (aggregated):")
            print(f"  rx max {fmt(s.net_rx_MBps.max)} MB/s | tx max {fmt(s.net_tx_MBps.max)} MB/s")
            print()

    if r.proc:
        p = r.proc
        print("Tracked process (optional):")
        print(f"  pid {p.pid}: cpu mean {fmt(p.proc_cpu.mean)}% max {fmt(p.proc_cpu.max)}% | RSS max {fmt(p.rss_mib.max,1)} MiB")
        print()

    if r.notes:
        print("Notes:")
        for n in r.notes:
            print(f"  - {n}")
        print()


def flatten_tsv(r: PrefixReport) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prefix": r.prefix,
        "start": r.start,
        "end": r.end,
        "duration_s": r.duration_s,
    }

    for g in r.gpus:
        k = f"gpu{g.gpu}"
        out[f"{k}_active"] = int(g.active)
        out[f"{k}_util_mean"] = g.util.mean
        out[f"{k}_util_max"] = g.util.max
        out[f"{k}_mem_max_mib"] = g.mem_mib.max
        out[f"{k}_power_mean_w"] = g.power_w.mean
        out[f"{k}_power_max_w"] = g.power_w.max

    if r.pmon:
        for pg in r.pmon:
            k = f"pmon_gpu{pg.gpu}"
            out[f"{k}_sm_mean"] = pg.sm_mean
            out[f"{k}_sm_max"] = pg.sm_max
            out[f"{k}_frac_sm_gt0"] = pg.frac_sm_gt0
            out[f"{k}_unique_pids"] = pg.unique_pids

    if r.sys:
        s = r.sys
        out["cpu_total_mean"] = s.cpu_total.mean
        out["cpu_total_p95"] = s.cpu_total.p95
        out["cpu_iowait_p95"] = s.cpu_iowait.p95
        out["mem_used_max_mib"] = s.mem_used_mib.max
        out["disk_write_MBps_max"] = s.disk_w_MBps.max
        out["disk_write_iops_max"] = s.disk_w_iops.max
        out["disk_read_MBps_max"] = s.disk_r_MBps.max
        out["disk_read_iops_max"] = s.disk_r_iops.max
        if s.net_rx_MBps and s.net_tx_MBps:
            out["net_rx_MBps_max"] = s.net_rx_MBps.max
            out["net_tx_MBps_max"] = s.net_tx_MBps.max

    return out


def write_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    # Single positional directory argument (requested). Keep --dir for backward compatibility.
    ap.add_argument("dir", nargs="?", help="compute_metrics directory (positional)")
    ap.add_argument("--dir", dest="dir_opt", default="", help="compute_metrics directory (legacy)")
    ap.add_argument("--prefix", default="", help="substring filter (e.g. 50869971_)")
    ap.add_argument("--gpus", type=int, default=4, help="expected GPUs when GPU logs exist (default: 4)")
    args = ap.parse_args()

    d_in = args.dir_opt or args.dir
    if not d_in:
        raise SystemExit("ERROR: provide compute_metrics directory as a single argument")

    d = os.path.realpath(d_in)
    if not os.path.isdir(d):
        raise SystemExit(f"ERROR: not a directory: {d}")

    prefixes = list_prefixes(d)
    if args.prefix:
        prefixes = [p for p in prefixes if args.prefix in p]
    if not prefixes:
        raise SystemExit(
            "ERROR: no metric files found matching any of: "
            + ", ".join([f"{stem}*{ext}" for stem, ext in _PREFIX_SPECS])
        )

    reports: List[PrefixReport] = []

    for pfx in prefixes:
        notes: List[str] = []

        gpu_path = os.path.join(d, f"gpu_metrics_{pfx}.csv")
        pmon_path = os.path.join(d, f"gpu_pmon_{pfx}.log")
        sys_path = os.path.join(d, f"sys_metrics_{pfx}.csv")
        disk_path = os.path.join(d, f"disk_io_{pfx}.csv")
        net_path = os.path.join(d, f"net_io_{pfx}.csv")
        proc_path = os.path.join(d, f"proc_metrics_{pfx}.csv")

        # Decide GPU count for this prefix:
        inferred_pmon_gpus = infer_expected_gpus_from_pmon(pmon_path)
        exp_gpus = args.gpus if os.path.exists(gpu_path) else inferred_pmon_gpus

        start, end, dur_g, gpus, n1 = summarize_gpu(gpu_path, exp_gpus)
        notes.extend(n1)

        sysrep, n2 = summarize_sys(sys_path, disk_path, net_path)
        notes.extend(n2)

        procrep = summarize_proc(proc_path)
        pmonrep = summarize_pmon(pmon_path, exp_gpus)
        if (not os.path.exists(gpu_path)) and pmonrep:
            notes.append(f"gpu_metrics missing; using pmon-only GPU visibility (gpus={len(pmonrep)})")

        # ---- KEY PATCH: fill start/end from sys/disk/net/proc if GPU missing ----
        if not start or not end:
            s2, e2, dur2 = time_range_from_csv(sys_path)
            if (not s2 or not e2):
                s2, e2, dur2 = time_range_from_csv(disk_path)
            if (not s2 or not e2):
                s2, e2, dur2 = time_range_from_csv(net_path)
            if (not s2 or not e2):
                s2, e2, dur2 = time_range_from_csv(proc_path)

            if s2 and e2:
                start, end = s2, e2
                if dur_g <= 0 and (not sysrep or sysrep.duration_s <= 0):
                    dur_g = dur2

        # Duration preference: sys duration if present else gpu-derived duration
        dur = sysrep.duration_s if (sysrep and sysrep.duration_s > 0) else dur_g

        # Bottleneck heuristics (only if sys exists)
        if sysrep:
            if sysrep.cpu_iowait.p95 >= 10:
                notes.append(f"high iowait p95={fmt(sysrep.cpu_iowait.p95)}% suggests storage bottleneck")
            active = [g for g in gpus if g.active]
            if gpus and not active:
                notes.append("no GPU appears active (heuristic)")
            elif len(active) == 1 and active[0].gpu == 0 and active[0].util.mean < 5:
                notes.append("only GPU0 active with low mean util: likely CPU/I/O dominated or bursty kernels")
        else:
            notes.append("no sys_metrics/disk_io available; cannot infer CPU/I/O bottlenecks")

        rep = PrefixReport(
            prefix=pfx,
            start=start,
            end=end,
            duration_s=dur,
            gpus=gpus,
            sys=sysrep,
            proc=procrep,
            pmon=pmonrep,
            notes=notes,
        )
        reports.append(rep)

        # Always write per-prefix TSV/JSON with requested naming.
        jobid, idx = parse_jobid_and_index(pfx)
        if jobid and idx:
            base = f"summary_{jobid}_{idx}"
        else:
            base = f"summary_{safe_slug(pfx)}"

        out_tsv = os.path.join(d, f"{base}.tsv")
        out_json = os.path.join(d, f"{base}.json")

        write_tsv(out_tsv, [flatten_tsv(rep)])
        write_json(out_json, asdict(rep))

    # Print all reports to stdout
    for r in reports:
        print_report(r)

    # Also write an aggregate "ALL" file (one row per prefix) for convenience.
    jobids = [parse_jobid_and_index(r.prefix)[0] for r in reports]
    jobids = [j for j in jobids if j]
    agg_jobid = jobids[0] if (jobids and all(j == jobids[0] for j in jobids)) else "ALL"

    agg_tsv = os.path.join(d, f"summary_{agg_jobid}_ALL.tsv")
    agg_json = os.path.join(d, f"summary_{agg_jobid}_ALL.json")
    write_tsv(agg_tsv, [flatten_tsv(r) for r in reports])
    write_json(agg_json, [asdict(r) for r in reports])

    print(f"Wrote per-prefix TSV/JSON into: {d}")
    print(f"Wrote aggregate TSV:  {agg_tsv}")
    print(f"Wrote aggregate JSON: {agg_json}")


if __name__ == "__main__":
    main()

