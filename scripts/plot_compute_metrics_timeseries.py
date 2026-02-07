#!/usr/bin/env python3
"""
plot_compute_metrics_timeseries.py

Generate a multi-page PDF with continuous time-series plots from compute_metrics logs.

Input: directory with files like:
  sys_metrics_<prefix>.csv
  disk_io_<prefix>.csv
  net_io_<prefix>.csv            (optional)
  proc_metrics_<prefix>.csv       (optional)
  gpu_metrics_<prefix>.csv        (optional)
  gpu_pmon_<prefix>.log           (optional)

Output:
  compute_metrics_timeseries_<jobid>.pdf (multi-page; one page per prefix)
  Optionally per-prefix PDFs with --split

Stdlib + matplotlib only.

Usage:
  python3 plot_compute_metrics_timeseries.py /path/to/compute_metrics
  python3 plot_compute_metrics_timeseries.py /path/to/compute_metrics --prefix stageA_50879909_
  python3 plot_compute_metrics_timeseries.py /path/to/compute_metrics --out out.pdf
  python3 plot_compute_metrics_timeseries.py /path/to/compute_metrics --split
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_time_iso(s: str) -> Optional[datetime]:
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
        return float("nan")


def read_csv_dicts(path: str, delim: str = ",") -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def time_seconds_from_rows(rows: List[Dict[str, str]], ts_field: str = "timestamp") -> Tuple[List[float], Optional[datetime], Optional[datetime]]:
    """
    Try to parse timestamp column; returns:
      t_sec: seconds since first timestamp (or index if missing)
      t0, t1: datetime range if available else None
    """
    times: List[datetime] = []
    for r in rows:
        t = parse_time_iso(r.get(ts_field, ""))
        if t:
            times.append(t)

    if times:
        times_sorted = sorted(times)
        t0 = times_sorted[0]
        t1 = times_sorted[-1]
        # Map row-wise timestamps to seconds (best effort; if some rows missing ts, they become NaN then replaced)
        t_sec: List[float] = []
        last_valid = t0
        for r in rows:
            t = parse_time_iso(r.get(ts_field, ""))
            if t is None:
                # fall back to last known time to keep lengths aligned
                t = last_valid
            else:
                last_valid = t
            t_sec.append((t - t0).total_seconds())
        return t_sec, t0, t1

    # fallback: index
    t_sec = [float(i) for i in range(len(rows))]
    return t_sec, None, None


def parse_jobid_from_prefix(prefix: str) -> Optional[str]:
    m = re.search(r"_(\d{5,})_(\d+)(?:_|$)", prefix)
    return m.group(1) if m else None


def prefix_sort_key(prefix: str) -> Tuple[int, int, str]:
    m = re.search(r"_(\d{5,})_(\d+)(?:_|$)", prefix)
    if not m:
        return (0, 10**9, prefix)
    return (int(m.group(1)), int(m.group(2)), prefix)


def pretty_prefix_label(prefix: str) -> str:
    # stageA_50879909_7_fsitgl-hpc107p -> "idx=7 node=hpc107p"
    m = re.search(r"_(\d{5,})_(\d+)_([^_]+)$", prefix)
    if not m:
        return prefix
    idx = m.group(2)
    node = m.group(3).replace("fsitgl-", "")
    return f"idx={idx}  node={node}"


_PREFIX_SPECS = [
    ("gpu_metrics_", ".csv"),
    ("gpu_pmon_", ".log"),
    ("sys_metrics_", ".csv"),
    ("disk_io_", ".csv"),
    ("net_io_", ".csv"),
    ("proc_metrics_", ".csv"),
]


def list_prefixes(d: str) -> List[str]:
    prefixes: set[str] = set()
    for stem, ext in _PREFIX_SPECS:
        pat = os.path.join(d, f"{stem}*{ext}")
        for p in glob.glob(pat):
            b = os.path.basename(p)
            if not (b.startswith(stem) and b.endswith(ext)):
                continue
            pfx = b[len(stem):-len(ext)]
            if pfx:
                prefixes.add(pfx)
    return sorted(prefixes, key=prefix_sort_key)


# -----------------------------
# Timeseries extraction
# -----------------------------

@dataclass
class TS:
    t: List[float]
    series: Dict[str, List[float]]
    t0: Optional[datetime]
    t1: Optional[datetime]


def load_sys_metrics(path: str) -> Optional[TS]:
    if not os.path.exists(path):
        return None
    rows = read_csv_dicts(path, delim=",")
    if not rows:
        return None
    t, t0, t1 = time_seconds_from_rows(rows, ts_field="timestamp")
    out = {
        "cpu_total_pct": [safe_float(r.get("cpu_total_pct", "")) for r in rows],
        "cpu_iowait_pct": [safe_float(r.get("cpu_iowait_pct", "")) for r in rows],
        "load1": [safe_float(r.get("load1", "")) for r in rows],
        "mem_used_mib": [safe_float(r.get("mem_used_mib", "")) for r in rows],
        "mem_avail_mib": [safe_float(r.get("mem_avail_mib", "")) for r in rows],
    }
    return TS(t=t, series=out, t0=t0, t1=t1)


def load_disk_io(path: str) -> Optional[TS]:
    if not os.path.exists(path):
        return None
    rows = read_csv_dicts(path, delim=",")
    if not rows:
        return None
    t, t0, t1 = time_seconds_from_rows(rows, ts_field="timestamp")
    out = {
        "disk_read_MBps": [safe_float(r.get("disk_read_MBps", "")) for r in rows],
        "disk_write_MBps": [safe_float(r.get("disk_write_MBps", "")) for r in rows],
        "disk_read_iops": [safe_float(r.get("disk_read_iops", "")) for r in rows],
        "disk_write_iops": [safe_float(r.get("disk_write_iops", "")) for r in rows],
    }
    return TS(t=t, series=out, t0=t0, t1=t1)


def load_net_io(path: str) -> Optional[TS]:
    if not os.path.exists(path):
        return None
    rows = read_csv_dicts(path, delim=",")
    if not rows:
        return None
    t, t0, t1 = time_seconds_from_rows(rows, ts_field="timestamp")
    out = {
        "net_rx_MBps": [safe_float(r.get("net_rx_MBps", "")) for r in rows],
        "net_tx_MBps": [safe_float(r.get("net_tx_MBps", "")) for r in rows],
    }
    return TS(t=t, series=out, t0=t0, t1=t1)


def load_proc_metrics(path: str) -> Optional[TS]:
    if not os.path.exists(path):
        return None
    rows = read_csv_dicts(path, delim=",")
    if not rows:
        return None
    t, t0, t1 = time_seconds_from_rows(rows, ts_field="timestamp")
    out = {
        "proc_cpu_pct": [safe_float(r.get("proc_cpu_pct", "")) for r in rows],
        "proc_rss_mib": [safe_float(r.get("proc_rss_mib", "")) for r in rows],
        "proc_vms_mib": [safe_float(r.get("proc_vms_mib", "")) for r in rows],
        "proc_nprocs": [safe_float(r.get("proc_nprocs", "")) for r in rows],
    }
    return TS(t=t, series=out, t0=t0, t1=t1)


def load_gpu_metrics(path: str) -> Optional[Tuple[Dict[int, TS], Optional[datetime], Optional[datetime]]]:
    """
    Returns:
      per_gpu: {gpu_id: TS(t, series=metrics)}
      t0/t1 overall if timestamps exist
    """
    if not os.path.exists(path):
        return None
    rows = read_csv_dicts(path, delim=",")
    if not rows:
        return None

    # Group by GPU
    by_gpu: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        try:
            g = int(float(r.get("gpu", "0") or "0"))
        except Exception:
            g = 0
        by_gpu.setdefault(g, []).append(r)

    # Determine overall range (best effort)
    all_times: List[datetime] = []
    for r in rows:
        t = parse_time_iso(r.get("timestamp", ""))
        if t:
            all_times.append(t)
    t0 = min(all_times) if all_times else None
    t1 = max(all_times) if all_times else None

    per_gpu_ts: Dict[int, TS] = {}
    for g, grows in by_gpu.items():
        t, gt0, gt1 = time_seconds_from_rows(grows, ts_field="timestamp")
        out = {
            "util_gpu_pct": [safe_float(r.get("util_gpu_pct", "")) for r in grows],
            "mem_used_mib": [safe_float(r.get("mem_used_mib", "")) for r in grows],
            "power_w": [safe_float(r.get("power_w", "")) for r in grows],
            "temp_c": [safe_float(r.get("temp_c", "")) for r in grows],
            "clock_sm_mhz": [safe_float(r.get("clock_sm_mhz", "")) for r in grows],
        }
        per_gpu_ts[g] = TS(t=t, series=out, t0=gt0, t1=gt1)

    return per_gpu_ts, t0, t1


def load_gpu_pmon(path: str) -> Optional[Tuple[Dict[int, TS], Optional[datetime], Optional[datetime]]]:
    """
    Parse nvidia-smi pmon -o DT log. We build SM% timeseries per GPU.
    """
    if not os.path.exists(path):
        return None

    date_re = re.compile(r"^\d{8}$")
    time_re = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    per_gpu_times: Dict[int, List[datetime]] = {}
    per_gpu_sm: Dict[int, List[float]] = {}
    all_times: List[datetime] = []

    with open(path, "r", errors="replace") as f:
        for ln in f:
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

            # pmon columns: date time gpu pid type sm mem ...
            sm_tok = toks[5] if len(toks) > 5 else "-"
            sm = safe_float(sm_tok) if sm_tok != "-" else float("nan")

            # timestamp
            try:
                dt = datetime.strptime(toks[0] + " " + toks[1], "%Y%m%d %H:%M:%S")
            except Exception:
                continue

            per_gpu_times.setdefault(g, []).append(dt)
            per_gpu_sm.setdefault(g, []).append(sm)
            all_times.append(dt)

    if not all_times:
        return None

    t0 = min(all_times)
    t1 = max(all_times)

    per_gpu_ts: Dict[int, TS] = {}
    for g in sorted(per_gpu_times.keys()):
        times = per_gpu_times[g]
        smvals = per_gpu_sm[g]
        tsec = [(t - t0).total_seconds() for t in times]
        per_gpu_ts[g] = TS(t=tsec, series={"pmon_sm_pct": smvals}, t0=t0, t1=t1)

    return per_gpu_ts, t0, t1


# -----------------------------
# Plotting
# -----------------------------

def _best_time_window(*ranges: Tuple[Optional[datetime], Optional[datetime]]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Choose an overall time window. Prefer earliest start and latest end among non-None.
    """
    starts = [r[0] for r in ranges if r[0] is not None]
    ends = [r[1] for r in ranges if r[1] is not None]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def plot_line(ax, t: List[float], y: List[float], label: Optional[str] = None):
    ax.plot(t, y, label=label, linewidth=1.2)


def make_prefix_page(prefix: str, d: str) -> Optional[plt.Figure]:
    sys_path  = os.path.join(d, f"sys_metrics_{prefix}.csv")
    disk_path = os.path.join(d, f"disk_io_{prefix}.csv")
    net_path  = os.path.join(d, f"net_io_{prefix}.csv")
    proc_path = os.path.join(d, f"proc_metrics_{prefix}.csv")
    gpu_path  = os.path.join(d, f"gpu_metrics_{prefix}.csv")
    pmon_path = os.path.join(d, f"gpu_pmon_{prefix}.log")

    sys_ts = load_sys_metrics(sys_path)
    disk_ts = load_disk_io(disk_path)
    net_ts = load_net_io(net_path)
    proc_ts = load_proc_metrics(proc_path)
    gpu_loaded = load_gpu_metrics(gpu_path)
    pmon_loaded = load_gpu_pmon(pmon_path)

    per_gpu = gpu_loaded[0] if gpu_loaded else None
    gpu_t0t1 = (gpu_loaded[1], gpu_loaded[2]) if gpu_loaded else (None, None)
    per_pmon = pmon_loaded[0] if pmon_loaded else None
    pmon_t0t1 = (pmon_loaded[1], pmon_loaded[2]) if pmon_loaded else (None, None)

    # If literally nothing exists, skip
    if not any([sys_ts, disk_ts, net_ts, proc_ts, per_gpu, per_pmon]):
        return None

    # Decide which panels to include
    have_gpu = per_gpu is not None and len(per_gpu) > 0
    have_pmon = per_pmon is not None and len(per_pmon) > 0
    have_net = net_ts is not None

    # Layout: 3 columns, variable rows
    # Row 1: CPU, IOwait, Load
    # Row 2: Mem used/avail, Disk MBps, Disk IOPS
    # Row 3: (optional) Net RX/TX, Proc CPU/RSS, GPU util
    # Row 4: (optional) GPU mem/power/temp, pmon SM
    panels: List[str] = []

    panels += ["cpu_total", "cpu_iowait", "load1"]
    panels += ["mem", "disk_mb", "disk_iops"]

    panels += ["net" if have_net else "blank",
               "proc" if proc_ts else "blank",
               "gpu_util" if have_gpu else ("pmon_sm" if have_pmon else "blank")]

    if have_gpu or have_pmon:
        panels += [
            "gpu_mem" if have_gpu else "blank",
            "gpu_power_temp" if have_gpu else "blank",
            "pmon_sm" if have_pmon else "blank",
        ]

    # Build figure
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig_w, fig_h = 16, max(9, 2.6 * nrows)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Overall time window for title (best effort)
    t0, t1 = _best_time_window(
        (sys_ts.t0, sys_ts.t1) if sys_ts else (None, None),
        (disk_ts.t0, disk_ts.t1) if disk_ts else (None, None),
        (net_ts.t0, net_ts.t1) if net_ts else (None, None),
        (proc_ts.t0, proc_ts.t1) if proc_ts else (None, None),
        gpu_t0t1,
        pmon_t0t1,
    )
    label = pretty_prefix_label(prefix)
    if t0 and t1:
        fig.suptitle(f"{prefix}  |  {label}\n{t0.isoformat(timespec='seconds')} → {t1.isoformat(timespec='seconds')}", fontsize=12)
    else:
        fig.suptitle(f"{prefix}  |  {label}", fontsize=12)

    for i, name in enumerate(panels, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("seconds", fontsize=9)

        if name == "blank":
            ax.axis("off")
            continue

        if name == "cpu_total":
            ax.set_title("CPU total (%)", fontsize=10)
            if sys_ts:
                plot_line(ax, sys_ts.t, sys_ts.series["cpu_total_pct"])
            ax.set_ylabel("%", fontsize=9)

        elif name == "cpu_iowait":
            ax.set_title("CPU iowait (%)", fontsize=10)
            if sys_ts:
                plot_line(ax, sys_ts.t, sys_ts.series["cpu_iowait_pct"])
            ax.set_ylabel("%", fontsize=9)

        elif name == "load1":
            ax.set_title("Load1", fontsize=10)
            if sys_ts:
                plot_line(ax, sys_ts.t, sys_ts.series["load1"])
            ax.set_ylabel("load", fontsize=9)

        elif name == "mem":
            ax.set_title("Memory (GiB)", fontsize=10)
            if sys_ts:
                used = [v / 1024.0 for v in sys_ts.series["mem_used_mib"]]
                avail = [v / 1024.0 for v in sys_ts.series["mem_avail_mib"]]
                plot_line(ax, sys_ts.t, used, label="used")
                plot_line(ax, sys_ts.t, avail, label="avail")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("GiB", fontsize=9)

        elif name == "disk_mb":
            ax.set_title("Disk throughput (MB/s)", fontsize=10)
            if disk_ts:
                plot_line(ax, disk_ts.t, disk_ts.series["disk_read_MBps"], label="read")
                plot_line(ax, disk_ts.t, disk_ts.series["disk_write_MBps"], label="write")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("MB/s", fontsize=9)

        elif name == "disk_iops":
            ax.set_title("Disk IOPS", fontsize=10)
            if disk_ts:
                plot_line(ax, disk_ts.t, disk_ts.series["disk_read_iops"], label="read")
                plot_line(ax, disk_ts.t, disk_ts.series["disk_write_iops"], label="write")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("IOPS", fontsize=9)

        elif name == "net":
            ax.set_title("Network (MB/s)", fontsize=10)
            if net_ts:
                plot_line(ax, net_ts.t, net_ts.series["net_rx_MBps"], label="rx")
                plot_line(ax, net_ts.t, net_ts.series["net_tx_MBps"], label="tx")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("MB/s", fontsize=9)

        elif name == "proc":
            ax.set_title("Tracked process", fontsize=10)
            if proc_ts:
                plot_line(ax, proc_ts.t, proc_ts.series["proc_cpu_pct"], label="cpu%")
                rss = [v / 1024.0 for v in proc_ts.series["proc_rss_mib"]]
                plot_line(ax, proc_ts.t, rss, label="rss GiB")
                if any(v == v for v in proc_ts.series.get("proc_nprocs", [])):
                    ax2 = ax.twinx()
                    plot_line(ax2, proc_ts.t, proc_ts.series["proc_nprocs"], label="nprocs")
                    ax2.set_ylabel("nprocs", fontsize=9)
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
                else:
                    ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("mixed", fontsize=9)

        elif name == "gpu_util":
            ax.set_title("GPU util (%)", fontsize=10)
            if per_gpu:
                for g in sorted(per_gpu.keys()):
                    ts = per_gpu[g]
                    plot_line(ax, ts.t, ts.series["util_gpu_pct"], label=f"GPU{g}")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("%", fontsize=9)

        elif name == "gpu_mem":
            ax.set_title("GPU memory used (GiB)", fontsize=10)
            if per_gpu:
                for g in sorted(per_gpu.keys()):
                    ts = per_gpu[g]
                    mem_gib = [v / 1024.0 for v in ts.series["mem_used_mib"]]
                    plot_line(ax, ts.t, mem_gib, label=f"GPU{g}")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("GiB", fontsize=9)

        elif name == "gpu_power_temp":
            ax.set_title("GPU power (W) / temp (°C)", fontsize=10)
            # overlay two quantities with twin axis for readability
            if per_gpu:
                ax2 = ax.twinx()
                # power on ax, temp on ax2
                for g in sorted(per_gpu.keys()):
                    ts = per_gpu[g]
                    plot_line(ax, ts.t, ts.series["power_w"], label=f"P GPU{g}")
                    plot_line(ax2, ts.t, ts.series["temp_c"], label=f"T GPU{g}")
                ax.set_ylabel("W", fontsize=9)
                ax2.set_ylabel("°C", fontsize=9)

                # Combine legends
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
            else:
                ax.set_ylabel("W", fontsize=9)

        elif name == "pmon_sm":
            ax.set_title("pmon SM (%)", fontsize=10)
            if per_pmon:
                for g in sorted(per_pmon.keys()):
                    ts = per_pmon[g]
                    plot_line(ax, ts.t, ts.series["pmon_sm_pct"], label=f"GPU{g}")
                ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("%", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="compute_metrics directory")
    ap.add_argument("--prefix", default="", help="substring filter for prefixes")
    ap.add_argument("--out", default="", help="output PDF (default: compute_metrics_timeseries_<jobid>.pdf in dir)")
    ap.add_argument("--split", action="store_true", help="also write per-prefix PDFs")
    args = ap.parse_args()

    d = os.path.realpath(args.dir)
    if not os.path.isdir(d):
        raise SystemExit(f"ERROR: not a directory: {d}")

    prefixes = list_prefixes(d)
    if args.prefix:
        prefixes = [p for p in prefixes if args.prefix in p]
    if not prefixes:
        raise SystemExit("ERROR: no metric files found for any prefix (or none match --prefix)")

    # Determine jobid for output naming (best effort: first prefix that yields one)
    jobid = None
    for p in prefixes:
        jobid = parse_jobid_from_prefix(p)
        if jobid:
            break
    jobid = jobid or "ALL"

    out_pdf = args.out or os.path.join(d, f"compute_metrics_timeseries_{jobid}.pdf")

    pages_written = 0
    with PdfPages(out_pdf) as pdf:
        for pfx in prefixes:
            fig = make_prefix_page(pfx, d)
            if fig is None:
                continue
            pdf.savefig(fig)
            plt.close(fig)
            pages_written += 1

            if args.split:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pfx)
                per_pdf = os.path.join(d, f"compute_metrics_timeseries_{safe}.pdf")
                # write single-page PDF
                with PdfPages(per_pdf) as pp:
                    fig2 = make_prefix_page(pfx, d)
                    if fig2 is not None:
                        pp.savefig(fig2)
                        plt.close(fig2)

    if pages_written == 0:
        raise SystemExit("ERROR: no pages written (no readable metric content)")
    print(f"Wrote PDF: {out_pdf}  (pages: {pages_written})")
    if args.split:
        print("Also wrote per-prefix PDFs in the same directory.")


if __name__ == "__main__":
    main()
