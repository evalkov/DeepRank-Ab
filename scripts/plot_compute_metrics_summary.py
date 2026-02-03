#!/usr/bin/env python3
"""
plot_compute_metrics_summary.py

Create a single-page multi-panel PDF from summarize_compute_metrics.py aggregate TSV.

Input:
  summary_<jobid>_ALL.tsv (generated in the compute_metrics directory)

Output:
  compute_metrics_<jobid>.pdf (single page, multi-panel)

Works with CPU-only runs (no GPU fields) and GPU runs (gpu* fields exist).
Stdlib + matplotlib only.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ffloat(x: Optional[str]) -> float:
    try:
        return float(x) if x is not None and x != "" else float("nan")
    except Exception:
        return float("nan")


def parse_jobid_from_filename(path: str) -> str:
    b = os.path.basename(path)
    m = re.search(r"summary_(\d{5,})_ALL\.tsv$", b)
    return m.group(1) if m else "ALL"


def prefix_sort_key(prefix: str) -> Tuple[int, int, str]:
    """
    Sort by (jobid, index_after_jobid, prefix).
    stageA_50879909_7_fsitgl-hpc107p -> (50879909, 7, prefix)
    """
    m = re.search(r"_(\d{5,})_(\d+)(?:_|$)", prefix)
    if not m:
        return (0, 10**9, prefix)
    return (int(m.group(1)), int(m.group(2)), prefix)


def has_any_key(rows: List[Dict[str, str]], key_prefix: str) -> bool:
    for r in rows:
        for k in r.keys():
            if k.startswith(key_prefix):
                v = r.get(k, "")
                if v not in ("", None):
                    return True
    return False


def collect_numeric(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [ffloat(r.get(key, "")) for r in rows]


def pretty_prefix_label(prefix: str) -> str:
    # keep it compact: show index + node
    # stageA_50879909_7_fsitgl-hpc107p -> "7 hpc107p"
    m = re.search(r"_(\d{5,})_(\d+)_([^_]+)$", prefix)
    if not m:
        return prefix
    idx = m.group(2)
    node = m.group(3)
    node = node.replace("fsitgl-", "")
    return f"{idx} {node}"


def hbar_panel(ax, labels: List[str], values: List[float], title: str, xlabel: str, logx: bool = False):
    # Filter NaNs for autoscale, but keep bars aligned (NaNs become 0 and get a hatch)
    import math

    y = list(range(len(labels)))
    vv = []
    nan_mask = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            vv.append(0.0)
            nan_mask.append(True)
        else:
            vv.append(float(v))
            nan_mask.append(False)

    bars = ax.barh(y, vv)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    if logx:
        # avoid log(0) problems: only enable if there are positive values
        if any(v > 0 for v in vv):
            ax.set_xscale("log")

    # hatch missing values
    for b, is_nan in zip(bars, nan_mask):
        if is_nan:
            b.set_hatch("//")
            b.set_alpha(0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="Path to summary_<jobid>_ALL.tsv")
    ap.add_argument("--out", default="", help="Output PDF path (default: compute_metrics_<jobid>.pdf next to TSV)")
    ap.add_argument("--title", default="", help="Figure title override")
    ap.add_argument("--max-prefix-label", type=int, default=28, help="truncate long labels")
    args = ap.parse_args()

    tsv_path = os.path.realpath(args.tsv)
    if not os.path.exists(tsv_path):
        raise SystemExit(f"ERROR: TSV not found: {tsv_path}")

    rows = read_tsv(tsv_path)
    if not rows:
        raise SystemExit("ERROR: TSV has no rows")

    rows.sort(key=lambda r: prefix_sort_key(r.get("prefix", "")))

    jobid = parse_jobid_from_filename(tsv_path)
    out_pdf = args.out or os.path.join(os.path.dirname(tsv_path), f"compute_metrics_{jobid}.pdf")

    prefixes = [r.get("prefix", "") for r in rows]
    labels = [pretty_prefix_label(p) for p in prefixes]
    labels = [l if len(l) <= args.max_prefix_label else (l[: args.max_prefix_label - 1] + "…") for l in labels]

    # Key series (may be missing; missing will be hatched)
    duration_s = collect_numeric(rows, "duration_s")
    cpu_mean = collect_numeric(rows, "cpu_total_mean")
    cpu_p95 = collect_numeric(rows, "cpu_total_p95")
    iow_p95 = collect_numeric(rows, "cpu_iowait_p95")
    mem_used = collect_numeric(rows, "mem_used_max_mib")

    dr_mb = collect_numeric(rows, "disk_read_MBps_max")
    dw_mb = collect_numeric(rows, "disk_write_MBps_max")
    dr_iops = collect_numeric(rows, "disk_read_iops_max")
    dw_iops = collect_numeric(rows, "disk_write_iops_max")

    net_rx = collect_numeric(rows, "net_rx_MBps_max")
    net_tx = collect_numeric(rows, "net_tx_MBps_max")

    # Convert some units for readability
    duration_min = [v / 60.0 for v in duration_s]
    mem_gb = [v / 1024.0 for v in mem_used]

    # Detect optional groups
    have_net = has_any_key(rows, "net_") or any(str(v) != "nan" for v in net_rx + net_tx)
    have_gpu = any(k.startswith("gpu0_") for k in rows[0].keys()) or has_any_key(rows, "gpu")

    # Determine GPU count present from columns
    gpu_ids = []
    if have_gpu:
        ks = set()
        for k in rows[0].keys():
            m = re.match(r"gpu(\d+)_", k)
            if m:
                ks.add(int(m.group(1)))
        gpu_ids = sorted(ks)

    # Layout: choose panels based on available data
    panels = [
        ("Duration", duration_min, "minutes"),
        ("CPU mean", cpu_mean, "%"),
        ("CPU p95", cpu_p95, "%"),
        ("IOwait p95", iow_p95, "%"),
        ("Mem max", mem_gb, "GB"),
        ("Disk read max", dr_mb, "MB/s"),
        ("Disk write max", dw_mb, "MB/s"),
        ("Disk read IOPS max", dr_iops, "IOPS", True),
        ("Disk write IOPS max", dw_iops, "IOPS", True),
    ]

    if have_net:
        panels += [
            ("Net RX max", net_rx, "MB/s"),
            ("Net TX max", net_tx, "MB/s"),
        ]

    # GPU panels: util mean and max per GPU
    if have_gpu and gpu_ids:
        for gid in gpu_ids:
            um = collect_numeric(rows, f"gpu{gid}_util_mean")
            ux = collect_numeric(rows, f"gpu{gid}_util_max")
            panels.append((f"GPU{gid} util mean", um, "%"))
            panels.append((f"GPU{gid} util max", ux, "%"))

    # Choose grid: aim ~12–16 panels per page
    n = len(panels)
    # 4 columns tends to read well on letter/A4 landscape; adjust rows accordingly
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    # Figure
    fig_w = 16
    fig_h = max(8, 2.2 * nrows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(args.title or f"Compute metrics summary (job {jobid})", fontsize=14)

    for i, p in enumerate(panels, start=1):
        title = p[0]
        values = p[1]
        xlabel = p[2]
        logx = p[3] if len(p) > 3 else False
        ax = fig.add_subplot(nrows, ncols, i)
        hbar_panel(ax, labels, values, title=title, xlabel=xlabel, logx=logx)

    # If grid has empty slots, turn them off
    for j in range(len(panels) + 1, nrows * ncols + 1):
        ax = fig.add_subplot(nrows, ncols, j)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf, format="pdf")
    print(f"Wrote PDF: {out_pdf}")


if __name__ == "__main__":
    main()

