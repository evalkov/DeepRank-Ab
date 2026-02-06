#!/usr/bin/env python3
"""Live progress viewer for Stage A shards.

Reads progress.json sidecars and STAGEA_DONE markers from each shard directory
and prints a compact per-shard + total progress table.

Usage:
    python3 scripts/stageA_progress_live.py --run-root /path/to/run
    python3 scripts/stageA_progress_live.py --run-root /path/to/run --watch 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def _short_host(hostname: str, max_len: int = 16) -> str:
    """Strip domain suffix and truncate to max_len."""
    short = hostname.split(".")[0] if "." in hostname else hostname
    if len(short) > max_len:
        short = short[:max_len - 1] + "~"
    return short


def _elapsed_str(started_at: Optional[str]) -> str:
    if not started_at:
        return "-"
    try:
        t0 = datetime.fromisoformat(started_at)
        delta = datetime.now() - t0
        secs = int(delta.total_seconds())
        if secs < 0:
            return "-"
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m"
        return f"{m}m {s:02d}s"
    except (ValueError, TypeError):
        return "-"


def collect_shards(run_root: Path) -> List[dict]:
    """Collect state for each shard from shard lists, progress.json, and STAGEA_DONE."""
    shard_lists_dir = run_root / "shard_lists"
    shards_dir = run_root / "shards"

    # Discover all shard IDs from shard list files
    shard_ids: List[str] = []
    if shard_lists_dir.is_dir():
        for lst in sorted(shard_lists_dir.glob("shard_*.lst")):
            sid = lst.stem.replace("shard_", "")
            shard_ids.append(sid)

    results: List[dict] = []
    for sid in shard_ids:
        shard_dir = shards_dir / f"shard_{sid}"
        done_marker = shard_dir / "STAGEA_DONE"
        progress_file = shard_dir / "progress.json"
        list_file = shard_lists_dir / f"shard_{sid}.lst"

        n_list_pdbs = _count_lines(list_file)
        prog = _read_json(progress_file)
        graph_prog = _read_json(shard_dir / "graph_progress.json")
        graphs_done = graph_prog.get("graphs_done", 0) if graph_prog else 0
        graphs_total = graph_prog.get("graphs_total", 0) if graph_prog else 0

        if done_marker.exists() and prog:
            results.append({
                "shard_id": sid,
                "status": "done",
                "pid": prog.get("pid", "-"),
                "hostname": prog.get("hostname", "-"),
                "stage": "done",
                "n_inputs": prog.get("n_inputs", n_list_pdbs),
                "n_models": prog.get("n_models", 0),
                "prep_done": prog.get("prep_done", 0),
                "prep_ok": prog.get("prep_ok", 0),
                "prep_fail": prog.get("prep_fail", 0),
                "started_at": prog.get("started_at"),
                "elapsed": _elapsed_str(prog.get("started_at")),
                "graphs_done": graphs_done,
                "graphs_total": graphs_total,
            })
        elif prog:
            results.append({
                "shard_id": sid,
                "status": "running",
                "pid": prog.get("pid", "-"),
                "hostname": prog.get("hostname", "-"),
                "stage": prog.get("stage", "?"),
                "n_inputs": prog.get("n_inputs", n_list_pdbs),
                "n_models": prog.get("n_models", 0),
                "prep_done": prog.get("prep_done", 0),
                "prep_ok": prog.get("prep_ok", 0),
                "prep_fail": prog.get("prep_fail", 0),
                "started_at": prog.get("started_at"),
                "elapsed": _elapsed_str(prog.get("started_at")),
                "graphs_done": graphs_done,
                "graphs_total": graphs_total,
            })
        else:
            results.append({
                "shard_id": sid,
                "status": "pending",
                "pid": "-",
                "hostname": "-",
                "stage": "pending",
                "n_inputs": n_list_pdbs,
                "n_models": 0,
                "prep_done": 0,
                "prep_ok": 0,
                "prep_fail": 0,
                "started_at": None,
                "elapsed": "-",
                "graphs_done": 0,
                "graphs_total": 0,
            })

    return results


def format_prep(s: dict) -> str:
    if s["status"] == "pending":
        return "-"
    n_models = s["n_models"]
    if n_models == 0:
        return "-"
    done = s["prep_done"]
    # Show graph-building progress when in "graphs" stage
    if s["stage"] == "graphs":
        gt = s.get("graphs_total", 0)
        if gt > 0:
            gd = s["graphs_done"]
            pct = gd / gt * 100
            return f"{gd}/{gt} ({pct:.0f}%)"
        return "starting..."
    if s["status"] == "done" or s["stage"] not in ("copy", "expand", "prep"):
        fail_str = f"  {s['prep_fail']} fail" if s["prep_fail"] else ""
        return f"{s['prep_ok']}/{n_models} ok{fail_str}"
    return f"{done}/{n_models}"


def print_table(run_root: Path, shards: List[dict]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"StageA progress   {now}   RUN_ROOT: {run_root}")
    print()

    # Header
    hdr = f"{'Shard':<8}{'PID':<8}{'Host':<18}{'Stage':<12}{'Progress':<18}{'Elapsed':<10}"
    print(hdr)
    print("-" * len(hdr))

    for s in shards:
        pid = str(s["pid"]) if s["pid"] != "-" else "-"
        host = _short_host(str(s["hostname"])) if s["hostname"] != "-" else "-"
        prep = format_prep(s)
        print(f"{s['shard_id']:<8}{pid:<8}{host:<18}{s['stage']:<12}{prep:<18}{s['elapsed']:<10}")

    print()

    # Totals
    n_total = len(shards)
    n_done = sum(1 for s in shards if s["status"] == "done")
    n_running = sum(1 for s in shards if s["status"] == "running")
    n_pending = sum(1 for s in shards if s["status"] == "pending")

    total_models = sum(s["n_models"] for s in shards)
    total_prepped = sum(s["prep_ok"] + s["prep_fail"] for s in shards)
    total_ok = sum(s["prep_ok"] for s in shards)
    total_fail = sum(s["prep_fail"] for s in shards)

    print(f"Shards: {n_done} done, {n_running} running, {n_pending} pending  ({n_total} total)")

    if total_models > 0:
        pct = total_prepped / total_models * 100
        fail_str = f"  {total_fail} fail" if total_fail else ""
        print(f"PDBs:   {total_prepped}/{total_models} prepped ({pct:.1f}%)   {total_ok} ok{fail_str}")
    else:
        total_inputs = sum(s["n_inputs"] for s in shards)
        print(f"PDBs:   {total_inputs} inputs across shards (models not yet expanded)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live progress viewer for Stage A shards")
    ap.add_argument("--run-root", required=True, help="Run root directory (BeeGFS)")
    ap.add_argument("--watch", type=int, default=0, metavar="N",
                    help="Auto-refresh every N seconds (0 = print once)")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.is_dir():
        print(f"ERROR: {run_root} is not a directory", file=sys.stderr)
        return 1

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                shards = collect_shards(run_root)
                print_table(run_root, shards)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
    else:
        shards = collect_shards(run_root)
        print_table(run_root, shards)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
