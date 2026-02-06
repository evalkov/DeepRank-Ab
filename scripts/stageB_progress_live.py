#!/usr/bin/env python3
"""Live progress viewer for Stage B shards.

Reads progress_shard_*.json sidecars and DONE_shard_*.ok markers from the preds
directory and prints a compact per-shard + total progress table.

Usage:
    python3 scripts/stageB_progress_live.py --run-root /path/to/run
    python3 scripts/stageB_progress_live.py --run-root /path/to/run --watch 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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
    """Collect state for each shard from shard dirs, preds progress files, and done/failed markers."""
    shards_dir = run_root / "shards"
    preds_dir = run_root / "preds"

    # Discover all shard IDs from Stage A shard directories
    shard_ids: List[str] = []
    if shards_dir.is_dir():
        for d in sorted(shards_dir.glob("shard_*")):
            if d.is_dir():
                sid = d.name.replace("shard_", "")
                if sid:
                    shard_ids.append(sid)

    results: List[dict] = []
    for sid in shard_ids:
        stageA_done = shards_dir / f"shard_{sid}" / "STAGEA_DONE"
        done_marker = preds_dir / f"DONE_shard_{sid}.ok"
        failed_marker = preds_dir / f"FAILED_shard_{sid}.txt"
        progress_file = preds_dir / f"progress_shard_{sid}.json"

        prog = _read_json(progress_file)

        if not stageA_done.exists():
            # Stage A not done yet — Stage B can't run
            results.append({
                "shard_id": sid,
                "status": "waiting_A",
                "pid": "-",
                "hostname": "-",
                "stage": "waiting_A",
                "n_sequences": 0,
                "started_at": None,
                "elapsed": "-",
            })
        elif failed_marker.exists():
            results.append({
                "shard_id": sid,
                "status": "failed",
                "pid": prog.get("pid", "-") if prog else "-",
                "hostname": prog.get("hostname", "-") if prog else "-",
                "stage": "FAILED",
                "n_sequences": prog.get("n_sequences", 0) if prog else 0,
                "started_at": prog.get("started_at") if prog else None,
                "elapsed": _elapsed_str(prog.get("started_at")) if prog else "-",
            })
        elif done_marker.exists():
            results.append({
                "shard_id": sid,
                "status": "done",
                "pid": prog.get("pid", "-") if prog else "-",
                "hostname": prog.get("hostname", "-") if prog else "-",
                "stage": "done",
                "n_sequences": prog.get("n_sequences", 0) if prog else 0,
                "started_at": prog.get("started_at") if prog else None,
                "elapsed": _elapsed_str(prog.get("started_at")) if prog else "-",
            })
        elif prog:
            results.append({
                "shard_id": sid,
                "status": "running",
                "pid": prog.get("pid", "-"),
                "hostname": prog.get("hostname", "-"),
                "stage": prog.get("stage", "?"),
                "n_sequences": prog.get("n_sequences", 0),
                "started_at": prog.get("started_at"),
                "elapsed": _elapsed_str(prog.get("started_at")),
            })
        else:
            results.append({
                "shard_id": sid,
                "status": "pending",
                "pid": "-",
                "hostname": "-",
                "stage": "pending",
                "n_sequences": 0,
                "started_at": None,
                "elapsed": "-",
            })

    return results


def print_table(run_root: Path, shards: List[dict], preds_dir: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"StageB progress   {now}   RUN_ROOT: {run_root}")
    print()

    # Header
    hdr = f"{'Shard':<8}{'PID':<8}{'Host':<18}{'Stage':<12}{'Seqs':<10}{'Elapsed':<10}"
    print(hdr)
    print("-" * len(hdr))

    for s in shards:
        pid = str(s["pid"]) if s["pid"] != "-" else "-"
        host = _short_host(str(s["hostname"])) if s["hostname"] != "-" else "-"
        seqs = str(s["n_sequences"]) if s["n_sequences"] > 0 else "-"
        print(f"{s['shard_id']:<8}{pid:<8}{host:<18}{s['stage']:<12}{seqs:<10}{s['elapsed']:<10}")

    print()

    # Totals
    n_total = len(shards)
    n_done = sum(1 for s in shards if s["status"] == "done")
    n_running = sum(1 for s in shards if s["status"] == "running")
    n_failed = sum(1 for s in shards if s["status"] == "failed")
    n_pending = sum(1 for s in shards if s["status"] == "pending")
    n_waiting = sum(1 for s in shards if s["status"] == "waiting_A")

    parts = [f"{n_done} done", f"{n_running} running", f"{n_pending} pending"]
    if n_failed:
        parts.append(f"{n_failed} FAILED")
    if n_waiting:
        parts.append(f"{n_waiting} waiting_A")
    print(f"Shards: {', '.join(parts)}  ({n_total} total)")

    total_seqs = sum(s["n_sequences"] for s in shards)
    if total_seqs > 0:
        done_seqs = sum(s["n_sequences"] for s in shards if s["status"] == "done")
        pct = done_seqs / total_seqs * 100
        print(f"Seqs:   {done_seqs}/{total_seqs} done ({pct:.1f}%)")

    # Show batch ESM status if available
    batch_prog = _read_json(preds_dir / "progress_batch_esm.json")
    if batch_prog:
        bstage = batch_prog.get("stage", "?")
        bseqs = batch_prog.get("n_total_sequences", 0)
        bhost = _short_host(batch_prog.get("hostname", "?"))
        cur = batch_prog.get("current_shard", "")
        cur_str = f"  shard={cur}" if cur else ""
        print(f"Batch:  stage={bstage}  seqs={bseqs}  host={bhost}{cur_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live progress viewer for Stage B shards")
    ap.add_argument("--run-root", required=True, help="Run root directory (BeeGFS)")
    ap.add_argument("--watch", type=int, default=0, metavar="N",
                    help="Auto-refresh every N seconds (0 = print once)")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.is_dir():
        print(f"ERROR: {run_root} is not a directory", file=sys.stderr)
        return 1

    preds_dir = run_root / "preds"

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                shards = collect_shards(run_root)
                print_table(run_root, shards, preds_dir)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
    else:
        shards = collect_shards(run_root)
        print_table(run_root, shards, preds_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
