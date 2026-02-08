#!/usr/bin/env python3
"""Unified live tracker for Stage A + Stage B.

Usage:
    python3 scripts/progress_live.py /path/to/RUN_ROOT

Behavior:
    - Single positional argument: RUN_ROOT
    - Refreshes every 5 seconds
    - Rows are aggregated by (stage, jobid), not shard
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_VISIBLE_ROWS = 32


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


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _fmt_wall_s(total_s: int) -> str:
    if total_s < 0:
        return "-"
    d, rem = divmod(total_s, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m {s:02d}s"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _find_stage_c_done_time(run_root: Path) -> Tuple[Optional[Path], Optional[datetime]]:
    logs_dir = run_root / "logs"
    if not logs_dir.is_dir():
        return None, None

    pipeline_jobs = _read_json(run_root / "pipeline_jobs.json") or {}
    jobs = pipeline_jobs.get("jobs", {}) if isinstance(pipeline_jobs, dict) else {}
    c_jobid = str(jobs.get("C", "")).strip() if isinstance(jobs, dict) else ""

    patterns: List[str] = []
    if c_jobid:
        patterns.append(f"drab-C_{c_jobid}.log")
    patterns.append("drab-C_*.log")

    seen: set = set()
    done_hits: List[Tuple[datetime, Path]] = []
    for pat in patterns:
        for log_path in sorted(logs_dir.glob(pat)):
            key = str(log_path)
            if key in seen:
                continue
            seen.add(key)
            try:
                txt = log_path.read_text(errors="ignore")
            except OSError:
                continue
            if "[StageC] Done." not in txt:
                continue
            try:
                dt = datetime.fromtimestamp(log_path.stat().st_mtime)
            except OSError:
                continue
            done_hits.append((dt, log_path))

    if not done_hits:
        return None, None

    done_hits.sort(key=lambda x: x[0], reverse=True)
    return done_hits[0][1], done_hits[0][0]


def _resolve_stage_c_log(run_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return the best Stage C log path and configured C job id (if known)."""
    logs_dir = run_root / "logs"
    if not logs_dir.is_dir():
        return None, None

    pipeline_jobs = _read_json(run_root / "pipeline_jobs.json") or {}
    jobs = pipeline_jobs.get("jobs", {}) if isinstance(pipeline_jobs, dict) else {}
    c_jobid = str(jobs.get("C", "")).strip() if isinstance(jobs, dict) else ""

    candidates: List[Path] = []
    if c_jobid:
        p = logs_dir / f"drab-C_{c_jobid}.log"
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(logs_dir.glob("drab-C_*.log")))

    if not candidates:
        return None, c_jobid or None

    # Prefer newest by mtime.
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return candidates[0], c_jobid or None


def _collect_stage_c_status(run_root: Path) -> str:
    """Build a concise, phase-oriented Stage C status summary from logs."""
    log_path, c_jobid = _resolve_stage_c_log(run_root)

    if log_path is None:
        if c_jobid:
            return f"Stage C: pending | job={c_jobid} (log not created yet)"
        return "Stage C: pending | no Stage C log yet"

    try:
        lines = [ln.strip() for ln in log_path.read_text(errors="ignore").splitlines() if ln.strip()]
    except OSError:
        return f"Stage C: unknown | unable to read {log_path.name}"

    if not lines:
        return f"Stage C: pending | {log_path.name} is empty"

    last_line = lines[-1]
    stage_lines = [ln for ln in lines if "[StageC]" in ln or ln.startswith("✓")]

    def _last_match(pattern: str) -> Optional[str]:
        for ln in reversed(stage_lines):
            if re.search(pattern, ln):
                return ln
        return None

    # Error takes precedence.
    err = _last_match(r"\[StageC\].*ERROR|^ERROR:")
    if err:
        return f"Stage C: failed | {err}"

    if _last_match(r"\[StageC\] Done\."):
        return f"Stage C: done | {log_path.name}"

    post = _last_match(r"compute-metrics|Metrics summarized|Summary plot|Time-series plot|compute_metrics\.(tsv|json)")
    if post:
        return f"Stage C: postprocessing | {post}"

    copy_ln = _last_match(r"predictions\.tsv\.gz ->|stats\.json ->")
    if copy_ln:
        return f"Stage C: copying outputs | {copy_ln}"

    export_ln = _last_match(r"✓ Wrote:|✓ Stats:")
    if export_ln:
        return f"Stage C: exporting | {export_ln}"

    merge_ln = _last_match(r"Merging ->|DONE shards:")
    if merge_ln:
        return f"Stage C: merging shard predictions | {merge_ln}"

    validate_ln = _last_match(r"Waiting for .* shards| done, .* failed, .* pending|All .* shards succeeded")
    if validate_ln:
        return f"Stage C: validating shard completion | {validate_ln}"

    run_root_ln = _last_match(r"\[StageC\] RUN_ROOT=")
    if run_root_ln:
        return f"Stage C: starting | {run_root_ln}"

    return f"Stage C: running | {last_line}"


def _pipeline_start_time(run_root: Path) -> Tuple[Optional[datetime], str]:
    """Resolve pipeline start time from launcher metadata, with fallback."""
    jobs_file = run_root / "pipeline_jobs.json"
    jobs_info = _read_json(jobs_file) or {}
    if isinstance(jobs_info, dict):
        for key in ("pipeline_invoked_at", "pipeline_launched_at", "pipeline_started_at"):
            dt = _parse_iso(jobs_info.get(key))
            if dt is not None:
                return dt, f"{jobs_file.name}:{key}"
    if jobs_file.exists():
        try:
            dt = datetime.fromtimestamp(jobs_file.stat().st_mtime)
            return dt, f"{jobs_file.name}:mtime"
        except OSError:
            pass
    return None, "stage-progress"


def _pipeline_final_line(run_root: Path, stage_a: List[dict], stage_b: List[dict]) -> Optional[str]:
    if not stage_a or not stage_b:
        return None

    a_done = all(s["status"] == "done" for s in stage_a)
    b_done = all(s["status"] == "done" for s in stage_b)
    if not (a_done and b_done):
        return None

    c_log, end_dt = _find_stage_c_done_time(run_root)
    if end_dt is None:
        return None

    start_dt, start_src = _pipeline_start_time(run_root)
    if start_dt is None:
        starts = []
        for s in stage_a:
            dt = _parse_iso(s.get("started_at"))
            if dt is not None:
                starts.append(dt)
        for s in stage_b:
            dt = _parse_iso(s.get("started_at"))
            if dt is not None:
                starts.append(dt)
        if not starts:
            return None
        start_dt = min(starts)
        start_src = "stage-progress"

    wall_s = int((end_dt - start_dt).total_seconds())

    start_s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    wall_s_str = _fmt_wall_s(wall_s)
    source = c_log.name if c_log else "drab-C log"
    return (
        f"Pipeline complete: start={start_s}  end={end_s}  wall={wall_s_str}  "
        f"(start={start_src}; end={source})"
    )


def _state_order(state: str) -> int:
    order = {
        "FAILED": 0,
        "running": 1,
        "mixed": 2,
        "done": 3,
        "pending": 4,
        "waiting_A": 5,
    }
    return order.get(state, 9)


def _discover_shard_ids(run_root: Path) -> List[str]:
    shard_lists_dir = run_root / "shard_lists"
    shards_dir = run_root / "shards"
    shard_ids = set()

    if shard_lists_dir.is_dir():
        for lst in shard_lists_dir.glob("shard_*.lst"):
            shard_ids.add(lst.stem.replace("shard_", ""))

    if shards_dir.is_dir():
        for d in shards_dir.glob("shard_*"):
            if d.is_dir():
                shard_ids.add(d.name.replace("shard_", ""))

    return sorted(sid for sid in shard_ids if sid)


def collect_stage_a(run_root: Path) -> List[dict]:
    shard_lists_dir = run_root / "shard_lists"
    shards_dir = run_root / "shards"
    shard_ids = _discover_shard_ids(run_root)

    out: List[dict] = []
    for sid in shard_ids:
        shard_dir = shards_dir / f"shard_{sid}"
        done_marker = shard_dir / "STAGEA_DONE"
        progress_file = shard_dir / "progress.json"
        graph_progress_file = shard_dir / "graph_progress.json"
        list_file = shard_lists_dir / f"shard_{sid}.lst"

        n_list_pdbs = _count_lines(list_file)
        prog = _read_json(progress_file)
        graph_prog = _read_json(graph_progress_file)
        graphs_done = int((graph_prog or {}).get("graphs_done", 0))
        graphs_total = int((graph_prog or {}).get("graphs_total", 0))

        if done_marker.exists():
            status = "done"
            stage = (prog or {}).get("stage", "done") or "done"
        elif prog:
            status = "running"
            stage = prog.get("stage", "?")
        else:
            status = "pending"
            stage = "pending"

        out.append({
            "stage_name": "A",
            "shard_id": sid,
            "status": status,
            "stage": stage,
            "jobid": (prog or {}).get("jobid", ""),
            "pid": (prog or {}).get("pid", ""),
            "hostname": (prog or {}).get("hostname", "-"),
            "n_inputs": int((prog or {}).get("n_inputs", n_list_pdbs)),
            "n_models": int((prog or {}).get("n_models", 0)),
            "prep_done": int((prog or {}).get("prep_done", 0)),
            "prep_ok": int((prog or {}).get("prep_ok", 0)),
            "prep_fail": int((prog or {}).get("prep_fail", 0)),
            "graphs_done": graphs_done,
            "graphs_total": graphs_total,
            "started_at": (prog or {}).get("started_at"),
            "updated_at": (prog or {}).get("updated_at"),
        })

    return out


def collect_stage_b(run_root: Path) -> List[dict]:
    shards_dir = run_root / "shards"
    preds_dir = run_root / "preds"
    shard_ids = _discover_shard_ids(run_root)

    out: List[dict] = []
    for sid in shard_ids:
        stagea_done = shards_dir / f"shard_{sid}" / "STAGEA_DONE"
        done_marker = preds_dir / f"DONE_shard_{sid}.ok"
        failed_marker = preds_dir / f"FAILED_shard_{sid}.txt"
        progress_file = preds_dir / f"progress_shard_{sid}.json"
        prog = _read_json(progress_file)

        if not stagea_done.exists():
            status = "waiting_A"
            stage = "waiting_A"
        elif failed_marker.exists():
            status = "failed"
            stage = "FAILED"
        elif done_marker.exists():
            status = "done"
            stage = (prog or {}).get("stage", "done") or "done"
        elif prog:
            status = "running"
            stage = prog.get("stage", "?")
        else:
            status = "pending"
            stage = "pending"

        out.append({
            "stage_name": "B",
            "shard_id": sid,
            "status": status,
            "stage": stage,
            "jobid": (prog or {}).get("jobid", ""),
            "pid": (prog or {}).get("pid", ""),
            "hostname": (prog or {}).get("hostname", "-"),
            "n_sequences": int((prog or {}).get("n_sequences", 0)),
            "started_at": (prog or {}).get("started_at"),
            "updated_at": (prog or {}).get("updated_at"),
        })

    return out


def _effective_jobid(item: dict) -> str:
    jobid = str(item.get("jobid") or "").strip()
    if jobid:
        return jobid
    pid = str(item.get("pid") or "").strip()
    if pid:
        return f"pid:{pid}"
    return "unknown"


def _stage_mix_text(stage_counts: Counter) -> str:
    if not stage_counts:
        return "-"
    ordered = sorted(stage_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ordered[:2]
    txt = ", ".join(f"{k}:{v}" for k, v in top)
    if len(ordered) > 2:
        txt += ", ..."
    return txt


def aggregate_job_rows(stage_a: List[dict], stage_b: List[dict]) -> List[dict]:
    rows: Dict[Tuple[str, str], dict] = {}

    def ingest(item: dict) -> None:
        if item["status"] in ("pending", "waiting_A"):
            return

        stage_name = item["stage_name"]
        jobid = _effective_jobid(item)
        key = (stage_name, jobid)
        row = rows.get(key)
        if row is None:
            row = {
                "stage_name": stage_name,
                "jobid": jobid,
                "hosts": Counter(),
                "statuses": Counter(),
                "stages": Counter(),
                "shards_total": 0,
                "started_at_min": None,
                "updated_at_max": None,
                "a_n_models": 0,
                "a_prep_ok": 0,
                "a_prep_fail": 0,
                "a_graphs_done": 0,
                "a_graphs_total": 0,
                "b_n_sequences": 0,
            }
            rows[key] = row

        host = str(item.get("hostname") or "-")
        if host and host != "-":
            row["hosts"][host] += 1
        row["statuses"][item["status"]] += 1
        row["stages"][str(item.get("stage") or "?")] += 1
        row["shards_total"] += 1

        started_at = item.get("started_at")
        if started_at:
            if row["started_at_min"] is None or started_at < row["started_at_min"]:
                row["started_at_min"] = started_at

        updated_at = item.get("updated_at")
        if updated_at:
            if row["updated_at_max"] is None or updated_at > row["updated_at_max"]:
                row["updated_at_max"] = updated_at

        if stage_name == "A":
            row["a_n_models"] += int(item.get("n_models", 0))
            row["a_prep_ok"] += int(item.get("prep_ok", 0))
            row["a_prep_fail"] += int(item.get("prep_fail", 0))
            row["a_graphs_done"] += int(item.get("graphs_done", 0))
            row["a_graphs_total"] += int(item.get("graphs_total", 0))
        else:
            row["b_n_sequences"] += int(item.get("n_sequences", 0))

    for it in stage_a:
        ingest(it)
    for it in stage_b:
        ingest(it)

    out: List[dict] = []
    for _, row in rows.items():
        failed = row["statuses"].get("failed", 0)
        running = row["statuses"].get("running", 0)
        done = row["statuses"].get("done", 0)
        total = row["shards_total"]

        if failed > 0:
            state = "FAILED"
        elif running > 0:
            state = "running"
        elif done == total:
            state = "done"
        else:
            state = "mixed"

        host = "-"
        if row["hosts"]:
            host = _short_host(row["hosts"].most_common(1)[0][0])
            if len(row["hosts"]) > 1:
                host = f"{host}+{len(row['hosts']) - 1}"

        stage_mix = _stage_mix_text(row["stages"])
        shards_done = done
        shard_text = f"{shards_done}/{total}"
        elapsed = _elapsed_str(row["started_at_min"])

        if row["stage_name"] == "A":
            work_text = "-"
            if row["a_graphs_total"] > 0 and row["a_graphs_done"] <= row["a_graphs_total"]:
                work_text = f"graphs {row['a_graphs_done']}/{row['a_graphs_total']}"
            elif row["a_n_models"] > 0:
                fail_sfx = f" f{row['a_prep_fail']}" if row["a_prep_fail"] else ""
                model_text = f"models {row['a_prep_ok']}/{row['a_n_models']}{fail_sfx}"
                work_text = model_text
        else:
            work_text = str(row["b_n_sequences"]) if row["b_n_sequences"] > 0 else "-"

        out.append({
            "stage_name": row["stage_name"],
            "jobid": row["jobid"],
            "host": host,
            "state": state,
            "shards": shard_text,
            "work": work_text,
            "now": stage_mix,
            "elapsed": elapsed,
            "started_at_min": row["started_at_min"] or "",
        })

    def _started_ts(row: dict) -> float:
        dt = _parse_iso(row.get("started_at_min"))
        if dt is None:
            return float("-inf")
        return dt.timestamp()

    # Most recently started jobs first.
    out.sort(key=lambda r: (-_started_ts(r), _state_order(r["state"]), r["stage_name"], r["jobid"]))
    return out


def _stage_a_totals(stage_a: List[dict]) -> str:
    n_total = len(stage_a)
    n_done = sum(1 for s in stage_a if s["status"] == "done")
    n_running = sum(1 for s in stage_a if s["status"] == "running")
    n_pending = sum(1 for s in stage_a if s["status"] == "pending")

    total_models = sum(s["n_models"] for s in stage_a)
    total_inputs = sum(s["n_inputs"] for s in stage_a)
    done_inputs = sum(s["n_inputs"] for s in stage_a if s["status"] == "done")
    fail = sum(s["prep_fail"] for s in stage_a)

    msg = f"Stage A: {n_done} done, {n_running} running, {n_pending} pending ({n_total} shards)"
    if total_inputs > 0:
        pct = 100.0 * done_inputs / total_inputs
        msg += f" | PDBs {done_inputs}/{total_inputs} ({pct:.1f}%)"
    if total_models > 0:
        done_models = sum(s["prep_ok"] for s in stage_a if s["status"] == "done")
        msg += f" | models(ok) {done_models}/{total_models}"
        if fail:
            msg += f" fail={fail}"
    return msg


def _stage_b_totals(stage_b: List[dict]) -> str:
    n_total = len(stage_b)
    n_done = sum(1 for s in stage_b if s["status"] == "done")
    n_running = sum(1 for s in stage_b if s["status"] == "running")
    n_failed = sum(1 for s in stage_b if s["status"] == "failed")
    n_pending = sum(1 for s in stage_b if s["status"] == "pending")
    n_waiting = sum(1 for s in stage_b if s["status"] == "waiting_A")

    msg = (
        f"Stage B: {n_done} done, {n_running} running, {n_pending} pending, "
        f"{n_waiting} waiting_A ({n_total} shards)"
    )
    if n_failed:
        msg += f" | FAILED={n_failed}"

    total_seqs = sum(s["n_sequences"] for s in stage_b)
    done_seqs = sum(s["n_sequences"] for s in stage_b if s["status"] == "done")
    if total_seqs > 0:
        pct = 100.0 * done_seqs / total_seqs
        msg += f" | seq {done_seqs}/{total_seqs} ({pct:.1f}%)"
    return msg


def print_table(run_root: Path, stage_a: List[dict], stage_b: List[dict], rows: List[dict]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Stage A+B live progress   {now}   RUN_ROOT: {run_root}")
    print("Refresh: every 5s (Ctrl+C to stop)")
    print()

    hdr = (
        f"{'Stage':<6}{'JobID':<16}{'Host':<18}{'State':<10}"
        f"{'Shards':<10}{'Work':<28}{'Now':<20}{'Elapsed':<10}"
    )
    print(hdr)
    print("-" * len(hdr))

    visible = rows[:MAX_VISIBLE_ROWS]
    hidden = rows[MAX_VISIBLE_ROWS:]

    if not visible:
        print("(no active job rows yet)")
    else:
        for r in visible:
            print(
                f"{r['stage_name']:<6}{r['jobid']:<16}{r['host']:<18}{r['state']:<10}"
                f"{r['shards']:<10}{r['work'][:27]:<28}{r['now'][:19]:<20}{r['elapsed']:<10}"
            )
        if hidden:
            hidden_counts: Dict[str, int] = {}
            for r in hidden:
                key = f"{r['stage_name']}:{r['state']}"
                hidden_counts[key] = hidden_counts.get(key, 0) + 1
            parts = [f"{k}={v}" for k, v in sorted(hidden_counts.items(), key=lambda x: (-x[1], x[0]))]
            print(f"... {len(hidden)} more job rows not shown ({', '.join(parts)})")

    print()
    print(_stage_a_totals(stage_a))
    print(_stage_b_totals(stage_b))
    print(_collect_stage_c_status(run_root))
    final_line = _pipeline_final_line(run_root, stage_a, stage_b)
    if final_line:
        print(final_line)


def usage() -> None:
    print("Usage: python3 scripts/progress_live.py RUN_ROOT", file=sys.stderr)
    print("   or: scripts/stageA_progress_live.py RUN_ROOT", file=sys.stderr)
    print("   or: scripts/stageB_progress_live.py RUN_ROOT", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        usage()
        return 2

    run_root = Path(args[0]).resolve()
    if not run_root.is_dir():
        print(f"ERROR: {run_root} is not a directory", file=sys.stderr)
        return 1

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            stage_a = collect_stage_a(run_root)
            stage_b = collect_stage_b(run_root)
            rows = aggregate_job_rows(stage_a, stage_b)
            print_table(run_root, stage_a, stage_b, rows)
            time.sleep(5)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
