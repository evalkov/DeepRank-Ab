#!/usr/bin/env python3
"""
collect_compute_metrics.py

Background compute-metrics collector for Slurm jobs.

Uses:
  - nvidia-smi: GPU device metrics + pmon per-process stream
  - psutil (preferred): CPU/mem/disk/net + optional per-core + optional per-process

Writes into --outdir (per --prefix):
  gpu_metrics_<prefix>.csv          (GPU device metrics: util/mem/power/temp/clocks/pcie)
  gpu_pmon_<prefix>.log             (nvidia-smi pmon stream, -o DT)
  sys_metrics_<prefix>.csv          (cpu_total, cpu_iowait, loadavg, mem/swap)
  cpu_percore_<prefix>.csv          (optional; per-core CPU%)
  disk_io_<prefix>.csv              (disk MB/s + IOPS, aggregated across devices)
  net_io_<prefix>.csv               (optional; net MB/s)
  proc_metrics_<prefix>.csv         (optional; if --pid provided: job RSS/CPU%%)
  snapshot_<prefix>.txt             (nvidia-smi -q snapshot)
  env_<prefix>.txt                  (environment variables)

Hardening vs previous version:
  - line-buffered file handles (buffering=1)
  - explicit flush after each write cycle
  - robust shutdown (SIGTERM/SIGINT) with finally close/terminate

Runs until SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import signal
import subprocess
import time
from typing import Optional, List, Tuple, Dict, Any, Set

STOP = False


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    print(f"[metrics] {now_iso()} {msg}", flush=True)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def dump_env(path: str) -> None:
    try:
        with open(path, "w") as f:
            for k in sorted(os.environ.keys()):
                f.write(f"{k}={os.environ[k]}\n")
    except Exception:
        pass


def run_snapshot(path: str) -> None:
    if not have("nvidia-smi"):
        return
    try:
        with open(path, "w") as f:
            subprocess.run(
                ["nvidia-smi", "-q"],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except Exception:
        pass


def start_pmon(out_path: str, interval_s: int, gpus: str) -> Tuple[Optional[subprocess.Popen], Optional[Any]]:
    """
    Start:
      nvidia-smi pmon -i 0,1,2,3 -d <interval> -s um -o DT

    Output is a *log* (fixed-width whitespace), not true CSV.

    Returns (Popen, file_handle) so we can close the handle explicitly at shutdown.
    """
    if not have("nvidia-smi"):
        return None, None
    args = ["nvidia-smi", "pmon", "-i", gpus, "-d", str(interval_s), "-s", "um", "-o", "DT"]
    try:
        fh = open(out_path, "w", buffering=1)  # line-buffered
        proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.DEVNULL, text=True)
        return proc, fh
    except Exception:
        return None, None


def query_gpus() -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Poll GPU device metrics using:
      nvidia-smi --query-gpu=... --format=csv,noheader,nounits
    """
    if not have("nvidia-smi"):
        return None, None

    fields = [
        "index",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "pstate",
        "pcie.link.gen.current",
        "pcie.link.width.current",
        "utilization.encoder",
        "utilization.decoder",
    ]
    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        return None, None

    header = (
        "timestamp,gpu,"
        "util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w,temp_c,"
        "clock_sm_mhz,clock_mem_mhz,pstate,pcie_gen,pcie_width,util_enc_pct,util_dec_pct"
    )

    ts = now_iso()
    rows: List[str] = []
    for ln in cp.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < len(fields):
            continue
        gpu = parts[0]
        rest = parts[1:]
        rows.append(f"{ts},{gpu}," + ",".join(rest))
    return header, rows


def handler(signum, frame):
    global STOP
    STOP = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--prefix", default="job")
    ap.add_argument("--percore", action="store_true")
    ap.add_argument("--net", action="store_true")
    ap.add_argument("--pid", type=int, default=0, help="optional PID to track (RSS/CPU%%)")
    ap.add_argument("--pid-tree", action="store_true", help="track --pid and descendants (aggregate CPU/RSS)")
    ap.add_argument("--heartbeat-s", type=float, default=30.0, help="monitor log heartbeat interval in seconds (0=off)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    prefix = args.prefix
    outdir = args.outdir

    gpu_csv = os.path.join(outdir, f"gpu_metrics_{prefix}.csv")
    pmon_log = os.path.join(outdir, f"gpu_pmon_{prefix}.log")
    sys_csv = os.path.join(outdir, f"sys_metrics_{prefix}.csv")
    percore_csv = os.path.join(outdir, f"cpu_percore_{prefix}.csv")
    disk_csv = os.path.join(outdir, f"disk_io_{prefix}.csv")
    net_csv = os.path.join(outdir, f"net_io_{prefix}.csv")
    proc_csv = os.path.join(outdir, f"proc_metrics_{prefix}.csv")
    snap_txt = os.path.join(outdir, f"snapshot_{prefix}.txt")
    env_txt = os.path.join(outdir, f"env_{prefix}.txt")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    log(
        "starting collector "
        f"prefix={prefix} interval={args.interval}s percore={int(args.percore)} "
        f"net={int(args.net)} pid={args.pid or '-'} pid_tree={int(args.pid_tree)} gpus={args.gpus}"
    )

    run_snapshot(snap_txt)
    dump_env(env_txt)

    # Start pmon stream
    pmon_proc, pmon_fh = start_pmon(pmon_log, max(1, int(round(args.interval))), args.gpus)

    # psutil is available on your cluster; keep safe fallback anyway
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
        log("WARNING: psutil unavailable; CPU/system/process metrics disabled")

    gpu_header_written = False

    # CSV writers / fields
    sys_fields = [
        "timestamp",
        "cpu_total_pct",
        "cpu_iowait_pct",
        "load1",
        "load5",
        "load15",
        "mem_used_mib",
        "mem_avail_mib",
        "swap_used_mib",
        "swap_free_mib",
    ]
    disk_fields = ["timestamp", "disk_read_MBps", "disk_write_MBps", "disk_read_iops", "disk_write_iops"]
    net_fields = ["timestamp", "net_rx_MBps", "net_tx_MBps"]
    proc_fields = ["timestamp", "pid", "proc_nprocs", "proc_cpu_pct", "proc_rss_mib", "proc_vms_mib"]

    proc = None
    primed_proc_pids: Set[int] = set()
    if psutil is not None:
        # Prime percent counters (psutil requires an initial call)
        psutil.cpu_percent(interval=None)
        psutil.cpu_times_percent(interval=None)
        if args.pid:
            try:
                proc = psutil.Process(args.pid)
                proc.cpu_percent(interval=None)
                primed_proc_pids.add(proc.pid)
                log(f"tracking process pid={proc.pid} pid_tree={int(args.pid_tree)}")
            except Exception:
                proc = None
                log(f"WARNING: requested pid={args.pid} is not available")

    prev_disk = psutil.disk_io_counters(perdisk=True) if psutil is not None else None
    prev_net = psutil.net_io_counters(pernic=False) if (psutil is not None and args.net) else None
    t_prev = time.time()

    # Line-buffered file handles to minimise header-only files on abrupt termination
    f_sys = open(sys_csv, "w", newline="", buffering=1)
    f_disk = open(disk_csv, "w", newline="", buffering=1)
    w_sys = csv.DictWriter(f_sys, fieldnames=sys_fields)
    w_disk = csv.DictWriter(f_disk, fieldnames=disk_fields)
    w_sys.writeheader()
    w_disk.writeheader()
    f_sys.flush()
    f_disk.flush()

    f_pc = None
    w_pc = None
    if args.percore and psutil is not None:
        ncpu = psutil.cpu_count(logical=True) or 0
        pc_fields = ["timestamp"] + [f"cpu{i}_pct" for i in range(ncpu)]
        f_pc = open(percore_csv, "w", newline="", buffering=1)
        w_pc = csv.DictWriter(f_pc, fieldnames=pc_fields)
        w_pc.writeheader()
        f_pc.flush()

    f_net = None
    w_net = None
    if args.net and psutil is not None:
        f_net = open(net_csv, "w", newline="", buffering=1)
        w_net = csv.DictWriter(f_net, fieldnames=net_fields)
        w_net.writeheader()
        f_net.flush()

    f_proc = None
    w_proc = None
    if proc is not None:
        f_proc = open(proc_csv, "w", newline="", buffering=1)
        w_proc = csv.DictWriter(f_proc, fieldnames=proc_fields)
        w_proc.writeheader()
        f_proc.flush()

    heartbeat_s = max(0.0, float(args.heartbeat_s))
    last_heartbeat_t = time.time()
    sample_count = 0
    last_sys_cpu = 0.0
    last_proc_cpu = 0.0
    last_proc_n = 0

    try:
        while not STOP:
            loop_t0 = time.time()
            dt_s = max(1e-6, loop_t0 - t_prev)
            t_prev = loop_t0
            ts = now_iso()
            sample_count += 1

            # ----------------------------
            # GPU device poll
            # ----------------------------
            hdr, rows = query_gpus()
            if hdr and rows:
                # GPU file is written via open/append to avoid buffering surprises.
                mode = "a" if os.path.exists(gpu_csv) else "w"
                with open(gpu_csv, mode, buffering=1) as f_gpu:
                    if mode == "w" or not gpu_header_written:
                        f_gpu.write(hdr + "\n")
                        gpu_header_written = True
                    for r in rows:
                        f_gpu.write(r + "\n")

            # ----------------------------
            # System counters (psutil path)
            # ----------------------------
            if psutil is not None:
                cpu_total = psutil.cpu_percent(interval=None)
                last_sys_cpu = float(cpu_total)
                cpu_times = psutil.cpu_times_percent(interval=None)
                iowait = float(getattr(cpu_times, "iowait", 0.0))

                try:
                    l1, l5, l15 = os.getloadavg()
                except Exception:
                    l1 = l5 = l15 = 0.0

                vm = psutil.virtual_memory()
                sm = psutil.swap_memory()

                w_sys.writerow({
                    "timestamp": ts,
                    "cpu_total_pct": f"{cpu_total:.3f}",
                    "cpu_iowait_pct": f"{iowait:.3f}",
                    "load1": f"{l1:.3f}",
                    "load5": f"{l5:.3f}",
                    "load15": f"{l15:.3f}",
                    "mem_used_mib": f"{vm.used / (1024**2):.1f}",
                    "mem_avail_mib": f"{vm.available / (1024**2):.1f}",
                    "swap_used_mib": f"{sm.used / (1024**2):.1f}",
                    "swap_free_mib": f"{sm.free / (1024**2):.1f}",
                })
                f_sys.flush()

                # Per-core CPU%
                if w_pc is not None and f_pc is not None:
                    per = psutil.cpu_percent(interval=None, percpu=True)
                    row: Dict[str, Any] = {"timestamp": ts}
                    for i, v in enumerate(per):
                        row[f"cpu{i}_pct"] = float(v)
                    w_pc.writerow(row)
                    f_pc.flush()

                # Disk I/O delta rates
                cur_disk = psutil.disk_io_counters(perdisk=True)
                if prev_disk is not None and cur_disk is not None:
                    rb = wb = rc = wc = 0
                    for dev, v in cur_disk.items():
                        pv = prev_disk.get(dev)
                        if pv is None:
                            continue
                        rb += max(0, v.read_bytes - pv.read_bytes)
                        wb += max(0, v.write_bytes - pv.write_bytes)
                        rc += max(0, v.read_count - pv.read_count)
                        wc += max(0, v.write_count - pv.write_count)
                    w_disk.writerow({
                        "timestamp": ts,
                        "disk_read_MBps": f"{(rb/(1024**2))/dt_s:.3f}",
                        "disk_write_MBps": f"{(wb/(1024**2))/dt_s:.3f}",
                        "disk_read_iops": f"{rc/dt_s:.3f}",
                        "disk_write_iops": f"{wc/dt_s:.3f}",
                    })
                else:
                    w_disk.writerow({
                        "timestamp": ts,
                        "disk_read_MBps": "0",
                        "disk_write_MBps": "0",
                        "disk_read_iops": "0",
                        "disk_write_iops": "0",
                    })
                f_disk.flush()
                prev_disk = cur_disk

                # Net I/O delta rates
                if w_net is not None and f_net is not None:
                    cur_net = psutil.net_io_counters(pernic=False)
                    if prev_net is not None and cur_net is not None:
                        rx = max(0, cur_net.bytes_recv - prev_net.bytes_recv)
                        tx = max(0, cur_net.bytes_sent - prev_net.bytes_sent)
                        w_net.writerow({
                            "timestamp": ts,
                            "net_rx_MBps": f"{(rx/(1024**2))/dt_s:.3f}",
                            "net_tx_MBps": f"{(tx/(1024**2))/dt_s:.3f}",
                        })
                    else:
                        w_net.writerow({"timestamp": ts, "net_rx_MBps": "0", "net_tx_MBps": "0"})
                    f_net.flush()
                    prev_net = cur_net

                # Optional tracked PID
                if w_proc is not None and f_proc is not None and proc is not None:
                    try:
                        tracked: List[Any]
                        if args.pid_tree:
                            tracked = [proc]
                            try:
                                tracked.extend(proc.children(recursive=True))
                            except Exception:
                                pass
                        else:
                            tracked = [proc]

                        cpu_p = 0.0
                        rss_b = 0
                        vms_b = 0
                        alive_n = 0

                        for p in tracked:
                            try:
                                if not p.is_running():
                                    continue
                                alive_n += 1
                                if p.pid not in primed_proc_pids:
                                    p.cpu_percent(interval=None)
                                    primed_proc_pids.add(p.pid)
                                cpu_p += max(0.0, float(p.cpu_percent(interval=None)))
                                mi = p.memory_info()
                                rss_b += int(getattr(mi, "rss", 0))
                                vms_b += int(getattr(mi, "vms", 0))
                            except Exception:
                                continue

                        last_proc_cpu = cpu_p
                        last_proc_n = alive_n
                        w_proc.writerow({
                            "timestamp": ts,
                            "pid": str(proc.pid),
                            "proc_nprocs": str(alive_n),
                            "proc_cpu_pct": f"{cpu_p:.3f}",
                            "proc_rss_mib": f"{rss_b/(1024**2):.3f}",
                            "proc_vms_mib": f"{vms_b/(1024**2):.3f}",
                        })
                        f_proc.flush()
                    except Exception:
                        pass

            if heartbeat_s > 0.0 and (time.time() - last_heartbeat_t) >= heartbeat_s:
                if proc is not None:
                    log(
                        f"samples={sample_count} sys_cpu={last_sys_cpu:.1f}% "
                        f"proc_cpu={last_proc_cpu:.1f}% tracked_procs={last_proc_n}"
                    )
                else:
                    log(f"samples={sample_count} sys_cpu={last_sys_cpu:.1f}%")
                last_heartbeat_t = time.time()

            # sleep until next tick
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, args.interval - elapsed))

    finally:
        log("stopping collector")
        # Ensure files are flushed/closed even on SIGTERM
        for fh in (f_sys, f_disk, f_pc, f_net, f_proc):
            if fh is None:
                continue
            try:
                fh.flush()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass

        if pmon_proc is not None:
            try:
                pmon_proc.terminate()
            except Exception:
                pass
            try:
                pmon_proc.wait(timeout=2)
            except Exception:
                try:
                    pmon_proc.kill()
                except Exception:
                    pass

        if pmon_fh is not None:
            try:
                pmon_fh.flush()
            except Exception:
                pass
            try:
                pmon_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
