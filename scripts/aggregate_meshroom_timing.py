#!/usr/bin/env python3
"""
Aggregate per-node Meshroom timing from MeshroomCache/<Node>/<hash>/0.log files.

Looks for three markers per node log:
  '[Process chunk] elapsed time: H:MM:SS.fff'  (Meshroom wrapper, sparse)
  'Task done in (s): NNNN.fff'                  (AliceVision binary self-report)
  '[HH:MM:SS.ffffff]' first → last              (log-span proxy, always present)

Prints a markdown table sorted by Meshroom-defined pipeline order.

Usage:
  scripts/aggregate_meshroom_timing.py <MeshroomCache dir> [label]
"""
import os, re, sys, glob

PIPELINE_ORDER = [
    "CameraInit", "FeatureExtraction", "ImageMatching", "FeatureMatching",
    "StructureFromMotion", "PrepareDenseScene", "DepthMap", "DepthMapFilter",
    "Meshing", "MeshFiltering", "Texturing", "CopyFiles",
]

elapsed_re = re.compile(r"\[Process chunk\] elapsed time:\s+(\d+):(\d+):(\d+(?:\.\d+)?)")
task_re    = re.compile(r"Task done in \(s\):\s+([\d.]+)")
ts_re      = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)\]")

def collect(cache_dir):
    rows = {}
    for node in PIPELINE_ORDER:
        ndir = os.path.join(cache_dir, node)
        if not os.path.isdir(ndir):
            rows[node] = None
            continue
        chunks_elapsed = 0.0
        chunks_task    = 0.0
        chunks_ts      = 0.0
        chunks         = 0
        had_elapsed    = False
        had_ts         = False
        for sub in sorted(os.listdir(ndir)):
            log_glob = sorted(glob.glob(os.path.join(ndir, sub, "*.log")))
            for log in log_glob:
                try:
                    txt = open(log, "r", errors="replace").read()
                except OSError:
                    continue
                em = elapsed_re.search(txt)
                if em:
                    h, m, s = em.groups()
                    chunks_elapsed += int(h)*3600 + int(m)*60 + float(s)
                    had_elapsed = True
                for tm in task_re.finditer(txt):
                    chunks_task += float(tm.group(1))
                first = last = None
                for line in txt.splitlines():
                    mm = ts_re.match(line)
                    if mm:
                        h, m, s = mm.groups()
                        t = int(h)*3600 + int(m)*60 + float(s)
                        if first is None: first = t
                        last = t
                if first is not None and last is not None and last >= first:
                    chunks_ts += last - first
                    had_ts = True
                chunks += 1
        rows[node] = (
            chunks_elapsed if had_elapsed else None,
            chunks_task,
            chunks_ts if had_ts else None,
            chunks,
        )
    return rows

def fmt_sec(x):
    if x is None: return "n/a"
    if x < 60:    return f"{x:.1f}s"
    m, s = divmod(x, 60)
    if m < 60:    return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{m:02d}m{s:04.1f}s"

def main():
    if len(sys.argv) < 2:
        print("usage: aggregate_meshroom_timing.py <MeshroomCache dir> [label]")
        sys.exit(1)
    cache = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.dirname(cache))
    rows = collect(cache)
    print(f"\n## {label}")
    print()
    print("| Stage | Wrapper elapsed | AV `Task done` sum | Log-span (first->last) | Chunks |")
    print("|---|---|---|---|---|")
    total_wrap = 0.0
    total_task = 0.0
    total_ts   = 0.0
    for n in PIPELINE_ORDER:
        r = rows.get(n)
        if r is None:
            print(f"| {n} | (no node) | - | - | - |")
            continue
        elapsed, task, span, chunks = r
        if elapsed is not None:
            total_wrap += elapsed
        if span is not None:
            total_ts += span
        total_task += task
        print(f"| {n} | {fmt_sec(elapsed)} | {fmt_sec(task)} | {fmt_sec(span)} | {chunks} |")
    print(f"| **Sum** | **{fmt_sec(total_wrap)}** | **{fmt_sec(total_task)}** | **{fmt_sec(total_ts)}** | |")

if __name__ == "__main__":
    main()
