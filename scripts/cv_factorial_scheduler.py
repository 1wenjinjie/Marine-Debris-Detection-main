"""CPU-only scheduling and filesystem coordination for the factorial experiment."""
from __future__ import annotations

import json
import errno
import multiprocessing
import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path


class LockBusy(RuntimeError):
    pass


@contextmanager
def file_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        raise LockBusy(f"Lock is held: {path}") from exc
                    time.sleep(0.1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise LockBusy(f"Lock is held: {path}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def write_consistent(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Existing experiment input differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def process_commands():
    if os.name == "nt":
        command = ("@(Get-CimInstance Win32_Process -ErrorAction Stop | "
                   "Where-Object { $_.Name -match '^python(w|[0-9.]*)?\\.exe$' } | "
                   "Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                check=True, capture_output=True, text=True, timeout=30)
        records = json.loads(output.stdout or "[]")
        return records if isinstance(records, list) else [records]
    records = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            records.append({"ProcessId": int(proc.name), "CommandLine": " ".join(args)})
        except (OSError, UnicodeError):
            continue
    return records


def reject_other_schedulers(records=None):
    if records is None:
        records = process_commands()
    others = [int(p["ProcessId"]) for p in records
              if int(p["ProcessId"]) != os.getpid()
              and re.search(r'(?:^|[\\/\s"\'])v2_cv_factorial\.py(?:[\s"\']|$)',
                            p.get("CommandLine") or "", re.IGNORECASE)]
    if others:
        raise RuntimeError(f"Another factorial scheduler is running (PID {others}). "
                           "Wait for its current training and OOF prediction to finish, then stop it "
                           "before starting this scheduler. Legacy processes do not use run locks.")


def build_tasks(designs, cells, seeds):
    return [(d, cell, seed) for seed in dict.fromkeys(seeds) for d in designs for cell in cells]


def run_locked(lock_path, run_dir, oof_path, action, skip_completed=True):
    try:
        with file_lock(lock_path):
            if skip_completed and Path(oof_path).is_file() and (Path(run_dir) / "weights" / "best.pt").is_file():
                return "skipped"
            action()
            if not Path(oof_path).is_file() or not (Path(run_dir) / "weights" / "best.pt").is_file():
                raise RuntimeError(f"Run did not produce checkpoint and OOF: {run_dir}")
            return "completed"
    except LockBusy:
        return "busy"


def schedule(tasks, worker, jobs, initializer=None, initargs=()):
    counts = Counter()
    failures = []
    remaining = iter(tasks)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initializer, initargs=initargs) as pool:
        futures = {}
        for _ in range(jobs):
            task = next(remaining, None)
            if task is not None:
                futures[pool.submit(worker, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                try:
                    status = future.result()
                    if status not in ("completed", "skipped", "busy"):
                        raise RuntimeError(f"Unknown task status: {status}")
                    counts[status] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    failures.append(str(exc))
                print(f"[progress] {sum(counts.values())}/{len(tasks)} {dict(counts)}", flush=True)
            if not failures and not counts["busy"]:
                for _ in done:
                    task = next(remaining, None)
                    if task is not None:
                        futures[pool.submit(worker, task)] = task
    if failures or counts["busy"]:
        raise RuntimeError(f"Incomplete training: {dict(counts)}; failures={failures[:3]}")
    return counts
