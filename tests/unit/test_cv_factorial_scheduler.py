import contextlib
import io
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import v2_cv_factorial as factorial
from scripts.cv_factorial_scheduler import (
    build_tasks, file_lock, reject_other_schedulers, run_locked, schedule, write_consistent,
)


def hold_lock(path, ready, release, crash=False):
    with file_lock(path):
        ready.set()
        release.wait(10)
        if crash:
            os._exit(7)


def fake_task(task):
    root, name, mode = task
    root = Path(root)
    (root / f"{name}.started").touch()
    if mode == "fail":
        raise RuntimeError("fake training failed")
    if mode == "crash":
        os._exit(9)
    if mode == "busy":
        return "busy"
    if mode in ("parallel", "fail_with_peer", "finish_after_failure"):
        deadline = time.monotonic() + 10
        while len(list(root.glob("*.started"))) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("second task never ran concurrently")
            time.sleep(0.01)
    if mode == "fail_with_peer":
        (root / "failure.ready").touch()
        raise RuntimeError("fake training failed with peer running")
    if mode == "finish_after_failure":
        deadline = time.monotonic() + 10
        while not (root / "failure.ready").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("peer failure never arrived")
            time.sleep(0.01)
        time.sleep(0.2)
    (root / f"{name}.done").touch()
    return "completed"


def child_marker(path):
    Path(path).touch()


def fake_nested_task(task):
    child = multiprocessing.get_context("spawn").Process(target=child_marker, args=(task,))
    child.start()
    child.join(10)
    if child.is_alive():
        child.terminate()
        child.join()
        raise RuntimeError("nested spawn timed out")
    if child.exitcode != 0:
        raise RuntimeError(f"nested spawn failed: {child.exitcode}")
    return "completed"


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctx = multiprocessing.get_context("spawn")

    def start_holder(self, crash=False):
        ready, release = self.ctx.Event(), self.ctx.Event()
        process = self.ctx.Process(target=hold_lock, args=(str(self.root / "run.lock"), ready, release, crash))
        process.start()
        self.addCleanup(lambda: process.terminate() if process.is_alive() else None)
        self.addCleanup(lambda: release.set())
        self.assertTrue(ready.wait(10))
        return process, release

    def completed_run(self):
        run = self.root / "run"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"checkpoint")
        oof = self.root / "oof.json"
        oof.write_text("[]")
        return run, oof

    def test_default_matrix_and_seed_filter(self):
        args = factorial.parse_args([])
        designs = [{"fold": k} for k in range(9)]
        self.assertEqual(len(build_tasks(designs, factorial.CELLS, args.seed)), 108)
        args = factorial.parse_args(["--seed", "2", "--jobs", "2"])
        tasks = build_tasks(designs, factorial.CELLS, args.seed)
        self.assertEqual(len(tasks), 36)
        self.assertEqual({task[2] for task in tasks}, {2})
        self.assertEqual(args.jobs, 2)
        self.assertEqual(len(build_tasks(designs, factorial.CELLS, [2, 2])), 36)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            factorial.parse_args(["--seed", "3"])

    def test_import_keeps_gpu_runtime_unloaded(self):
        self.assertNotIn("rex", vars(factorial))
        self.assertNotIn("cvp", vars(factorial))

    def test_different_tasks_really_run_concurrently(self):
        tasks = [(str(self.root), str(i), "parallel") for i in range(2)]
        counts = schedule(tasks, fake_task, 2)
        self.assertEqual(counts["completed"], 2)
        self.assertEqual(len(list(self.root.glob("*.done"))), 2)

    def test_cross_process_same_run_is_busy_even_if_complete(self):
        process, release = self.start_holder()
        run, oof = self.completed_run()
        status = run_locked(self.root / "run.lock", run, oof,
                            lambda: self.fail("must not run while held"))
        self.assertEqual(status, "busy")
        release.set()
        process.join(10)
        self.assertEqual(process.exitcode, 0)

    def test_exception_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, "fake"):
            with file_lock(self.root / "run.lock"):
                raise RuntimeError("fake")
        with file_lock(self.root / "run.lock"):
            pass

    def test_crashed_process_releases_lock(self):
        process, release = self.start_holder(crash=True)
        release.set()
        process.join(10)
        self.assertEqual(process.exitcode, 7)
        with file_lock(self.root / "run.lock"):
            pass

    def test_completed_run_is_skipped_and_preserved(self):
        run, oof = self.completed_run()
        before = (oof.stat().st_mtime_ns, (run / "weights" / "best.pt").read_bytes())
        self.assertEqual(run_locked(self.root / "run.lock", run, oof,
                                   lambda: self.fail("complete run reran")), "skipped")
        self.assertEqual(before, (oof.stat().st_mtime_ns, (run / "weights" / "best.pt").read_bytes()))

    def test_incomplete_run_retrains_and_removes_old_output(self):
        d = {"fold": 0}
        run = self.root / "fold0_clean_ndwi_s0"
        run.mkdir()
        (run / "stale").write_text("partial")
        oof_dir = self.root / "oof"
        oof_dir.mkdir()
        oof = oof_dir / "fold0_clean_ndwi_s0_oof.json"
        oof.write_text("stale")

        class Trainer:
            def __init__(self, overrides):
                self.overrides = overrides

            def train(inner):
                self.assertFalse((run / "stale").exists())
                self.assertFalse(oof.exists())
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").write_bytes(b"new checkpoint")

        runtime = SimpleNamespace(LOCKED_HYPERPARAMS={"imgsz": 512},
                                  _ensure_initialization=lambda: {7: "init.pt"}, MPBTrainer=Trainer,
                                  collect_predictions=lambda *args, **kwargs: [],
                                  _write_prediction_ledger=lambda path, entries: path.write_text(json.dumps(entries)))
        with patch.multiple(factorial, RUNS=self.root, OOF_DIR=oof_dir, rex=runtime, create=True), \
                patch.object(factorial, "prepare_cell", return_value=(self.root, {})):
            self.assertEqual(factorial.run_cell(d, "clean_ndwi", 0, "0"), "completed")
        self.assertEqual(oof.read_text(), "[]")
        self.assertEqual((run / "weights" / "best.pt").read_bytes(), b"new checkpoint")

    def test_missing_outputs_cannot_be_reported_complete(self):
        with self.assertRaisesRegex(RuntimeError, "did not produce"):
            run_locked(self.root / "run.lock", self.root / "run", self.root / "oof.json", lambda: None)

    def test_failure_stops_submitting_more_tasks(self):
        tasks = [(str(self.root), "first", "fail"), (str(self.root), "later", "ok")]
        with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
            schedule(tasks, fake_task, 1)
        self.assertFalse((self.root / "later.started").exists())

    def test_failure_allows_inflight_peer_to_finish(self):
        tasks = [(str(self.root), "first", "fail_with_peer"),
                 (str(self.root), "peer", "finish_after_failure"),
                 (str(self.root), "later", "ok")]
        with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
            schedule(tasks, fake_task, 2)
        self.assertTrue((self.root / "peer.done").is_file())
        self.assertFalse((self.root / "later.started").exists())

    def test_pool_workers_can_spawn_dataloader_children(self):
        paths = [str(self.root / f"nested{i}.done") for i in range(2)]
        counts = schedule(paths, fake_nested_task, 2)
        self.assertEqual(counts["completed"], 2)
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_busy_is_incomplete_and_stops_more_tasks(self):
        tasks = [(str(self.root), "first", "busy"), (str(self.root), "later", "ok")]
        with self.assertRaisesRegex(RuntimeError, "busy"):
            schedule(tasks, fake_task, 1)
        self.assertFalse((self.root / "later.started").exists())

    def test_worker_crash_is_incomplete(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete training"):
            schedule([(str(self.root), "crashed", "crash")], fake_task, 1)

    def test_repeated_cell_preparation_preserves_inputs(self):
        image = self.root / "image.npy"
        image.touch()
        d = {"fold": 0, "train": {"clean": ["a"]}, "val": ["b"], "panel": ["c"]}
        cvp = SimpleNamespace(NAMES={0: "debris"})
        with patch.multiple(factorial, FACT_ROOT=self.root, cvp=cvp, create=True), \
                patch.object(factorial, "cell_image", return_value=image), \
                patch.object(factorial, "fold_stats", return_value={"p01": [0], "p99": [1]}):
            folder, counts = factorial.prepare_cell(d, "clean_ndwi")
            before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
            self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})
            factorial.prepare_cell(d, "clean_ndwi")
            after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
        self.assertEqual(before, after)

    def test_mismatched_input_is_rejected(self):
        path = self.root / "manifest.txt"
        write_consistent(path, "original")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            write_consistent(path, "changed")
        self.assertEqual(path.read_text(), "original")

    def test_old_process_is_rejected_and_self_ignored(self):
        own = {"ProcessId": os.getpid(), "CommandLine": 'python "scripts/v2_cv_factorial.py" --jobs 2'}
        reject_other_schedulers([own])
        old = {"ProcessId": 44636, "CommandLine": 'python "D:\\repo\\scripts\\v2_cv_factorial.py" --device 0'}
        with self.assertRaisesRegex(RuntimeError, "44636"):
            reject_other_schedulers([own, old])
        reject_other_schedulers([{"ProcessId": 1, "CommandLine": "python v2_cv_factorial.py.backup"}])

    def test_analysis_still_requires_all_three_seeds(self):
        with self.assertRaisesRegex(RuntimeError, "missing 108"):
            factorial.load_oof([{} for _ in range(9)], self.root, range(9), factorial.SEEDS)


if __name__ == "__main__":
    unittest.main()
