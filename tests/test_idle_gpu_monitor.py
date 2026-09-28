"""Consumer-observable lifecycle tests for scripts/wait_for_idle_gpu.py.

Every GPU observation in these tests is synthetic: a fake nvidia-smi on PATH
replays time-staged CSV scenarios, and the launched "job" is a short local
child process. No real GPU is ever queried.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait_for_idle_gpu.py"

UUID_A = "GPU-fakeaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-fakebbbb-1111-2222-3333-444444444444"

_spec = importlib.util.spec_from_file_location("wait_for_idle_gpu", SCRIPT)
WFI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WFI)


FAKE_NVIDIA_SMI = r"""#!/usr/bin/env python3
import json, os, sys, time

def start_time(state_path):
    if os.path.exists(state_path):
        with open(state_path) as handle:
            return json.load(handle)["start"]
    tmp = state_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"start": time.time()}, handle)
    os.replace(tmp, state_path)
    with open(state_path) as handle:
        return json.load(handle)["start"]

def main():
    state_path = os.environ["FAKE_GPU_STATE"]
    scenario_path = os.environ["FAKE_GPU_SCENARIO"]
    start = start_time(state_path)
    elapsed = time.time() - start
    # One line per invocation, prefixed with its offset from `start`, so a
    # test can see exactly when (and how often) the monitor observed.
    with open(state_path + ".calls", "a") as handle:
        handle.write("%.3f %s\n" % (elapsed, " ".join(sys.argv[1:])))
    with open(scenario_path) as handle:
        stages = json.load(handle)
    stage = stages[0]
    for candidate in stages:
        if elapsed >= candidate["after"]:
            stage = candidate
    if stage.get("exit_code"):
        sys.exit(stage["exit_code"])
    query = sys.argv[1] if sys.argv else ""
    rows = stage.get("apps", []) if "compute-apps" in query else stage.get("gpus", [])
    sys.stdout.write("".join(row + "\n" for row in rows))

main()
"""


def gpu_row(index, uuid, free_mib, utilization):
    return f"{index}, {uuid}, {free_mib}, {utilization}"


def app_row(uuid, pid):
    return f"{uuid}, {pid}"


def stage(after, gpus=(), apps=(), exit_code=None):
    return {"after": after, "gpus": list(gpus), "apps": list(apps),
            "exit_code": exit_code}


class ParserUnitTests(unittest.TestCase):
    """The CSV parsers must fail closed: any doubt means None."""


    def test_malformed_gpu_rows_fail_closed(self):
        for text in ("garbage",
                     "0, GPU-x, 48678",
                     "0, GPU-x, notanumber, 0",
                     "0, , 48678, 0",
                     "-1, GPU-x, 48678, 0",
                     f"0, {UUID_A}, 1, 0\n0, {UUID_B}, 2, 0",  # dup index
                     f"0, {UUID_A}, 1, 0\n1, {UUID_A}, 2, 0",  # dup uuid
                     ""):
            self.assertIsNone(WFI.parse_gpu_rows(text), text)


    def test_malformed_app_rows_fail_closed(self):
        for text in (UUID_A, "1234", f"{UUID_A}, notanumber"):
            self.assertIsNone(WFI.parse_app_rows(text), text)


class RecheckContinuityTests(unittest.TestCase):
    """A failed recheck must not leave the rest of a ready list launchable."""

    def check_requalification(self, *, observation_error):
        clock = [0.0]
        ready_pass_calls = [0]
        launched = []
        gpus = [
            {"index": 0, "uuid": UUID_A, "free_mib": 48000, "utilization": 0},
            {"index": 1, "uuid": UUID_B, "free_mib": 48000, "utilization": 0},
        ]
        monitor = SimpleNamespace(
            gpu_indices=[0, 1], min_free_mib=40000, max_utilization=5,
            idle_seconds=10, poll_seconds=1,
            publish=lambda **kwargs: None, log=lambda message: None,
        )

        def observe(*args):
            if clock[0] == 10:
                ready_pass_calls[0] += 1
                if ready_pass_calls[0] == 2:
                    if observation_error:
                        return {"gpus": None, "occupied": None, "error": "untrusted query"}
                    return {"gpus": gpus, "occupied": {UUID_A, UUID_B}, "error": None}
            return {"gpus": gpus, "occupied": set(), "error": None}

        def sleep(seconds):
            clock[0] += seconds
            if clock[0] > 30:
                self.fail("no launch after requalification")

        def launch(_monitor, gpu):
            launched.append((gpu["uuid"], clock[0]))
            return 0

        with patch.object(WFI, "observe", side_effect=observe), \
             patch.object(WFI.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(WFI, "sleep_interruptible", side_effect=sleep), \
             patch.object(WFI, "acquire_uuid_lock", return_value=object()), \
             patch.object(WFI, "release_lock"), \
             patch.object(WFI, "launch", side_effect=launch):
            self.assertEqual(WFI.wait_and_launch(monitor, Path("/unused")), 0)
        self.assertEqual(len(launched), 1)
        # Failure/busy was observed at t=10. First fresh idle observation is
        # t=11, so neither stale candidate can launch before t=21.
        self.assertGreaterEqual(launched[0][1], 21)

    def test_failed_recheck_invalidates_remaining_ready_candidates(self):
        self.check_requalification(observation_error=True)

    def test_recheck_busy_blip_invalidates_other_candidate_window(self):
        self.check_requalification(observation_error=False)


class WatcherLifecycleTests(unittest.TestCase):
    """End-to-end runs against the fake nvidia-smi on PATH."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        fake = self.bin_dir / "nvidia-smi"
        fake.write_text(FAKE_NVIDIA_SMI)
        fake.chmod(0o755)
        self.scenario_path = self.root / "scenario.json"
        self.state_path = self.root / "fake-gpu-state.json"
        self.state_dir = self.root / "state"
        self.lock_dir = self.root / "locks"
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if proc.stdout is not None:
                proc.stdout.close()
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------

    def set_scenario(self, stages):
        self.scenario_path.write_text(json.dumps(stages))

    def env(self):
        copy = os.environ.copy()
        copy["PATH"] = f"{self.bin_dir}{os.pathsep}{copy.get('PATH', '')}"
        copy["FAKE_GPU_SCENARIO"] = str(self.scenario_path)
        copy["FAKE_GPU_STATE"] = str(self.state_path)
        return copy

    def base_args(self, *, idle=0.6, poll=0.1, initial=0.0, indices=None):
        args = ["--state-dir", str(self.state_dir),
                "--lock-dir", str(self.lock_dir),
                "--poll-seconds", str(poll),
                "--idle-seconds", str(idle),
                "--initial-delay", str(initial),
                "--min-free-mib", "40000",
                "--max-utilization", "5"]
        if indices is not None:
            args += ["--gpu-indices", indices]
        return args

    def popen(self, command, extra=()):
        argv = [sys.executable, str(SCRIPT), *self.base_args(), *extra,
                "--", *command]
        proc = subprocess.Popen(argv, env=self.env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        self.procs.append(proc)
        return proc

    def read_status(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                return json.loads((self.state_dir / "status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
        self.fail("status.json never appeared")

    def wait_phase(self, proc, phase, timeout=20):
        deadline = time.monotonic() + timeout
        status = self.read_status()
        while time.monotonic() < deadline:
            status = self.read_status()
            if status.get("phase") == phase:
                return status
            if proc.poll() is not None:
                self.fail(f"watcher exited rc={proc.returncode} before "
                          f"phase {phase!r}; last status={status}")
            time.sleep(0.05)
        self.fail(f"phase {phase!r} not reached; last={status}")

    def marker_command(self, marker, envfile=None, extra=""):
        envfile = envfile or (self.root / "cuda-env.txt")
        code = (f"import os, time;"
                f"open({str(marker)!r}, 'a').write(str(os.getpid()) + chr(10));"
                f"open({str(envfile)!r}, 'w').write("
                f"os.environ.get('CUDA_VISIBLE_DEVICES', ''));{extra}")
        return [sys.executable, "-c", code]

    def fake_start(self):
        return json.loads(self.state_path.read_text())["start"]

    def observation_starts(self):
        """Offsets (seconds from the fake's start) of each observation.

        Every observation runs the GPU query first, so each even-indexed
        line of the call log marks one observation.
        """
        calls_path = Path(str(self.state_path) + ".calls")
        try:
            lines = calls_path.read_text().splitlines()
        except FileNotFoundError:
            return []
        return [float(line.split()[0]) for line in lines[0::2]]

    def probe_command(self, record):
        """A child that reports the GPU it was given and whether the
        monitor's own per-GPU advisory lock was held while it ran."""
        script = self.root / "probe.py"
        script.write_text(
            "import fcntl, importlib.util, json, os, sys\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('wfi', sys.argv[1])\n"
            "wfi = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(wfi)\n"
            "uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')\n"
            "handle = open(wfi._uuid_lock_path(Path(sys.argv[2]), uuid), 'w')\n"
            "free = True\n"
            "try:\n"
            "    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "except OSError:\n"
            "    free = False\n"
            "with open(sys.argv[3], 'w') as out:\n"
            "    json.dump({'cuda': uuid, 'lock_free': free}, out)\n")
        return [sys.executable, str(script), str(SCRIPT), str(self.lock_dir),
                str(record)]

    # -- lifecycle ---------------------------------------------------------

    def test_stable_idle_launches_once_with_uuid_and_completes(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker))
        status = self.wait_phase(proc, "completed")
        self.assertEqual(proc.wait(timeout=10), 0)
        self.assertEqual(status["returncode"], 0)
        self.assertEqual(status["selected"]["uuid"], UUID_A)
        self.assertEqual(status["selected"]["cuda_visible_devices"], UUID_A)
        self.assertEqual((self.root / "cuda-env.txt").read_text(), UUID_A)
        self.assertEqual(len(marker.read_text().splitlines()), 1)

    def test_index_filter_selects_requested_gpu(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0),
                                          gpu_row(1, UUID_B, 48000, 0)])])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker), extra=["--gpu-indices", "1"])
        self.wait_phase(proc, "completed")
        self.assertEqual(proc.wait(timeout=10), 0)
        self.assertEqual((self.root / "cuda-env.txt").read_text(), UUID_B)
        self.assertEqual(len(marker.read_text().splitlines()), 1)

    def test_busy_midwindow_resets_idle_window(self):
        # Idle 0-0.8s, busy 0.8-1.2s, idle again after 1.2s. With a 1.0s
        # idle window the launch cannot happen before ~2.2s; an
        # implementation that fails to reset on busy would launch ~1.2s.
        self.set_scenario([
            stage(0.0, gpus=[gpu_row(0, UUID_A, 48000, 0)]),
            stage(0.8, gpus=[gpu_row(0, UUID_A, 48000, 90)]),
            stage(1.2, gpus=[gpu_row(0, UUID_A, 48000, 0)]),
        ])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker, extra="time.sleep(1.5)"),
                          extra=["--idle-seconds", "1.0"])
        self.wait_phase(proc, "completed", timeout=25)
        self.assertEqual(proc.wait(timeout=10), 0)
        launch_elapsed = marker.stat().st_mtime - self.fake_start()
        self.assertGreaterEqual(launch_elapsed, 2.0,
                                "launched without a full post-busy idle window")
        self.assertLess(launch_elapsed, 8.0)

    def test_untrusted_observations_never_launch(self):
        never_idle = [
            ("malformed-gpu-row", [stage(0, gpus=["garbage"])]),
            ("gpu-query-fails", [stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)],
                                      exit_code=1)]),
            ("empty-gpu-output", [stage(0)]),
            ("malformed-app-row", [stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)],
                                        apps=[UUID_A])]),
            ("occupied-by-compute-app",
             [stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)],
                    apps=[app_row(UUID_A, 777)])]),
            ("unknown-app-gpu",
             [stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)],
                    apps=[app_row(UUID_B, 777)])]),
            ("occupied-with-empty-pid",
             [stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)],
                    apps=[f"{UUID_A}, "])]),
        ]
        for name, stages in never_idle:
            with self.subTest(case=name):
                # Fresh dirs per case.
                self.state_dir = self.root / f"state-{name}"
                self.lock_dir = self.root / f"locks-{name}"
                self.state_path = self.root / f"fake-gpu-state-{name}.json"
                self.set_scenario(stages)
                marker = self.root / f"marker-{name}.txt"
                proc = self.popen(self.marker_command(marker))
                time.sleep(1.3)
                status = self.read_status()
                self.assertEqual(status["phase"], "waiting")
                self.assertFalse(marker.exists(), "child must not launch")
                if not name.startswith("occupied-"):
                    self.assertTrue(status["last_observation"]["error"])
                else:
                    self.assertFalse(status["last_observation"]["error"])
                    self.assertTrue(status["last_observation"]["gpus"][0]["compute_apps"])
                proc.terminate()
                self.assertNotEqual(proc.wait(timeout=10), 0)
                self.assertEqual(self.read_status()["phase"], "interrupted")

    def test_completed_state_dir_never_reruns(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker))
        self.wait_phase(proc, "completed")
        self.assertEqual(proc.wait(timeout=10), 0)
        marker_before = marker.read_text()
        rerun = subprocess.run(
            [sys.executable, str(SCRIPT), *self.base_args(), "--",
             *self.marker_command(marker)],
            env=self.env(), capture_output=True, text=True, timeout=20)
        self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
        self.assertEqual(marker.read_text(), marker_before)
        self.assertEqual(self.read_status()["phase"], "completed")

    def test_failed_child_is_not_retried(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        marker = self.root / "failed-attempts.txt"
        command = self.marker_command(marker, extra="raise SystemExit(3)")
        argv = [sys.executable, str(SCRIPT), *self.base_args(idle=0.4),
                "--", *command]
        run = subprocess.run(
            argv, env=self.env(), capture_output=True, text=True, timeout=25)
        self.assertEqual(run.returncode, 3)
        status = self.read_status()
        self.assertEqual(status["phase"], "failed")
        self.assertEqual(status["returncode"], 3)
        first_attempt = marker.read_text()
        rerun = subprocess.run(
            argv, env=self.env(), capture_output=True, text=True, timeout=20)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual(marker.read_text(), first_attempt)
        self.assertEqual(self.read_status()["phase"], "failed")

    def test_mismatched_command_requires_new_state_dir(self):
        # Crash a waiting watcher with SIGKILL so its state stays "waiting".
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 90)])])
        proc = self.popen(["true"])
        self.read_status()  # status published immediately
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        mismatch = subprocess.run(
            [sys.executable, str(SCRIPT), *self.base_args(), "--",
             "echo", "different"],
            env=self.env(), capture_output=True, text=True, timeout=20)
        self.assertNotEqual(mismatch.returncode, 0)

    def test_changed_working_directory_is_not_the_completed_job(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        first_cwd = self.root / "first"
        second_cwd = self.root / "second"
        first_cwd.mkdir()
        second_cwd.mkdir()
        marker = self.root / "marker.txt"
        command = self.marker_command(marker)
        first = self.popen(command, extra=["--cwd", str(first_cwd)])
        self.wait_phase(first, "completed")
        self.assertEqual(first.wait(timeout=10), 0)
        original = marker.read_text()
        changed = subprocess.run(
            [sys.executable, str(SCRIPT), *self.base_args(),
             "--cwd", str(second_cwd), "--", *command],
            env=self.env(), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(changed.returncode, 0)
        self.assertEqual(marker.read_text(), original)

    def test_duplicate_watcher_on_same_state_dir_refused(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 90)])])
        first = self.popen(["true"])
        self.wait_phase(first, "waiting")
        second = subprocess.run(
            [sys.executable, str(SCRIPT), *self.base_args(), "--", "true"],
            env=self.env(), capture_output=True, text=True, timeout=20)
        self.assertNotEqual(second.returncode, 0)
        first.terminate()
        first.wait(timeout=10)

    def test_held_uuid_lock_blocks_launch_until_released(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        lock_path = self.lock_dir / f"gpu-{UUID_A}.lock"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        held = open(lock_path, "w")
        fcntl.flock(held, fcntl.LOCK_EX)
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker))
        time.sleep(1.3)
        self.assertEqual(self.read_status()["phase"], "waiting")
        self.assertFalse(marker.exists())
        held.close()  # release contention; watcher should proceed
        self.wait_phase(proc, "completed")
        self.assertEqual(proc.wait(timeout=10), 0)
        self.assertEqual(len(marker.read_text().splitlines()), 1)

    def test_sigterm_stops_only_launched_child(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0)])])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker, extra="time.sleep(30)"),
                          extra=["--idle-seconds", "0.4"])
        status = self.wait_phase(proc, "running")
        child_pid = status["child_pid"]
        canary = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            proc.send_signal(signal.SIGTERM)
            returncode = proc.wait(timeout=15)
            self.assertNotEqual(returncode, 0)
            final = self.read_status()
            self.assertEqual(final["phase"], "interrupted")
            self.assertLess(final["returncode"], 0, "child was killed, not exited")
            self._assert_dead(child_pid)
            self._assert_alive(canary.pid)
        finally:
            canary.kill()
            canary.wait(timeout=10)
        # No other launch happened afterwards.
        self.assertEqual(len(marker.read_text().splitlines()), 1)

    def test_contended_gpu_does_not_reset_the_other_idle_windows(self):
        # Both GPUs have earned their window while another monitor holds
        # A's lock. This monitor must fall through to B within the same
        # qualification pass: B launches after one idle window, not after a
        # second window re-earned from scratch.
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0),
                                          gpu_row(1, UUID_B, 48000, 0)])])
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        held = open(WFI._uuid_lock_path(self.lock_dir, UUID_A), "w")
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            record = self.root / "launch.json"
            proc = self.popen(self.probe_command(record),
                              extra=["--poll-seconds", "0.05",
                                     "--idle-seconds", "1.0"])
            self.wait_phase(proc, "completed", timeout=25)
            self.assertEqual(proc.wait(timeout=10), 0)
            info = json.loads(record.read_text())
            self.assertEqual(info["cuda"], UUID_B)
            self.assertFalse(
                info["lock_free"],
                "the monitor must hold the launched GPU's UUID lock for the "
                "whole child run")
            elapsed = record.stat().st_mtime - self.fake_start()
            self.assertGreaterEqual(elapsed, 1.0,
                                    "launched without a full idle window")
            self.assertLess(
                elapsed, 1.6,
                "the second qualified GPU re-earned an idle window instead "
                "of being claimed in the same qualification pass")
        finally:
            held.close()

    def test_second_candidate_is_reverified_under_its_own_lock(self):
        # A's lock is held, so the pass falls through to B, which is still
        # occupied by a compute application. The fresh recheck under B's
        # lock must refuse the launch until B is genuinely idle again.
        self.set_scenario([
            stage(0.0, gpus=[gpu_row(0, UUID_A, 48000, 0),
                             gpu_row(1, UUID_B, 48000, 0)],
                  apps=[app_row(UUID_B, 4242)]),
            stage(0.6, gpus=[gpu_row(0, UUID_A, 48000, 0),
                             gpu_row(1, UUID_B, 48000, 0)]),
        ])
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        held = open(WFI._uuid_lock_path(self.lock_dir, UUID_A), "w")
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            record = self.root / "launch.json"
            proc = self.popen(self.probe_command(record),
                              extra=["--idle-seconds", "0.0"])
            self.wait_phase(proc, "completed", timeout=25)
            self.assertEqual(proc.wait(timeout=10), 0)
            info = json.loads(record.read_text())
            self.assertEqual(info["cuda"], UUID_B)
            self.assertFalse(info["lock_free"])
            elapsed = record.stat().st_mtime - self.fake_start()
            self.assertGreaterEqual(
                elapsed, 0.55,
                "launched on B without re-verifying that B had gone idle")
            self.assertLess(elapsed, 5.0)
        finally:
            held.close()

    def test_all_qualified_gpus_held_keeps_windows_without_busy_spin(self):
        self.set_scenario([stage(0, gpus=[gpu_row(0, UUID_A, 48000, 0),
                                          gpu_row(1, UUID_B, 48000, 0)])])
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        held = [open(WFI._uuid_lock_path(self.lock_dir, uuid), "w")
                for uuid in (UUID_A, UUID_B)]
        for handle in held:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            record = self.root / "launch.json"
            proc = self.popen(self.probe_command(record),
                              extra=["--poll-seconds", "0.3",
                                     "--idle-seconds", "2.0"])
            self.wait_for(lambda: len(self.observation_starts()) >= 8,
                          timeout=20)
            self.assertFalse(record.exists(),
                             "child must not launch while every GPU is held")
            starts = self.observation_starts()
            self.assertGreaterEqual(
                starts[-1], 2.0,
                "premise: a qualified GPU must have been refused while held")
            gaps = [later - earlier
                    for earlier, later in zip(starts, starts[1:])]
            self.assertTrue(gaps, "expected repeated observations while held")
            self.assertGreaterEqual(
                min(gaps), 0.2,
                "monitor busy-spun while every qualified GPU was held")
            self.assertEqual(self.read_status()["phase"], "waiting")
            released_at = time.monotonic()
            for handle in held:
                handle.close()
            self.wait_phase(proc, "completed", timeout=20)
            self.assertEqual(proc.wait(timeout=10), 0)
            self.assertLess(
                time.monotonic() - released_at, 1.0,
                "already-earned idle windows were discarded while held")
            info = json.loads(record.read_text())
            self.assertIn(info["cuda"], (UUID_A, UUID_B))
            self.assertFalse(info["lock_free"])
        finally:
            for handle in held:
                handle.close()

    def test_observation_failure_midwindow_resets_idle_window(self):
        # Idle, then the query itself fails, then idle again. The failed
        # observations must discard the window earned before them: with a
        # 0.6s window a launch cannot happen before ~1.5s.
        self.set_scenario([
            stage(0.0, gpus=[gpu_row(0, UUID_A, 48000, 0)]),
            stage(0.4, exit_code=1),
            stage(0.9, gpus=[gpu_row(0, UUID_A, 48000, 0)]),
        ])
        marker = self.root / "marker.txt"
        proc = self.popen(self.marker_command(marker),
                          extra=["--poll-seconds", "0.05",
                                 "--idle-seconds", "0.6"])
        self.wait_phase(proc, "completed", timeout=25)
        self.assertEqual(proc.wait(timeout=10), 0)
        launch_elapsed = marker.stat().st_mtime - self.fake_start()
        self.assertGreaterEqual(
            launch_elapsed, 1.35,
            "an observation failure did not reset the earned idle window")
        self.assertLess(launch_elapsed, 5.0)

    # -- small utils -------------------------------------------------------

    def wait_for(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("condition not reached")

    def _assert_dead(self, pid):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                return
            time.sleep(0.1)
        self.fail(f"child pid {pid} is still alive")

    def _assert_alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.fail(f"canary pid {pid} was killed; unrelated PIDs must survive")


if __name__ == "__main__":
    unittest.main()
