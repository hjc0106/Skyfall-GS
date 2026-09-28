#!/usr/bin/env python3
"""Run GPU jobs through the repository's idle-device monitor.

Each job is handed to ``<protocol['repo_root']>/scripts/wait_for_idle_gpu.py``.
The monitor observes sustained idleness, takes a shared per-GPU lock, rechecks
occupancy, and sets CUDA_VISIBLE_DEVICES to the selected GPU UUID.

Per-job status and logs live in ``<stage_dir>/<job id>``; ``dispatch.json``
records aggregate progress. Waiting jobs resume only with matching commands
and settings. Completed jobs are reused only when their recorded command and
expected outputs still match. Failed, interrupted, or running state is never
silently retried. Signals affect only this dispatcher's own monitor/child
process groups.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

SCHEMA_VERSION = 1
MAX_LIVE_MONITORS = 6
WAIT_SLICE_SECONDS = 0.2
MONITOR_STOP_GRACE_SECONDS = 30.0
MONITOR_KILL_GRACE_SECONDS = 5.0
HEARTBEAT_SECONDS = 5.0
LOG_TAIL_LINES = 40
DISPATCH_FILENAME = "dispatch.json"
MONITOR_LOG_NAME = "monitor.log"
STATUS_FILENAME = "status.json"
MONITOR_RELATIVE_PATH = ("scripts", "wait_for_idle_gpu.py")
TERMINAL_STATUSES = ("completed", "failed", "interrupted", "skipped")


class DispatchFailed(Exception):
    """A job (or the controller) failed; partial results are attached."""

    def __init__(self, message: str, results):
        super().__init__(message)
        self.results = results


class DispatchInterrupted(Exception):
    """The controller was terminated by a signal; partial results attached."""

    def __init__(self, signal_number, results):
        super().__init__(f"dispatch interrupted by signal {signal_number}")
        self.signal_number = signal_number
        self.results = results


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, payload) -> None:
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _protocol_field(protocol: dict, key: str):
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    if key not in protocol:
        raise ValueError(f"protocol is missing required key {key!r}")
    return protocol[key]


def _protocol_number(protocol: dict, key: str):
    value = _protocol_field(protocol, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"protocol key {key!r} must be a number, got {value!r}")
    return value


def _protocol_int(protocol: dict, key: str) -> int:
    value = _protocol_number(protocol, key)
    if int(value) != value:
        raise ValueError(f"protocol key {key!r} must be an integer, got {value!r}")
    return int(value)


def _signal_group(pid, sig) -> None:
    """Signal one process group this controller spawned; never another PID."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _tail_lines(path: Path, limit: int = LOG_TAIL_LINES):
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        return [f"<unreadable log: {exc}>"]
    lines = text.splitlines()
    if not lines:
        return ["<empty log>"]
    return lines[-limit:]


def _normalize_jobs(jobs):
    if not isinstance(jobs, (list, tuple)) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    normalized = []
    seen = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"job #{index} must be an object")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError(f"job #{index} needs a non-empty string id")
        if job_id != os.path.basename(job_id) or job_id in (".", ".."):
            raise ValueError(f"job id {job_id!r} is not a safe directory name")
        if job_id in seen:
            raise ValueError(f"duplicate job id {job_id!r}")
        seen.add(job_id)
        command = job.get("command")
        if (not isinstance(command, (list, tuple)) or not command
                or not all(isinstance(part, str) and part for part in command)):
            raise ValueError(
                f"job {job_id!r} needs command as a non-empty list of "
                "non-empty strings (full argv)")
        expected = job.get("expected_outputs")
        if (not isinstance(expected, (list, tuple))
                or not all(isinstance(item, str) and item for item in expected)):
            raise ValueError(
                f"job {job_id!r} needs expected_outputs as a list of "
                "non-empty path strings")
        normalized.append({
            "id": job_id,
            "command": [str(part) for part in command],
            "expected_outputs": list(expected),
        })
    return normalized


class _SignalState:
    """Single-shot signal latch; the controller never raises from a handler."""

    def __init__(self):
        self.number = None

    def fire(self, signal_number, frame):
        if self.number is None:
            self.number = signal_number


class _Dispatch:
    """Synchronous CPU controller over one monitor subprocess per job."""

    def __init__(self, jobs, protocol: dict, stage_dir: str):
        self.jobs = jobs
        self.jobs_by_id = {job["id"]: job for job in jobs}
        self.protocol = protocol
        self.stage_dir = Path(stage_dir).expanduser().resolve()
        self.order = [job["id"] for job in jobs]
        self.lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.started_at = None
        self.ended_at = None
        self.records = {job["id"]: self._new_record(job) for job in jobs}
        self.monitors = {}  # job id -> Popen of the monitor
        self.pending = deque(self.order)
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.phase = "running"
        self.signals = _SignalState()
        self._installed_signals = {}
        self._last_publish = 0.0

        gpu_indices = _protocol_field(protocol, "gpu_indices")
        if not isinstance(gpu_indices, (list, tuple)) or not gpu_indices:
            raise ValueError("protocol key 'gpu_indices' must be a non-empty list")
        self.gpu_indices = sorted({int(index) for index in gpu_indices})
        policy_root = protocol.get("output_dir")
        policy_path = Path(policy_root) / "gpu_policy.json" if policy_root else None
        if policy_path is not None and policy_path.exists():
            policy = _read_json(policy_path)
            allowed = policy.get("gpu_indices") if isinstance(policy, dict) else None
            if (not isinstance(allowed, list) or not allowed
                    or any(type(index) is not int or index < 0 for index in allowed)):
                raise ValueError(f"Invalid GPU resource policy: {policy_path}")
            self.gpu_indices = sorted(set(self.gpu_indices).intersection(allowed))
            if not self.gpu_indices:
                raise ValueError(f"GPU resource policy permits no protocol GPUs: {policy_path}")
            self._println(f"GPU_POLICY indices={self.gpu_indices} path={policy_path}")
        self.lock_dir = str(_protocol_field(protocol, "lock_dir"))
        self.idle_seconds = _protocol_number(protocol, "idle_seconds")
        self.poll_seconds = _protocol_number(protocol, "poll_seconds")
        self.min_free_mib = _protocol_int(protocol, "min_free_mib")
        self.max_utilization = _protocol_int(protocol, "max_utilization")

        code_dir = Path(str(_protocol_field(protocol, "repo_root"))).expanduser()
        self.monitor_script = code_dir.joinpath(*MONITOR_RELATIVE_PATH)
        if not self.monitor_script.is_file():
            raise ValueError(
                "idle-GPU monitor not found at "
                f"{self.monitor_script} (protocol['repo_root'] must contain "
                "scripts/wait_for_idle_gpu.py)")
        python = protocol.get("python") or sys.executable
        self.python = str(python)

        # Bounded monitor pool: never more live monitors than candidate GPUs.
        self.max_live = max(1, min(len(self.gpu_indices), MAX_LIVE_MONITORS,
                                   len(self.jobs)))

    # -- records ----------------------------------------------------------

    def _new_record(self, job):
        return {
            "id": job["id"],
            "status": "pending",
            "command": list(job["command"]),
            "expected_outputs": list(job["expected_outputs"]),
            "state_dir": str(self.stage_dir / job["id"]),
            "monitor_argv": None,
            "monitor_pid": None,
            "monitor_exit_code": None,
            "monitor_phase": None,
            "monitor_error": None,
            "monitor_fingerprint": None,
            "child_pid": None,
            "child_returncode": None,
            "selected_gpu": None,
            "outputs": [],
            "error": None,
            "logs": {
                "monitor": str(self.stage_dir / job["id"] / MONITOR_LOG_NAME),
                "job": str(self.stage_dir / job["id"] / "job.log"),
                "status": str(self.stage_dir / job["id"] / STATUS_FILENAME),
            },
            "started_at": None,
            "ended_at": None,
        }

    def _println(self, message: str) -> None:
        """One whole line per job/progress event, even from worker threads."""
        with self.print_lock:
            print(message, flush=True)

    def _update(self, job_id: str, publish: bool = True, **fields) -> dict:
        with self.lock:
            record = self.records[job_id]
            record.update(fields)
        if publish:
            self._publish()
        return record

    def _fail(self, job_id: str, message: str) -> dict:
        record = self._update(job_id, status="failed", error=message,
                              ended_at=_utc_now())
        self._println(f"JOB_FAILED id={job_id} error={message}")
        return record

    # -- publishing -------------------------------------------------------

    def _counts(self):
        counts = {"total": len(self.order), "pending": 0, "running": 0,
                  "completed": 0, "failed": 0, "interrupted": 0, "skipped": 0}
        for job_id in self.order:
            status = self.records[job_id]["status"]
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _snapshot(self):
        with self.lock:
            records = [json.loads(json.dumps(self.records[job_id]))
                       for job_id in self.order]
            counts = self._counts()
            monitors = {job_id: process.pid
                        for job_id, process in self.monitors.items()}
        buckets = {status: [record["id"] for record in records
                            if record["status"] == status]
                   for status in ("pending", "running", "completed", "failed",
                                  "interrupted", "skipped")}
        return records, counts, monitors, buckets

    def _publish(self, status=None) -> dict:
        with self.publish_lock:
            return self._publish_locked(status)

    def _publish_locked(self, status=None) -> dict:
        records, counts, monitors, buckets = self._snapshot()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage_dir": str(self.stage_dir),
            "status": status or self.phase,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "ended_at": self.ended_at,
            "counts": counts,
            "pending": buckets["pending"],
            "running": buckets["running"],
            "completed": buckets["completed"],
            "failed": buckets["failed"],
            "interrupted": buckets["interrupted"],
            "skipped": buckets["skipped"],
            "live_monitors": monitors,
            "max_live_monitors": self.max_live,
            "monitor": {
                "script": str(self.monitor_script),
                "python": self.python,
                "lock_dir": self.lock_dir,
                "gpu_indices": self.gpu_indices,
                "idle_seconds": self.idle_seconds,
                "poll_seconds": self.poll_seconds,
                "min_free_mib": self.min_free_mib,
                "max_utilization": self.max_utilization,
            },
            "stop_reason": self.stop_reason,
            "jobs": records,
        }
        _write_json_atomic(self.stage_dir / DISPATCH_FILENAME, payload)
        self._last_publish = time.monotonic()
        return payload

    def _heartbeat(self) -> None:
        if time.monotonic() - self._last_publish >= HEARTBEAT_SECONDS:
            self._publish()

    # -- monitor spawn ----------------------------------------------------

    def _monitor_argv(self, job, state_dir: Path):
        return [
            self.python, str(self.monitor_script),
            "--state-dir", str(state_dir),
            "--lock-dir", self.lock_dir,
            "--poll-seconds", str(self.poll_seconds),
            "--idle-seconds", str(self.idle_seconds),
            "--min-free-mib", str(self.min_free_mib),
            "--max-utilization", str(self.max_utilization),
            "--gpu-indices", ",".join(str(index) for index in self.gpu_indices),
            "--", *job["command"],
        ]

    def _child_pid(self, job_id: str):
        record = self.records[job_id]
        status = _read_json(Path(record["state_dir"]) / STATUS_FILENAME)
        if not isinstance(status, dict):
            return None
        pid = status.get("child_pid")
        return pid if isinstance(pid, int) and pid > 0 else None

    def _run_job(self, job_id: str) -> None:
        """One monitor process for one whole job; runs in a worker thread."""
        job = self.jobs_by_id[job_id]
        record = self.records[job_id]
        if self.stop_event.is_set():
            self._update(job_id, status="skipped",
                         error="dispatch stopped before this job started",
                         ended_at=_utc_now())
            return
        state_dir = Path(record["state_dir"])
        previous = _read_json(state_dir / STATUS_FILENAME)
        if isinstance(previous, dict) and previous.get("phase") == "completed":
            # A changed resource allowlist must not rerun completed computation.
            # Classification still verifies the command, exit status and outputs.
            self._classify(job, record, 0)
            return
        argv = self._monitor_argv(job, state_dir)
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(state_dir / MONITOR_LOG_NAME, "ab")
        except OSError as exc:
            self._fail(job_id, f"cannot prepare monitor state dir/log: {exc}")
            return
        try:
            process = subprocess.Popen(
                argv, cwd=str(state_dir), stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True)
        except OSError as exc:
            log_handle.close()
            self._fail(job_id, f"failed to spawn idle-GPU monitor: {exc}")
            return
        with self.lock:
            self.monitors[job_id] = process
            record["monitor_argv"] = argv
            record["monitor_pid"] = process.pid
            record["status"] = "running"
            record["started_at"] = _utc_now()
        self._println(f"JOB_STARTED id={job_id} monitor_pid={process.pid} "
                      f"state_dir={state_dir}")
        self._publish()
        while True:
            try:
                returncode = process.wait(timeout=WAIT_SLICE_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            log_handle.close()
        except OSError:
            pass
        with self.lock:
            self.monitors.pop(job_id, None)
        self._classify(job, record, returncode)

    # -- outcome classification ------------------------------------------

    def _check_outputs(self, job):
        """Expected outputs exist non-empty; relative paths resolve against the
        job's own state dir, which is also the child's working directory."""
        detail = []
        problems = []
        base = self.stage_dir / job["id"]
        for raw in job["expected_outputs"]:
            path = Path(raw)
            if not path.is_absolute():
                path = base / path
            if path.is_dir():
                detail.append({"path": raw, "resolved": str(path),
                               "kind": "directory", "exists": True})
            elif path.is_file():
                size = path.stat().st_size
                detail.append({"path": raw, "resolved": str(path),
                               "kind": "file", "exists": True, "size": size})
                if size == 0:
                    problems.append(f"{raw} exists but is empty")
            else:
                detail.append({"path": raw, "resolved": str(path),
                               "kind": "missing", "exists": False})
                problems.append(f"{raw} is missing")
        return detail, problems

    def _failure_reason(self, status, returncode):
        parts = []
        if returncode is not None and returncode > 128:
            parts.append(f"monitor exited {returncode} (watcher signalled or "
                         f"child died from signal {returncode - 128})")
        elif returncode not in (0, None):
            parts.append(f"monitor exited {returncode}")
        if returncode == 1:
            parts.append("monitor refused to run/launch (exit 1: environment "
                         "or existing-state refusal; a failed/interrupted/"
                         "running state dir needs a fresh --state-dir, and the "
                         "command/config fingerprint must match)")
        if status is None:
            parts.append("monitor status.json is missing or unreadable; "
                         "completion cannot be confirmed")
        else:
            phase = status.get("phase")
            if phase != "completed":
                parts.append(f"monitor phase is {phase!r}, not 'completed'")
            if status.get("error"):
                parts.append(f"monitor error: {status['error']}")
        return "; ".join(parts) or f"monitor exited {returncode}"

    def _classify(self, job, record, returncode) -> None:
        job_id = job["id"]
        state_dir = Path(record["state_dir"])
        status = _read_json(state_dir / STATUS_FILENAME)
        updates = {"monitor_exit_code": returncode}
        if isinstance(status, dict):
            updates["monitor_phase"] = status.get("phase")
            updates["monitor_error"] = status.get("error")
            updates["monitor_fingerprint"] = status.get("fingerprint")
            child_pid = status.get("child_pid")
            updates["child_pid"] = child_pid if isinstance(child_pid, int) else None
            child_returncode = status.get("returncode")
            updates["child_returncode"] = (child_returncode
                                           if isinstance(child_returncode, int)
                                           else None)
            selected = status.get("selected")
            if isinstance(selected, dict):
                updates["selected_gpu"] = {
                    "uuid": selected.get("uuid"),
                    "physical_index": selected.get("physical_index"),
                    "free_mib": selected.get("free_mib"),
                    "selected_at": selected.get("selected_at"),
                }
            monitor_command = status.get("command")
        else:
            monitor_command = None

        confirmed = (returncode == 0 and isinstance(status, dict)
                     and status.get("phase") == "completed")
        if confirmed and status.get("returncode") != 0:
            confirmed = False
            updates["error"] = "monitor reported completion without a successful child exit"
        if confirmed and monitor_command != list(job["command"]):
            confirmed = False
            updates["error"] = ("monitor completed a different command "
                                "(fingerprint mismatch); refusing to treat "
                                "this job as complete")
        if confirmed:
            detail, problems = self._check_outputs(job)
            updates["outputs"] = detail
            if problems:
                confirmed = False
                updates["error"] = (
                    "monitor reported completion but required outputs are not "
                    "present: " + "; ".join(problems))

        if confirmed:
            updates.update(status="completed", ended_at=_utc_now())
            self._update(job_id, **updates)
            gpu = self.records[job_id]["selected_gpu"] or {}
            self._println(
                f"JOB_COMPLETED id={job_id} exit=0 gpu={gpu.get('uuid')} "
                f"gpu_index={gpu.get('physical_index')} "
                f"child_exit={self.records[job_id]['child_returncode']} "
                f"outputs={len(updates['outputs'])}")
            return

        if self.stop_event.is_set() and "error" not in updates:
            updates.update(status="interrupted", ended_at=_utc_now(),
                           error="dispatch stopped this monitor before the job "
                                 "finished")
            self._update(job_id, **updates)
            self._println(f"JOB_INTERRUPTED id={job_id} exit={returncode}")
            return

        if "error" not in updates:
            updates["error"] = self._failure_reason(status, returncode)[:2000]
        if not updates.get("outputs"):
            updates["outputs"], problems = self._check_outputs(job)
            if problems and "required outputs" not in updates["error"]:
                updates["error"] = (updates["error"] + "; required outputs not "
                                    "present: " + "; ".join(problems))[:2000]
        updates.update(status="failed", ended_at=_utc_now())
        self._update(job_id, **updates)
        self._println(f"JOB_FAILED id={job_id} exit={returncode} "
                      f"error={updates['error']}")

    # -- stop / cleanup ---------------------------------------------------

    def _stop_reason_now(self):
        if self.signals.number is not None:
            return {"kind": "signal", "signal": self.signals.number,
                    "at": _utc_now()}
        for job_id in self.order:
            if self.records[job_id]["status"] == "failed":
                return {"kind": "failure", "signal": None, "at": _utc_now()}
        return None

    def _record_stop(self, reason) -> None:
        self.stop_reason = reason
        self.stop_event.set()
        for job_id in list(self.pending):
            self._update(job_id, publish=False, status="skipped",
                         error="dispatch stopped before this job started",
                         ended_at=_utc_now())
        self.pending.clear()
        self.phase = "failed" if reason["kind"] == "failure" else "interrupted"
        self._println(f"DISPATCH_STOP reason={reason['kind']} "
                      f"signal={reason['signal']}")
        self._publish()

    def _wait_monitors(self, processes, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in processes):
                return
            time.sleep(WAIT_SLICE_SECONDS)

    def _stop_monitors(self) -> None:
        """SIGTERM the monitor groups we spawned; escalate only if needed.

        Each monitor already stops exactly its own child process group when it
        receives SIGTERM, so this never touches another user's processes.
        """
        with self.lock:
            snapshot = list(self.monitors.items())
        live = [(job_id, process) for job_id, process in snapshot
                if process.poll() is None]
        if not live:
            return
        for _job_id, process in live:
            _signal_group(process.pid, signal.SIGTERM)
        self._wait_monitors([process for _, process in live],
                            MONITOR_STOP_GRACE_SECONDS)
        stubborn = [(job_id, process) for job_id, process in live
                    if process.poll() is None]
        if stubborn:
            for job_id, _process in stubborn:
                child_pid = self._child_pid(job_id)
                if child_pid:
                    _signal_group(child_pid, signal.SIGTERM)
            self._wait_monitors([process for _, process in stubborn],
                                MONITOR_KILL_GRACE_SECONDS)
        remaining = [(job_id, process) for job_id, process in stubborn
                     if process.poll() is None]
        for job_id, process in remaining:
            child_pid = self._child_pid(job_id)
            if child_pid:
                _signal_group(child_pid, signal.SIGKILL)
            _signal_group(process.pid, signal.SIGKILL)
        if remaining:
            self._wait_monitors([process for _, process in remaining],
                                MONITOR_KILL_GRACE_SECONDS)

    # -- signals ----------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # handlers are main-thread only; caller keeps ownership
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._installed_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, self.signals.fire)
            except (ValueError, OSError):
                self._installed_signals.pop(signum, None)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._installed_signals.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        self._installed_signals.clear()

    # -- main loop --------------------------------------------------------

    def _results(self):
        return [json.loads(json.dumps(self.records[job_id]))
                for job_id in self.order]

    def _raise_if_stopped(self) -> None:
        if self.stop_reason is None:
            return
        results = self._results()
        failing = [record for record in results
                   if record["status"] in ("failed", "interrupted")]
        if self.stop_reason["kind"] == "signal":
            raise DispatchInterrupted(self.stop_reason["signal"], results)
        lines = [f"dispatch failed: {len(failing)} job(s) did not complete"]
        for record in failing:
            lines.append(f"job {record['id']}: {record['error']}")
            lines.append(f"  state_dir: {record['state_dir']}")
            for name in ("monitor", "job"):
                lines.append(f"  {name} log tail ({record['logs'][name]}):")
                for line in _tail_lines(Path(record["logs"][name])):
                    lines.append(f"    {line}")
        raise DispatchFailed("\n".join(lines), results)

    def run(self):
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = _utc_now()
        self._install_signal_handlers()
        try:
            self._publish()
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_live) as pool:
                futures = {}
                while self.pending or futures:
                    while (self.pending and self.stop_reason is None
                           and len(futures) < self.max_live):
                        job_id = self.pending.popleft()
                        futures[pool.submit(self._run_job, job_id)] = job_id
                    if not futures:
                        break
                    done, _ = concurrent.futures.wait(
                        list(futures), timeout=WAIT_SLICE_SECONDS,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        job_id = futures.pop(future)
                        error = future.exception()
                        if error is not None:
                            self._fail(job_id, f"internal worker error: {error!r}")
                    if self.stop_reason is None:
                        reason = self._stop_reason_now()
                        if reason is not None:
                            self._record_stop(reason)
                    if self.stop_reason is not None:
                        self._stop_monitors()
                    self._heartbeat()
            if self.stop_reason is None:
                self.phase = ("failed" if any(
                    self.records[job_id]["status"] == "failed"
                    for job_id in self.order) else "completed")
        finally:
            self._restore_signal_handlers()
        self.ended_at = _utc_now()
        counts = self._counts()
        self._println(
            f"DISPATCH_COMPLETE status={self.phase} total={counts['total']} "
            f"completed={counts['completed']} failed={counts['failed']} "
            f"interrupted={counts['interrupted']} skipped={counts['skipped']} "
            f"stage_dir={self.stage_dir}")
        self._publish()
        self._raise_if_stopped()
        return self._results()


def run_jobs(jobs, *, protocol: dict, stage_dir: str):
    """Run every job on an idle GPU, at most `min(len(gpu_indices), 6)` at once.

    `jobs` is a list of ``{"id": str, "command": [argv...],
    "expected_outputs": [path...]}``. Each job gets its own monitor subprocess
    and its own ``<stage_dir>/<id>`` state dir, so a re-run resumes waiting
    jobs and refuses to touch state dirs left failed/interrupted/running.
    Relative expected outputs resolve against ``<stage_dir>/<id>``. Returns one
    record per job in input order. Raises DispatchFailed (with the job logs) or
    DispatchInterrupted; call it from the main thread to receive SIGINT/SIGTERM
    handling.
    """
    normalized = _normalize_jobs(jobs)
    return _Dispatch(normalized, protocol, stage_dir).run()


def _load_json_arg(value: str, flag: str):
    path = Path(value).expanduser()
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{flag}: cannot read JSON from {value}: {exc}")
    try:
        return json.loads(value)
    except ValueError as exc:
        raise ValueError(
            f"{flag}: not valid JSON and not an existing file path: {exc}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Dispatch whole GPU jobs onto idle GPUs via "
                    "scripts/wait_for_idle_gpu.py monitors.")
    parser.add_argument("--protocol", required=True,
                        help="protocol JSON object or path to protocol.json")
    parser.add_argument("--jobs", required=True,
                        help="jobs JSON array or path to a jobs JSON file")
    parser.add_argument("--state-dir", required=True,
                        help="stage dir for per-job monitor state and dispatch.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        protocol = _load_json_arg(args.protocol, "--protocol")
        jobs = _load_json_arg(args.jobs, "--jobs")
        results = run_jobs(jobs, protocol=protocol, stage_dir=args.state_dir)
    except DispatchInterrupted as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 128 + (exc.signal_number or 0)
    except (DispatchFailed, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for record in results:
        gpu = record.get("selected_gpu") or {}
        print(f"RESULT id={record['id']} status={record['status']} "
              f"monitor_exit={record['monitor_exit_code']} "
              f"gpu={gpu.get('uuid')} child_exit={record['child_returncode']}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
