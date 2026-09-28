#!/usr/bin/env python3
"""Wait for an idle NVIDIA GPU, then launch exactly one job on it.

Stdlib-only monitor. Polls nvidia-smi CSV queries, requires a GPU to stay
continuously idle (enough free memory, low utilization, zero compute
processes, complete and trustworthy observations) for --idle-seconds, then
takes a per-GPU advisory lock, re-verifies the observation while holding it,
and launches the command with CUDA_VISIBLE_DEVICES set to the GPU UUID.
Every GPU that already earned its idle window is considered in the same
qualification pass, so a per-GPU lock held by another monitor skips only that
GPU instead of restarting every other candidate's earned window.

One job per --state-dir (state contract):
- status.json phases: waiting -> running -> completed | failed | interrupted.
- A completed state dir is never rerun (exit 0, no launch).
- failed / interrupted / leftover-running state dirs refuse to rerun: create
  a fresh --state-dir deliberately.
- A stale "waiting" state dir resumes only when the command/config
  fingerprint matches.
- watcher.lock (flock) prevents duplicate watchers on one state dir.
- gpu-<uuid>.lock files in --lock-dir coordinate monitors from this tooling
  only; they are advisory and make no claim against other users' processes.

Exit codes: 0 success (or already-completed state dir); the child's exit
code on child failure (128+N when the child died from signal N); 1 for
environment/state refusals and launch failures; 128+N when the watcher
itself is terminated by signal N; 2 for usage errors.

This tool never kills or competes with processes it did not launch: on
termination it stops only the launched child's process group.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
NVIDIA_SMI = "nvidia-smi"  # resolved via PATH; tests inject a fake there
OBSERVE_TIMEOUT_SECONDS = 30
TERMINATE_GRACE_SECONDS = 10
INTERRUPT_SLICE_SECONDS = 0.2


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# nvidia-smi CSV parsing. Anything unparseable makes the WHOLE observation
# invalid (fail closed): a broken query must never look like an idle GPU.
# ---------------------------------------------------------------------------

def parse_gpu_rows(text: str):
    """Parse `--query-gpu=index,uuid,memory.free,utilization.gpu` output.

    Returns a list of rows, or None when any row is malformed, duplicated,
    or there are no rows at all.
    """
    gpus = []
    seen_index = set()
    seen_uuid = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            return None
        index_text, uuid, free_text, util_text = parts
        if not uuid.startswith("GPU-") or len(uuid) <= 4:
            return None
        try:
            index = int(index_text)
            free_mib = int(free_text)
            utilization = int(util_text)
        except ValueError:
            return None
        if index < 0 or free_mib < 0 or not 0 <= utilization <= 100:
            return None
        if index in seen_index or uuid in seen_uuid:
            return None
        seen_index.add(index)
        seen_uuid.add(uuid)
        gpus.append({"index": index, "uuid": uuid,
                     "free_mib": free_mib, "utilization": utilization})
    if not gpus:
        return None
    return gpus


def parse_app_rows(text: str):
    """Parse `--query-compute-apps=gpu_uuid,pid` output.

    Returns a list of rows, or None when any row is malformed. An empty pid
    still counts the UUID as occupied (fail closed).
    """
    apps = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            return None
        uuid, pid_text = parts
        if not uuid.startswith("GPU-") or len(uuid) <= 4:
            return None
        pid = None
        if pid_text:
            try:
                pid = int(pid_text)
            except ValueError:
                return None
            if pid <= 0:
                return None
        apps.append({"uuid": uuid, "pid": pid})
    return apps


def observe(nvidia_smi: str = NVIDIA_SMI):
    """Run both queries; any failure invalidates the entire observation."""
    gpu_query = [nvidia_smi, "--query-gpu=index,uuid,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"]
    app_query = [nvidia_smi, "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader,nounits"]
    try:
        gpu_run = subprocess.run(gpu_query, capture_output=True, text=True,
                                 timeout=OBSERVE_TIMEOUT_SECONDS)
        app_run = subprocess.run(app_query, capture_output=True, text=True,
                                 timeout=OBSERVE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"gpus": None, "occupied": None,
                "error": f"nvidia-smi query failed: {exc}"}
    if gpu_run.returncode != 0:
        detail = (gpu_run.stderr or "").strip()[:200]
        return {"gpus": None, "occupied": None,
                "error": f"nvidia-smi gpu query exited {gpu_run.returncode}: {detail}"}
    if app_run.returncode != 0:
        detail = (app_run.stderr or "").strip()[:200]
        return {"gpus": None, "occupied": None,
                "error": f"nvidia-smi compute-apps query exited {app_run.returncode}: {detail}"}
    gpus = parse_gpu_rows(gpu_run.stdout)
    if gpus is None:
        return {"gpus": None, "occupied": None,
                "error": "malformed or empty gpu query output"}
    apps = parse_app_rows(app_run.stdout)
    if apps is None:
        return {"gpus": None, "occupied": None,
                "error": "malformed compute-apps query output"}
    known_uuids = {gpu["uuid"] for gpu in gpus}
    if any(app["uuid"] not in known_uuids for app in apps):
        return {"gpus": None, "occupied": None,
                "error": "compute-apps query contains a GPU absent from the GPU inventory"}
    occupied = sorted({app["uuid"] for app in apps})
    return {"gpus": gpus, "occupied": occupied, "error": None}


# ---------------------------------------------------------------------------
# Locks and status file
# ---------------------------------------------------------------------------

def _uuid_lock_path(lock_dir: Path, uuid: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in uuid)
    return lock_dir / f"gpu-{safe}.lock"


def acquire_uuid_lock(lock_dir: Path, uuid: str):
    """Non-blocking advisory flock; returns an open handle or None."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = open(_uuid_lock_path(lock_dir, uuid), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release_lock(handle) -> None:
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    handle.close()


def write_status_atomic(state_dir: Path, status: dict) -> None:
    tmp = state_dir / "status.json.tmp"
    with open(tmp, "w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, state_dir / "status.json")


def command_fingerprint(command, config) -> str:
    return _sha256_json({"command": [str(part) for part in command], "config": config})


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------

def gpu_qualifies(gpu: dict, occupied: set, min_free_mib: int,
                  max_utilization: int) -> bool:
    """A GPU qualifies only on a complete observation with enough free
    memory, low utilization, and zero compute applications."""
    if gpu["uuid"] in occupied:
        return False
    if gpu["free_mib"] < min_free_mib:
        return False
    if gpu["utilization"] > max_utilization:
        return False
    return True


def filter_indices(gpus, gpu_indices):
    if gpu_indices is None:
        return list(gpus)
    wanted = set(gpu_indices)
    return [gpu for gpu in gpus if gpu["index"] in wanted]


# ---------------------------------------------------------------------------
# Status file ownership
# ---------------------------------------------------------------------------

class Monitor:
    """Owns status.json (accumulated, atomically rewritten) and job.log."""

    def __init__(self, state_dir: Path, command, fingerprint: str,
                 min_free_mib: int, max_utilization: int, idle_seconds,
                 poll_seconds, gpu_indices, cwd, log_name="job.log"):
        self.state_dir = state_dir
        self.command = [str(part) for part in command]
        self.log_path = state_dir / log_name
        self.cwd = cwd
        self.min_free_mib = min_free_mib
        self.max_utilization = max_utilization
        self.idle_seconds = idle_seconds
        self.poll_seconds = poll_seconds
        self.gpu_indices = gpu_indices
        self._log_handle = None
        self.status = {
            "schema_version": SCHEMA_VERSION,
            "phase": "waiting",
            "command": self.command,
            "fingerprint": fingerprint,
            "config": {
                "min_free_mib": min_free_mib,
                "max_utilization": max_utilization,
                "idle_seconds": idle_seconds,
                "poll_seconds": poll_seconds,
                "gpu_indices": list(gpu_indices) if gpu_indices is not None else None,
                "cwd": cwd,
            },
            "selected": None,
            "child_pid": None,
            "returncode": None,
            "error": None,
            "last_observation": None,
            "started_at": None,
            "ended_at": None,
            "updated_at": _utc_now(),
        }

    def publish(self, **updates) -> dict:
        self.status.update(updates)
        self.status["updated_at"] = _utc_now()
        write_status_atomic(self.state_dir, self.status)
        return self.status

    def log(self, message: str) -> None:
        print(message, flush=True)

    def open_log(self):
        self._log_handle = open(self.log_path, "ab")
        return self._log_handle

    def close_log(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def summarize_observation(result) -> dict:
    gpus = result.get("gpus")
    return {
        "at": _utc_now(),
        "error": result.get("error"),
        "gpus": None if gpus is None else [
            {"index": gpu["index"], "uuid": gpu["uuid"],
             "free_mib": gpu["free_mib"], "utilization": gpu["utilization"],
             "compute_apps": gpu["uuid"] in (result.get("occupied") or ())}
            for gpu in gpus
        ],
    }


# ---------------------------------------------------------------------------
# Interruption handling
# ---------------------------------------------------------------------------

class InterruptEvent:
    def __init__(self):
        self.triggered = False
        self.signal_number = None

    def fire(self, signal_number, frame):
        self.triggered = True
        self.signal_number = signal_number


class WatcherInterrupted(Exception):
    pass


interrupt_event = InterruptEvent()


def sleep_interruptible(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        if interrupt_event.triggered:
            raise WatcherInterrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(INTERRUPT_SLICE_SECONDS, remaining))


# ---------------------------------------------------------------------------
# Wait / select / launch
# ---------------------------------------------------------------------------

class IdleWaiter:
    """Accumulates the continuous-idle window of every candidate GPU.

    The windows outlive a single observation pass, so losing a per-GPU
    advisory lock to another monitor only skips that GPU: every other
    candidate keeps the window it has already earned and can be claimed in
    the same pass, instead of every card having to re-earn a full window.

    A window resets when its GPU becomes busy, disappears, is filtered out,
    or when an observation fails/malforms (an untrusted observation resets
    every window).
    """

    def __init__(self, monitor: Monitor):
        self.monitor = monitor
        self.idle_since = {}
        self.last_signature = None

    def reset(self) -> None:
        """Forget every earned window: the last observation was untrusted."""
        self.idle_since = {}

    def invalidate(self, uuid: str) -> None:
        """Forget one GPU's window: it no longer looks continuously idle."""
        self.idle_since.pop(uuid, None)

    def poll(self, nvidia_smi: str = NVIDIA_SMI):
        """Observe once and return the ready GPUs, most preferred first.

        A GPU is ready once it has stayed qualified for idle_seconds. Every
        ready candidate is returned so the caller can fall through to the
        next one when a lock is held elsewhere; callers that cannot launch
        keep the returned windows for the next pass.
        """
        monitor = self.monitor
        if interrupt_event.triggered:
            raise WatcherInterrupted()
        result = observe(nvidia_smi)
        monitor.publish(phase="waiting",
                        last_observation=summarize_observation(result),
                        error=result["error"])
        if result["error"] is not None:
            self.reset()
            if self.last_signature != result["error"]:
                self.last_signature = result["error"]
                monitor.log(f"[gpu-wait] observation error: {result['error']}")
            return []
        occupied = set(result["occupied"])
        candidates = filter_indices(result["gpus"], monitor.gpu_indices)
        eligible_now = set()
        for gpu in candidates:
            if gpu_qualifies(gpu, occupied, monitor.min_free_mib,
                             monitor.max_utilization):
                eligible_now.add(gpu["uuid"])
        for uuid in list(self.idle_since):
            if uuid not in eligible_now:
                del self.idle_since[uuid]
        now = time.monotonic()
        for gpu in candidates:
            if gpu["uuid"] in eligible_now and gpu["uuid"] not in self.idle_since:
                self.idle_since[gpu["uuid"]] = now
        signature = "|".join(
            f"{gpu['index']}/{gpu['uuid']}/{gpu['free_mib']}MiB/"
            f"{gpu['utilization']}%/"
            f"{'idle' if gpu['uuid'] in eligible_now else 'busy-or-occupied'}"
            for gpu in candidates) or "no-gpus"
        if signature != self.last_signature:
            self.last_signature = signature
            monitor.log(f"[gpu-wait] {signature}")
        ready = [gpu for gpu in candidates
                 if gpu["uuid"] in self.idle_since
                 and now - self.idle_since[gpu["uuid"]] >= monitor.idle_seconds]
        ready.sort(key=lambda gpu: (-gpu["free_mib"], gpu["index"]))
        return ready


def recheck_under_lock(monitor: Monitor, waiter: IdleWaiter, gpu: dict,
                       nvidia_smi: str = NVIDIA_SMI):
    """Freshly verify one candidate while its advisory lock is held.

    Returns the refreshed gpu dict, or None when this candidate must not be
    launched. A failed observation distrusts every window; a candidate that
    stopped qualifying loses its own window, so it must earn a new one.
    """
    recheck = observe(nvidia_smi)
    if recheck["error"] is not None:
        monitor.publish(last_observation=summarize_observation(recheck),
                        error=recheck["error"])
        monitor.log(f"[gpu-wait] recheck observation error: {recheck['error']}")
        waiter.reset()
        return None
    occupied = set(recheck["occupied"])
    matched = None
    candidates = filter_indices(recheck["gpus"], monitor.gpu_indices)
    eligible = {
        candidate["uuid"] for candidate in candidates
        if gpu_qualifies(candidate, occupied, monitor.min_free_mib,
                         monitor.max_utilization)
    }
    # A recheck observes every GPU, not just the one whose lock we hold.
    # Preserve no window across a busy blip seen while checking another card.
    for uuid in list(waiter.idle_since):
        if uuid not in eligible:
            waiter.invalidate(uuid)
    for candidate in candidates:
        if candidate["uuid"] == gpu["uuid"]:
            matched = candidate
            break
    if (matched is None
            or not gpu_qualifies(matched, occupied, monitor.min_free_mib,
                                 monitor.max_utilization)):
        monitor.log(f"[gpu-wait] {gpu['uuid']} no longer idle at launch recheck")
        waiter.invalidate(gpu["uuid"])
        return None
    return matched


def wait_and_launch(monitor: Monitor, lock_dir: Path,
                    nvidia_smi: str = NVIDIA_SMI) -> int:
    """Wait for an idle GPU, claim it, re-verify it, and launch the child.

    Every already-qualified GPU is tried in the same pass: a GPU whose
    advisory lock is held by another monitor merely moves the pass on to the
    next qualified GPU, and the candidates that were skipped keep their
    earned windows for the next pass. Returns the watcher's exit code.
    """
    waiter = IdleWaiter(monitor)
    while True:
        launched = None
        for gpu in waiter.poll(nvidia_smi):
            if gpu["uuid"] not in waiter.idle_since:
                # A previous recheck in this pass invalidated this candidate
                # (or all windows after an observation error).
                continue
            if interrupt_event.triggered:
                raise WatcherInterrupted()
            gpu_lock = acquire_uuid_lock(lock_dir, gpu["uuid"])
            if gpu_lock is None:
                monitor.log(f"[gpu-wait] {gpu['uuid']} held by another "
                            "monitor; trying the next qualified GPU")
                continue
            try:
                matched = recheck_under_lock(monitor, waiter, gpu, nvidia_smi)
                if matched is None:
                    continue
                if interrupt_event.triggered:
                    raise WatcherInterrupted()
                monitor.log(f"[gpu-select] {matched['uuid']} index "
                            f"{matched['index']} ({matched['free_mib']}MiB free, "
                            f"{matched['utilization']}% util)")
                launched = launch(monitor, matched)
                break
            finally:
                release_lock(gpu_lock)
        if launched is not None:
            return launched
        sleep_interruptible(monitor.poll_seconds)


def selected_record(gpu: dict) -> dict:
    return {
        "uuid": gpu["uuid"],
        "physical_index": gpu["index"],
        "free_mib": gpu["free_mib"],
        "utilization": gpu["utilization"],
        "cuda_visible_devices": gpu["uuid"],
        "selected_at": _utc_now(),
    }


def launch(monitor: Monitor, gpu: dict) -> int:
    """Launch the child on the selected GPU and wait for it. Returns the
    exit code for the watcher process."""
    log_handle = monitor.open_log()
    log_handle.write(
        f"\n=== {_utc_now()} cuda_visible_devices={gpu['uuid']} "
        f"command={json.dumps(monitor.command)}\n".encode())
    log_handle.flush()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    try:
        process = subprocess.Popen(
            monitor.command, cwd=monitor.cwd, stdout=log_handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            start_new_session=True)
    except OSError as exc:
        monitor.close_log()
        monitor.publish(phase="failed", selected=selected_record(gpu),
                        error=f"failed to launch command: {exc}",
                        ended_at=_utc_now())
        monitor.log(f"[gpu-run] launch failed: {exc}")
        return 1
    monitor.publish(phase="running", selected=selected_record(gpu),
                    child_pid=process.pid, error=None,
                    started_at=_utc_now())
    monitor.log(f"[gpu-run] launched pid {process.pid} on {gpu['uuid']}")
    while True:
        try:
            returncode = process.wait(timeout=INTERRUPT_SLICE_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if interrupt_event.triggered:
                stop_child(process)
                monitor.publish(
                    phase="interrupted", returncode=process.returncode,
                    error=(f"watcher received signal "
                           f"{interrupt_event.signal_number}; child stopped"),
                    ended_at=_utc_now())
                monitor.log("[gpu-run] watcher terminated; child stopped")
                raise WatcherInterrupted()
    monitor.close_log()
    if returncode == 0:
        monitor.publish(phase="completed", returncode=0, ended_at=_utc_now())
        monitor.log("[gpu-run] child completed with code 0")
        return 0
    if returncode < 0:
        monitor.publish(phase="failed", returncode=returncode,
                        error=f"child died from signal {-returncode}",
                        ended_at=_utc_now())
        monitor.log(f"[gpu-run] child died from signal {-returncode}")
        return 128 - returncode
    monitor.publish(phase="failed", returncode=returncode, ended_at=_utc_now())
    monitor.log(f"[gpu-run] child failed with code {returncode}")
    return returncode


def stop_child(process) -> None:
    """Stop only the launched child's process group; never other PIDs."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Wait for an idle GPU, then launch exactly one job on it.")
    parser.add_argument("--state-dir", required=True,
                        help="directory for status.json / job.log / watcher.lock")
    parser.add_argument("--lock-dir", required=True,
                        help="directory for advisory per-GPU locks")
    parser.add_argument("--poll-seconds", type=float, default=30.0,
                        help="seconds between observations (default 30)")
    parser.add_argument("--idle-seconds", type=float, default=120.0,
                        help="continuous idle duration required (default 120)")
    parser.add_argument("--initial-delay", type=float, default=0.0,
                        help="seconds to wait before the first observation")
    parser.add_argument("--min-free-mib", type=int, default=40000,
                        help="minimum free MiB to qualify (default 40000)")
    parser.add_argument("--max-utilization", type=int, default=5,
                        help="maximum utilization %% to qualify (default 5)")
    parser.add_argument("--gpu-indices", default=None,
                        help="comma-separated physical GPU indices to consider")
    parser.add_argument("--cwd", default=None, help="child working directory")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to launch once idle (pass after --)")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given after --")
    if not all(math.isfinite(value) for value in
               (args.poll_seconds, args.idle_seconds, args.initial_delay)):
        parser.error("timing values must be finite")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be > 0")
    if args.idle_seconds < 0 or args.initial_delay < 0:
        parser.error("--idle-seconds/--initial-delay must be >= 0")
    if args.min_free_mib < 0 or not 0 <= args.max_utilization <= 100:
        parser.error("--min-free-mib must be nonnegative; utilization must be in 0..100")
    if args.gpu_indices is not None:
        indices = []
        for part in args.gpu_indices.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                indices.append(int(part))
            except ValueError:
                parser.error(f"--gpu-indices has a non-integer entry: {part!r}")
        if any(index < 0 for index in indices):
            parser.error("--gpu-indices must be nonnegative")
        if not indices:
            parser.error("--gpu-indices is empty")
        args.gpu_indices = sorted(set(indices))
    if args.cwd is not None and not Path(args.cwd).is_dir():
        parser.error("--cwd must name an existing directory")
    return args


def read_existing_status(state_dir: Path):
    status_path = state_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        value = json.loads(status_path.read_text())
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported status structure or schema")
        return value
    except (OSError, ValueError) as exc:
        print(f"error: unreadable status.json in {state_dir}: {exc}; "
              "use a new --state-dir", file=sys.stderr)
        return False


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(args.lock_dir)
    # One watcher per state dir.
    watcher_lock = open(state_dir / "watcher.lock", "w")
    try:
        fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("error: another watcher already owns this state dir",
              file=sys.stderr)
        return 1

    command = [str(part) for part in args.command]
    cwd = str(Path(args.cwd or os.getcwd()).resolve())
    fingerprint = command_fingerprint(command, {
        "min_free_mib": args.min_free_mib,
        "max_utilization": args.max_utilization,
        "gpu_indices": args.gpu_indices,
        "idle_seconds": args.idle_seconds,
        "poll_seconds": args.poll_seconds,
        "initial_delay": args.initial_delay,
        "cwd": cwd,
    })

    previous = read_existing_status(state_dir)
    if previous is False:
        return 1
    if previous is not None:
        if previous.get("fingerprint") != fingerprint:
            print("error: state dir belongs to a different command/config; "
                  "use a new --state-dir", file=sys.stderr)
            return 1
        phase = previous.get("phase")
        if phase == "completed":
            print("[gpu-wait] job already completed; not rerunning", flush=True)
            return 0
        if phase in ("failed", "interrupted", "running"):
            print(f"error: previous run left phase {phase!r}; create a new "
                  "--state-dir to rerun deliberately", file=sys.stderr)
            return 1
        if phase != "waiting" or previous.get("schema_version") != SCHEMA_VERSION:
            print("error: unrecognized previous state; use a new --state-dir", file=sys.stderr)
            return 1

    monitor = Monitor(state_dir, command, fingerprint, args.min_free_mib,
                      args.max_utilization, args.idle_seconds,
                      args.poll_seconds, args.gpu_indices, cwd)
    monitor.status["config"]["initial_delay"] = args.initial_delay


    signal.signal(signal.SIGINT, interrupt_event.fire)
    signal.signal(signal.SIGTERM, interrupt_event.fire)

    # Publish waiting immediately, then honor the initial delay.
    monitor.publish(phase="waiting")
    monitor.log(f"[gpu-wait] waiting for an idle GPU for: "
                f"{json.dumps(command)}")
    if args.initial_delay > 0:
        try:
            sleep_interruptible(args.initial_delay)
        except WatcherInterrupted:
            return _finish_interrupted(monitor)
    if interrupt_event.triggered:
        return _finish_interrupted(monitor)

    try:
        return wait_and_launch(monitor, lock_dir, NVIDIA_SMI)
    except WatcherInterrupted:
        return _finish_interrupted(monitor)


def _finish_interrupted(monitor: Monitor) -> int:
    signal_number = interrupt_event.signal_number or 0
    if monitor.status["phase"] != "interrupted":
        monitor.publish(phase="interrupted",
                        error=f"watcher received signal {signal_number}",
                        ended_at=_utc_now())
    return 128 + signal_number


if __name__ == "__main__":
    sys.exit(main())
