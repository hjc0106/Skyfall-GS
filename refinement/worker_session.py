"""Sequential request-directory IPC for isolated, episode-scoped model workers."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Mapping


class WorkerSession:
    """Load a worker lazily; wait for its exit before the next GPU phase."""

    def __init__(self, command: list[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None):
        self.command = command
        self.env = env
        self.cwd = cwd
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot reopen a closed model worker session")
        return self

    def _diagnostics(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell() - 8192))
        return self._stderr.read().decode("utf-8", errors="replace")

    def run(self, directory: str | Path) -> None:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed model worker session")
        if self._process is None:
            self._stderr = tempfile.TemporaryFile(mode="w+b")
            try:
                self._process = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._stderr, text=True, bufsize=1, env=self.env, cwd=self.cwd,
                )
            except BaseException:
                self.close()
                raise
        process = self._process
        try:
            process.stdin.write(json.dumps({"directory": str(directory)}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"Model worker connection failed.\n{self._diagnostics()}") from error
        if not line:
            raise RuntimeError(f"Model worker exited without acknowledging the request.\n{self._diagnostics()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid model worker response: {line[:512]}\n{self._diagnostics()}") from error
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"Model worker request failed: {response}\n{self._diagnostics()}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        try:
            if process is not None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                finally:
                    process.stdout.close()
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
        if exc_type is None and self._process is not None and self._process.returncode != 0:
            raise RuntimeError(f"Model worker exited with status {self._process.returncode}")
        return False


def serve(handler: Callable[[str], None]) -> None:
    """Keep model output off the acknowledgment pipe; EOF closes the session."""
    output = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                handler(request["directory"])
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            output.write(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}) + "\n")
            output.flush()
            raise SystemExit(1) from error
        output.write(json.dumps({"ok": True}) + "\n")
        output.flush()
