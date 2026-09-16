"""D3 test harness: run one pipeline session against the mock Control API
and keep both stdout streams for later evaluation. No assertions yet."""

import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "tests" / "logs"
MOCK = ROOT / "scripts" / "mock-control-api.py"
PORT = 8787
SONGS = ["data/test.wav"]
WINDOW, HOP, EMIT_EVERY = 30.0, 1.0, 10.0
CREDENTIALS = {"SEED42_EMAIL": "harness@test.local", "SEED42_PASSWORD": "harness"}


def wait_for_port(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_session():
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mock_log = LOGS / f"{stamp}_mock.log"
    pipe_log = LOGS / f"{stamp}_pipeline.log"

    with open(mock_log, "w") as mf, open(pipe_log, "w") as pf:
        mock = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", str(PORT), "--always-ready"],
            stdout=mf, stderr=subprocess.STDOUT, cwd=ROOT,
        )
        try:
            if not wait_for_port(PORT):
                raise RuntimeError("mock did not start on port %d" % PORT)
            code = (
                "from seed42_audio.pipeline import run; "
                f"run({SONGS!r}, window={WINDOW}, hop={HOP}, emit_every={EMIT_EVERY})"
            )
            pipeline = subprocess.run(
                [sys.executable, "-c", code],
                stdout=pf, stderr=subprocess.STDOUT, cwd=ROOT,
                env={**__import__("os").environ, **CREDENTIALS},
            )
        finally:
            mock.terminate()
            mock.wait(timeout=5)

    print(f"pipeline exit code {pipeline.returncode}")
    print(f"logs: {pipe_log.name}, {mock_log.name}")
    return pipe_log, mock_log


if __name__ == "__main__":
    run_session()