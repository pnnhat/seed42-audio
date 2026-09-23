"""D3 test harness: run the dev orchestrator against the mock Control API in
each failure mode and check the client behaves as docs/seam-contract.md says.

    python tests/harness.py                # every mode
    python tests/harness.py always-ready   # one or more named modes

Run with the seed42 conda env active. Both subprocesses use sys.executable.
Attempt counts are read from the mock's own request log, one line per HTTP
request with its status code, so they do not depend on client-side logging.
Logs land in tests/logs/ (gitignored).
"""

import os
import re
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
SONG = "data/test.wav"
WINDOW, HOP, EMIT_EVERY = 30.0, 1.0, 10.0
STAGE = 1  # placeholder until Stage selection lands in orchestrate.py
CREDENTIALS = {"SEED42_EMAIL": "harness@test.local", "SEED42_PASSWORD": "harness"}

# Mock flags per mode. The not-ready window is stretched to 20 s so the first
# PATCH is guaranteed to hit it. unstable kills the stream 45 s after creation
# in wall-clock time, so that mode plays the clip three times to stay alive
# past the cutoff.
MODES = {
    "always-ready": (["--always-ready"], 1),
    "not-ready-404": (["--not-ready", "404", "--ready-after", "20"], 1),
    "not-ready-409": (["--not-ready", "409", "--ready-after", "20"], 1),
    "async-create": (["--async-create", "--ready-after", "20"], 1),
    "unstable": (["--unstable"], 3),
    "fail-rate": (["--always-ready", "--fail-rate", "0.3"], 1),
}

REQUEST = re.compile(r'"(GET|POST|PATCH|DELETE) (\S+) HTTP/1\.1" (\d{3})')
PATCH_LINE = re.compile(r"patch #(\d+) on (\S+): (.*)")


# Running 

def wait_for_port(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_mode(name):
    mock_args, repeats = MODES[name]
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mock_log = LOGS / f"{stamp}_{name}_mock.log"
    pipe_log = LOGS / f"{stamp}_{name}_pipeline.log"
    songs = [SONG] * repeats

    with open(mock_log, "w") as mf, open(pipe_log, "w") as pf:
        mock = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", str(PORT), *mock_args],
            stdout=mf, stderr=subprocess.STDOUT, cwd=ROOT,
        )
        try:
            if not wait_for_port(PORT):
                raise RuntimeError("mock did not start on port %d" % PORT)
            code = (
                "from seed42_audio.orchestrate import run_dev; "
                f"run_dev({songs!r}, window={WINDOW}, hop={HOP}, emit_every={EMIT_EVERY})"
            )
            started = time.monotonic()
            pipeline = subprocess.run(
                [sys.executable, "-c", code],
                stdout=pf, stderr=subprocess.STDOUT, cwd=ROOT,
                env={**os.environ, **CREDENTIALS},
            )
            seconds = time.monotonic() - started
        finally:
            mock.terminate()
            mock.wait(timeout=5)

    return {
        "mode": name,
        "exit": pipeline.returncode,
        "seconds": seconds,
        "pipe": parse_pipeline(pipe_log),
        "mock": parse_mock(mock_log),
        "logs": (pipe_log.name, mock_log.name),
    }


# Parsing


def parse_pipeline(path):
    text = path.read_text()
    lines = text.splitlines()
    return {
        "emits": [l for l in lines if l.startswith("{")],
        "sent": [l for l in lines if l.strip().startswith("sent ")],
        "traceback": "Traceback" in text,
    }


def parse_mock(path):
    lines = path.read_text().splitlines()
    requests = [m.groups() for l in lines if (m := REQUEST.search(l))]
    patches = [m.groups() for l in lines if (m := PATCH_LINE.search(l))]
    return {
        "requests": [(method, p, int(code)) for method, p, code in requests],
        "patches": patches,
        "cold": sum("COLD FIELD IN PATCH" in l for l in lines),
    }


def codes(result, method):
    return [c for m, _, c in result["mock"]["requests"] if m == method]


# Assertions, one function per mode, each returns a list of failures


def check_common(r):
    fails = []
    if r["pipe"]["traceback"]:
        fails.append("pipeline raised an exception")
    if r["mock"]["cold"]:
        fails.append("%d cold-field PATCH(es)" % r["mock"]["cold"])
    for _, _, summary in r["mock"]["patches"]:
        if "prompt=" not in summary:
            fails.append("a PATCH landed without a prompt")
            break
        cn = re.search(r"cn=\[([^\]]*)\]", summary)
        if not cn or len(cn.group(1).split(",")) != 3:
            fails.append("a PATCH landed without three controlnet scales")
            break
    return fails


def check_always_ready(r):
    fails = check_common(r)
    if r["exit"] != 0:
        fails.append("exit code %d" % r["exit"])
    n_emit, n_patch = len(r["pipe"]["emits"]), len(r["mock"]["patches"])
    if n_patch == 0:
        fails.append("no PATCH reached the mock")
    if n_emit != n_patch:
        fails.append("%d emits but %d patches landed" % (n_emit, n_patch))
    if any(c != 200 for c in codes(r, "PATCH")):
        fails.append("a PATCH returned something other than 200")
    if codes(r, "GET"):
        fails.append("client polled GET after a 201 create")
    return fails


def check_not_ready(r):
    fails = check_common(r)
    if r["exit"] != 0:
        fails.append("exit code %d, client did not survive not-ready" % r["exit"])
    patch = codes(r, "PATCH")
    not_ready = [c for c in patch if c in (404, 409)]
    if not not_ready:
        fails.append("no not-ready response was seen, mock window too short")
    if 200 not in patch:
        fails.append("no PATCH ever landed after not-ready")
    emits = max(len(r["pipe"]["emits"]), 1)
    if len(not_ready) > 3 * emits:
        fails.append("%d not-ready attempts for %d emits, more than 3 per emit"
                     % (len(not_ready), emits))
    return fails


def check_async_create(r):
    fails = check_not_ready(r)
    if 202 not in codes(r, "POST"):
        fails.append("mock did not answer 202 on create")
    if not codes(r, "GET"):
        fails.append("client never polled GET after the 202")
    return fails


def check_unstable(r):
    fails = check_common(r)
    if r["exit"] != 0:
        fails.append("exit code %d, client did not treat the dead stream as gone" % r["exit"])
    if 502 not in codes(r, "PATCH"):
        fails.append("stream never died, session was under 45 s (%.0f s), add repeats"
                     % r["seconds"])
    if not codes(r, "DELETE"):
        fails.append("client never attempted DELETE")
    return fails


def check_fail_rate(r):
    # Informational until the contract says what follows a 502 on PATCH.
    fails = check_common(r)
    if r["exit"] != 0:
        fails.append("exit code %d after injected 502s" % r["exit"])
    return fails


CHECKS = {
    "always-ready": check_always_ready,
    "not-ready-404": check_not_ready,
    "not-ready-409": check_not_ready,
    "async-create": check_async_create,
    "unstable": check_unstable,
    "fail-rate": check_fail_rate,
}


# Entry 


def main(names):
    results = []
    for name in names:
        print("running %s ..." % name, flush=True)
        r = run_mode(name)
        fails = CHECKS[name](r)
        results.append((r, fails))
        patch = codes(r, "PATCH")
        print("  exit %d, %.0f s, %d emits, PATCH codes %s, logs %s"
              % (r["exit"], r["seconds"], len(r["pipe"]["emits"]), patch, r["logs"][1]))
    print("\nmode            result")
    for r, fails in results:
        print("%-15s %s" % (r["mode"], "PASS" if not fails else "FAIL"))
        for f in fails:
            print("%-15s   %s" % ("", f))
    return int(any(fails for _, fails in results))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(MODES)))