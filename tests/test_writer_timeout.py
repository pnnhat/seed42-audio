# D-12: a PATCH that gets no reply within the writer's timeout is logged and
# dropped, and the session carries on.
#
# A stub server stands in for a slow seed42. It holds its reply to the first
# PATCH for longer than the writer waits, then answers the next one at once.
# The mock has no slow mode, which is why this test brings its own server.
# The timeout is shortened here so the test takes about a second, not twelve.
#
#   pytest tests/test_writer_timeout.py

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from seed42_audio.output import writer

TIMEOUT = 0.5  # seconds the writer waits in this test, standing in for 12
HOLD = 2.0  # seconds the stub sits on its first reply, well past TIMEOUT


# Answers PATCH only: slowly the first time, straight away after that.
class SlowFirstPatch(BaseHTTPRequestHandler):
    patches = 0

    def do_PATCH(self):
        SlowFirstPatch.patches += 1
        self.rfile.read(int(self.headers["Content-Length"]))
        if SlowFirstPatch.patches == 1:
            time.sleep(HOLD)
        body = json.dumps({"status": "applied"}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # the writer stopped waiting for this reply, which is the point

    def log_message(self, *args):
        pass  # keep the output to the writer's own lines


# The writer's JSON records from captured output, one per HTTP attempt.
def _records(out):
    prefix = "[output.writer] "
    return [
        json.loads(line[len(prefix):])
        for line in out.splitlines()
        if line.startswith(prefix + "{")
    ]


def test_timed_out_patch_is_dropped_and_session_carries_on(monkeypatch, capsys):
    SlowFirstPatch.patches = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowFirstPatch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(writer, "BASE_URL", "http://127.0.0.1:%d" % server.server_port)
    monkeypatch.setattr(writer, "REQUEST_TIMEOUT", TIMEOUT)
    monkeypatch.setattr(writer, "_token", "test-token")
    params = {"prompt": "slow reply test"}
    try:
        first = writer.send("stream-1", params)  # the stub is slow, so dropped
        second = writer.send("stream-1", params)  # the next cycle lands
    finally:
        server.shutdown()
        server.server_close()

    out = capsys.readouterr().out
    records = _records(out)

    # The slow update was dropped without raising, and the writer said so.
    assert first == {}
    assert "update dropped" in out
    # Its attempt was logged as a timeout, cut off at the timeout rather than
    # left waiting for the late reply.
    assert records[0]["op"] == "patch"
    assert records[0]["status"] is None
    assert records[0]["error"] == "ReadTimeout"
    assert TIMEOUT * 1000 * 0.9 <= records[0]["ms"] < HOLD * 1000
    # The session carried on: the next update went through as normal.
    assert second == {"status": "applied"}
    assert records[1]["status"] == 200
