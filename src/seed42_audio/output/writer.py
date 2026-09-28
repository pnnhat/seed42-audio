"""Client for the seed42 Control API: PATCHes parameters onto a running stream.

Four calls make a session: log in once, create or accept a stream id, PATCH a
params dict each cycle, delete at the end. The orchestrator assembles the
params dict (Stage prompt plus mapping sliders), this module only sends it.

Credentials come from the environment, never from the repo. Set them once per
machine before the client will run:
  export SEED42_EMAIL="..."
  export SEED42_PASSWORD="..."

Build against the mock (scripts/mock-control-api.py), not a live stream. A live
stream can end on its own, which is indistinguishable from a bug in here.
Switching to live is one change, BASE_URL.
"""

import json
import os
import time

import requests

# --- Configuration ----------------------------------------------------------

MOCK_URL = "http://127.0.0.1:8787"
LIVE_URL = "https://seed42.app"

BASE_URL = MOCK_URL  # the one line that changes when we go live

# Request discipline, from docs/seam-contract.md.
REQUEST_TIMEOUT = 12  # seconds to wait on the server before a call is aborted
MAX_ATTEMPTS = 3  # tries at a PATCH that answers not-ready, the first one included
RETRY_DELAY = 2  # seconds between those tries
POLL_INTERVAL = 2  # seconds between status checks after a 202 create
POLL_LIMIT = 30  # seconds to wait for a 202 stream to start, our own choice

# Fields that update the running stream. Anything outside this set is cold: it
# reloads the pipeline for about 30 seconds and the performance stops. The real
# API answers 200 and reloads anyway, so this is the only place it can be
# caught. Mirrors the hot field list in docs/seam-contract.md.
HOT_FIELDS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "seed",
        "t_index_list",
        "guidance_scale",
        "delta",
        "num_inference_steps",
        "controlnets",
    }
)

ALLOWED_CONTROLNET_FIELDS = frozenset({"conditioning_scale"})

_token = None  # set by login(), read by every call after it


class ColdFieldError(ValueError):
    """Raised when a params dict carries a field that would reload the pipeline."""


class StreamGoneError(RuntimeError):
    """Raised on a 502: the stream has ended and a new one is needed."""


def _headers():
    """Bearer token header. Every call but login needs it."""
    if _token is None:
        raise RuntimeError("call login() first")
    return {"Authorization": "Bearer " + _token}


def _log_attempt(op, attempt, started, status, error):
    """Print one structured line per HTTP attempt so the harness can check them.

    t is when the attempt started, on a monotonic clock, so only the gap between
    two lines means anything: that gap is how the harness checks the 2 second
    retry spacing. ms is how long the attempt took, which is how it checks the
    12 second abort. Flushed straight away because stdout is block buffered when
    it is piped, which is exactly how the harness reads it.
    """
    record = {
        "op": op,
        "attempt": attempt,
        "status": status,
        "error": error,
        "t": round(started, 3),
        "ms": round((time.monotonic() - started) * 1000),
    }
    print("[output.writer] " + json.dumps(record), flush=True)


def _request(op, method, path, attempt=1, **kwargs):
    """Make one HTTP call, abort it after REQUEST_TIMEOUT seconds, and log it.

    Every call in this module goes through here, so every attempt is logged.
    """
    started = time.monotonic()
    try:
        response = requests.request(
            method, BASE_URL + path, timeout=REQUEST_TIMEOUT, **kwargs
        )
    except requests.RequestException as error:
        _log_attempt(op, attempt, started, None, type(error).__name__)
        raise
    _log_attempt(op, attempt, started, response.status_code, None)
    return response


# --- Session ----------------------------------------------------------------


def login() -> str:
    """Log in with the team account and keep the token for the calls below.

    Reads SEED42_EMAIL and SEED42_PASSWORD from the environment. The token
    lasts 30 days, so a sudden 401 later is usually expiry rather than a bad
    request.
    """
    global _token
    response = _request(
        "login",
        "POST",
        "/api/auth",
        json={
            "action": "login",
            "email": os.environ["SEED42_EMAIL"],
            "password": os.environ["SEED42_PASSWORD"],
        },
    )
    response.raise_for_status()
    _token = response.json()["token"]
    return _token


def create_stream(params=None) -> str:
    """Create a stream and return its id once it can take parameters.

    For development only. In production seed42 creates the stream and gives us
    the id, which we pass straight to send(). model_id, width and height
    default server-side if omitted. A 201 is ready to use. A 202 means accepted
    and still starting, so poll until it is ready. Never poll after a 201, the
    status endpoint may answer 404 then and that is not an error.
    """
    response = _request(
        "create",
        "POST",
        "/api/streams",
        headers=_headers(),
        json={"params": params or {}},
    )
    response.raise_for_status()
    stream_id = response.json()["id"]
    if response.status_code == 202:
        _wait_until_ready(stream_id)
    return stream_id


def _wait_until_ready(stream_id):
    """Poll a stream that answered 202 until it reports ready.

    A 502 means it died while starting. The spec sets no limit on the wait, so
    POLL_LIMIT is our own choice.
    """
    for attempt in range(1, POLL_LIMIT // POLL_INTERVAL + 1):
        if attempt > 1:
            time.sleep(POLL_INTERVAL)
        response = _request(
            "poll",
            "GET",
            "/api/streams",
            attempt=attempt,
            headers=_headers(),
            params={"id": stream_id},
        )
        if response.status_code == 502:
            raise StreamGoneError("stream %s ended before it was ready" % stream_id)
        response.raise_for_status()
        if response.json().get("status") == "ready":
            return
    raise TimeoutError("stream %s not ready after %d seconds" % (stream_id, POLL_LIMIT))


def delete_stream(stream_id):
    """End the stream. A failure here is not fatal and is not retried, since
    seed42 closes abandoned sessions server-side.
    """
    try:
        _request(
            "delete",
            "DELETE",
            "/api/streams",
            headers=_headers(),
            json={"stream_id": stream_id},
        )
    except requests.RequestException:
        pass  # already logged by _request, and seed42 cleans up on its own


# --- The per-cycle call -----------------------------------------------------


def _check_hot(params):
    """Raise if params carries a cold field, at the top level or inside a
    controlnets entry.

    This is the last gate before the PATCH, since the real API answers 200 and
    reloads anyway. Each controlnets entry may carry only conditioning_scale:
    a model_id or preprocessor inside one reloads the pipeline just the same.
    """
    cold = set(params) - HOT_FIELDS
    if cold:
        raise ColdFieldError("cold field(s) in PATCH: %s" % ", ".join(sorted(cold)))
    for i, cn in enumerate(params.get("controlnets") or []):
        if not isinstance(cn, dict):
            continue
        cold_sub = set(cn) - ALLOWED_CONTROLNET_FIELDS
        if cold_sub:
            raise ColdFieldError(
                "cold field(s) in controlnets[%d]: %s"
                % (i, ", ".join(sorted(cold_sub)))
            )


def _not_ready(response):
    """True for the one answer worth retrying: the stream is still starting.

    The API says so in two ways and both mean the same: a 409, or a 404 whose
    body says not ready. A 404 without it means the stream does not exist.
    """
    if response.status_code == 409:
        return True
    return response.status_code == 404 and "not ready" in response.text.lower()


def send(stream_id, params) -> dict:
    """PATCH one params dict onto the running stream.

    Retries only while the stream is still starting, and nothing else: a 400
    fails the same way forever, a 502 means the stream has gone, and a call
    that times out is dropped rather than sent late. Blocks until the call
    lands or fails, so with one orchestrator loop only one request is ever in
    flight.
    """
    _check_hot(params)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(RETRY_DELAY)
        response = _request(
            "patch",
            "PATCH",
            "/api/streams",
            attempt=attempt,
            headers=_headers(),
            json={"id": stream_id, "params": params},
        )
        if not _not_ready(response):
            break
    if _not_ready(response):
        # Stream still starting after every retry. Drop this update instead of
        # crashing the session; the next cycle tries again, last write wins.
        print(
            "[output.writer] stream not ready after %d attempts, update dropped"
            % MAX_ATTEMPTS,
            flush=True,
        )
        return {}
    if response.status_code == 502:
        raise StreamGoneError("stream %s has ended, a new one is needed" % stream_id)
    response.raise_for_status()
    return response.json()
