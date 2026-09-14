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

import os

import requests

# --- Configuration ----------------------------------------------------------

MOCK_URL = "http://127.0.0.1:8787"
LIVE_URL = "https://seed42.app"

BASE_URL = MOCK_URL  # the one line that changes when we go live

# Fields that update the running stream. Anything outside this set is cold: it
# reloads the pipeline for about 30 seconds and the performance stops. The real
# API answers 200 and reloads anyway, so this is the only place it can be
# caught. Mirrors the hot field table in docs/seed42_api.md.
HOT_FIELDS = frozenset({
    "prompt",
    "negative_prompt",
    "seed",
    "t_index_list",
    "guidance_scale",
    "delta",
    "num_inference_steps",
    "controlnets",
})

_token = None  # set by login(), read by every call after it


class ColdFieldError(ValueError):
    """Raised when a params dict carries a field that would reload the pipeline."""


def _headers():
    """Bearer token header. Every call but login needs it."""
    if _token is None:
        raise RuntimeError("call login() first")
    return {"Authorization": "Bearer " + _token}


# --- Session ----------------------------------------------------------------


def login() -> str:
    """Log in with the team account and keep the token for the calls below.

    Reads SEED42_EMAIL and SEED42_PASSWORD from the environment. The token
    lasts 30 days, so a sudden 401 later is usually expiry rather than a bad
    request.
    """
    global _token
    response = requests.post(
        BASE_URL + "/api/auth",
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
    """Create a stream and return its id.

    For development only. In production seed42 creates the stream and gives us
    the id, which we pass straight to send(). model_id, width and height
    default server-side if omitted.
    """
    response = requests.post(
        BASE_URL + "/api/streams",
        headers=_headers(),
        json={"params": params or {}},
    )
    response.raise_for_status()
    return response.json()["id"]


def delete_stream(stream_id):
    """End the stream. A failure here is not fatal and is not retried, since
    seed42 closes abandoned sessions server-side.
    """
    requests.delete(
        BASE_URL + "/api/streams",
        headers=_headers(),
        json={"stream_id": stream_id},
    )


# --- The per-cycle call -----------------------------------------------------


def _check_hot(params):
    """Raise if params carries a cold field.

    Belongs with next week's request discipline, but it is here now because the
    failure is invisible: the API answers 200 and reloads anyway, so nothing
    downstream would notice until the projector went dark.
    """
    cold = set(params) - HOT_FIELDS
    if cold:
        raise ColdFieldError(
            "these fields would reload the pipeline and stop the show: "
            + ", ".join(sorted(cold))
        )


def send(stream_id, params) -> dict:
    """PATCH one params dict onto the running stream."""
    _check_hot(params)
    response = requests.patch(
        BASE_URL + "/api/streams",
        headers=_headers(),
        json={"id": stream_id, "params": params},
    )
    response.raise_for_status()
    return response.json()