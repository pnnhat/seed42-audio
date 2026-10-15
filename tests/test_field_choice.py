# Sponsor feedback F1: a session drives only the fields chosen in
# SEED42_FIELDS, and a name that is not a field the module drives stops the
# run before it starts.
#
#   python -m pytest tests/test_field_choice.py

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from seed42_audio import orchestrate

ROOT = Path(__file__).resolve().parents[1]

# One window's params, as build_params would make them.
WINDOW = {
    "prompt": "slow drifting fog",
    "negative_prompt": "harsh flashing light",
    "t_index_list": "12",
    "controlnets": [{"conditioning_scale": 0.5}] * 3,
}


def test_empty_setting_drives_every_field():
    assert orchestrate.parse_fields("") == orchestrate.FIELDS


def test_setting_picks_fields_and_ignores_spaces():
    chosen = orchestrate.parse_fields(" prompt , controlnets ")
    assert chosen == ("prompt", "controlnets")


@pytest.mark.parametrize("setting", ["seed", "promt", "prompt,model_id"])
def test_a_name_outside_fields_is_refused(setting):
    with pytest.raises(ValueError, match="SEED42_FIELDS"):
        orchestrate.parse_fields(setting)


# Run run_dev over two windows with the pipeline and writer replaced, and
# return the params each PATCH would have carried.
def sent_by_run_dev(monkeypatch, **kwargs):
    sent = []

    def fake_run(paths, on_emit=None, **rest):
        for timestamp in (60.0, 120.0):
            on_emit(SimpleNamespace(timestamp=timestamp, energy=0.1))

    monkeypatch.setattr(orchestrate, "run", fake_run)
    monkeypatch.setattr(orchestrate.writer, "login", lambda: None)
    monkeypatch.setattr(orchestrate.writer, "create_stream", lambda: "stream")
    monkeypatch.setattr(
        orchestrate.writer, "send", lambda stream_id, params: sent.append(params)
    )
    monkeypatch.setattr(orchestrate.writer, "delete_stream", lambda stream_id: None)
    monkeypatch.setattr(
        orchestrate, "build_params", lambda state, stage_module: dict(WINDOW)
    )
    orchestrate.run_dev(["unused.wav"], stage=1, **kwargs)
    return sent


def test_session_sends_only_the_chosen_fields(monkeypatch):
    sent = sent_by_run_dev(monkeypatch, fields=("t_index_list", "controlnets"))
    sliders = {"t_index_list": "12", "controlnets": WINDOW["controlnets"]}
    assert sent == [sliders, sliders]


def test_session_with_no_choice_sends_every_field(monkeypatch):
    assert sent_by_run_dev(monkeypatch) == [WINDOW, WINDOW]


# The setting is read when the orchestrator starts, so a bad one must stop it
# before it logs in. The address points at a closed port, so if the check
# were skipped the run would fail on the login instead, with another message.
def test_bad_setting_stops_the_orchestrator_before_it_logs_in():
    env = {
        **os.environ,
        "SEED42_FIELDS": "seed",
        "SEED42_BASE_URL": "http://127.0.0.1:9",
    }
    result = subprocess.run(
        [sys.executable, "-m", "seed42_audio.orchestrate"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "SEED42_FIELDS: seed not one of" in result.stderr
    assert "[output.writer]" not in result.stdout
