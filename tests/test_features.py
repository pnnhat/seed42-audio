# The synthetic checks as pytest: every synthetic clip in data/ against the
# answer recorded for it in data/synth_answer_key.csv. Windowing copies
# windows() in scripts/evaluate_stages.py and the pass rules copy
# synthetic_checks() in scripts/evaluation.py, so this file and the evaluation
# report agree by construction. The real extractors run on real windows.
#
# Clips are decoded inside module-scoped fixtures, so collection is instant and
# a missing clip skips only the tests that need it.
#
#   python -m pytest        # from the repo root with the seed42 env active

import csv
import math
from pathlib import Path

import pytest

from seed42_audio.features import chroma, rhythm, spectral
from seed42_audio.io.stream import AudioStream

DATA = Path(__file__).resolve().parent.parent / "data"
SR = 22050
HOP = 512          # librosa's default hop, which sets the tempo steps
WINDOW = 30.0      # the pipeline's trailing window, in seconds
STRIDE = 5.0       # one window every 5 seconds of audio
LIMIT = 60.0       # stop after 60 s, except for the transition clip
SWITCH = 45.0      # the transition clip changes tempo and key at 45 s


# --- Windowing, as evaluate_stages.py does it -----------------------------------


def windows(name, limit=LIMIT):
    path = DATA / name
    if not path.exists():
        pytest.skip("%s is missing from data/" % name)
    stream = AudioStream(str(path), sr=SR, window=WINDOW, hop=1.0)
    next_at = WINDOW
    out = []
    for timestamp, samples in stream.segments():
        if timestamp > limit + 1e-6:
            break
        if timestamp + 1e-6 >= next_at:
            out.append((round(timestamp, 1), samples))
            next_at += STRIDE
    return out


@pytest.fixture(scope="module")
def clicks():
    return windows("synth_clicks_120bpm.wav")


@pytest.fixture(scope="module")
def c_major():
    return windows("synth_chord_c_major.wav")


@pytest.fixture(scope="module")
def a_minor():
    return windows("synth_chord_a_minor.wav")


@pytest.fixture(scope="module")
def noise():
    return windows("synth_white_noise.wav")


@pytest.fixture(scope="module")
def silence():
    return windows("synth_silence.wav")


@pytest.fixture(scope="module")
def transition():
    return windows("synth_transition_90_to_140bpm.wav", limit=math.inf)


# --- The answer key --------------------------------------------------------------


@pytest.fixture(scope="module")
def answers():
    # One row per clip, keyed by file name. Cells hold a single value, "none",
    # or a before-and-after pair written as "90 then 140".
    path = DATA / "synth_answer_key.csv"
    if not path.exists():
        pytest.skip("synth_answer_key.csv is missing from data/")
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def expected(row, column):
    value = row[column].strip()
    if value == "none":
        return None
    parts = [p.strip() for p in value.split(" then ")]
    return parts if len(parts) > 1 else parts[0]


# --- Pass rules, as evaluation.py scores them -----------------------------------


def tempo_steps(true_bpm):
    # The tracker can only report 60 * sr / (hop * lag) for a whole-number lag,
    # so a reading counts as correct if it is either step around the true tempo.
    lag = 60 * SR / (HOP * true_bpm)
    return [60 * SR / (HOP * l) for l in (math.floor(lag), math.ceil(lag))]


def on_step(tempo, true_bpm):
    return any(abs(tempo - s) < 0.1 for s in tempo_steps(true_bpm))


def key_mode(samples):
    result = chroma.extract(samples, SR)
    return result["key"], result["mode"]


def check(wins, test, label):
    # Apply test to every window and name the timestamps that fail.
    assert wins, "no windows to check for %s" % label
    bad = [t for t, samples in wins if not test(samples)]
    assert not bad, "%s failed at t=%s (%d of %d windows passed)" % (
        label, bad, len(wins) - len(bad), len(wins))


# --- The seven checks ----------------------------------------------------------


def test_clicks_tempo_on_nearest_step(clicks, answers):
    bpm = float(expected(answers["synth_clicks_120bpm.wav"], "tempo_bpm"))
    check(clicks, lambda s: on_step(rhythm.extract(s, SR)["tempo"], bpm),
          "tempo on a step around %g BPM %s" % (bpm, [round(x, 2) for x in tempo_steps(bpm)]))


def test_c_major_chord_key_and_mode(c_major, answers):
    row = answers["synth_chord_c_major.wav"]
    want = (expected(row, "key"), expected(row, "mode"))
    check(c_major, lambda s: key_mode(s) == want, "key and mode %s" % (want,))


def test_a_minor_chord_key_and_mode(a_minor, answers):
    row = answers["synth_chord_a_minor.wav"]
    want = (expected(row, "key"), expected(row, "mode"))
    check(a_minor, lambda s: key_mode(s) == want, "key and mode %s" % (want,))


def test_white_noise_flatter_than_every_chord_window(noise, c_major, a_minor, answers):
    # The key records flatness as "high" for noise and "low" for the chords;
    # evaluation.py scores that as the lowest noise window above the highest chord window.
    assert expected(answers["synth_white_noise.wav"], "flatness") == "high"
    noise_flat = [spectral.extract(s, SR)["flatness"] for _, s in noise]
    chord_flat = [spectral.extract(s, SR)["flatness"] for _, s in c_major + a_minor]
    assert noise_flat and chord_flat, "no windows to compare"
    assert min(noise_flat) > max(chord_flat), "noise min %.4f, chord max %.4f" % (
        min(noise_flat), max(chord_flat))


def test_transition_before_the_change(transition, answers):
    row = answers["synth_transition_90_to_140bpm.wav"]
    bpm = float(expected(row, "tempo_bpm")[0])
    want = (expected(row, "key")[0], expected(row, "mode")[0])
    before = [(t, s) for t, s in transition if t <= SWITCH]
    check(before,
          lambda s: on_step(rhythm.extract(s, SR)["tempo"], bpm) and key_mode(s) == want,
          "before the change, %g BPM and %s" % (bpm, want))


def test_transition_after_the_change(transition, answers):
    row = answers["synth_transition_90_to_140bpm.wav"]
    bpm = float(expected(row, "tempo_bpm")[1])
    want = (expected(row, "key")[1], expected(row, "mode")[1])
    after = [(t, s) for t, s in transition if t >= SWITCH + WINDOW]
    check(after,
          lambda s: on_step(rhythm.extract(s, SR)["tempo"], bpm) and key_mode(s) == want,
          "after the change, %g BPM and %s" % (bpm, want))


def test_silence_runs_without_error(silence):
    # The three Phase 1 extractors must return on an all-zero window. The full
    # classify and mapping path on silence is covered by evaluate_stages.py.
    def runs(samples):
        rhythm.extract(samples, SR)
        spectral.extract(samples, SR)
        chroma.extract(samples, SR)
        return True

    check(silence, runs, "extractors on silence")


# --- Known open defect ---------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="D-07: bass_energy divides by zero on an empty spectrum, fix in PR #23",
)
def test_silence_spectral_has_no_nan(silence):
    # strict=True means this flips to a failure the moment D-07 merges, which is
    # the signal to delete the xfail marker and keep the test as a plain pass.
    def clean(samples):
        result = spectral.extract(samples, SR)
        return (not any(math.isnan(v) for v in result.values())
                and result["bass_energy"] == 0.0)

    check(silence, clean, "spectral on silence has no NaN")
