# The synthetic checks as pytest: every synthetic clip in data/ against its
# known answer in data/synth_answer_key.csv. Windowing copies windows() in
# scripts/evaluate_stages.py and the pass rules copy synthetic_checks() in
# scripts/evaluation.py, so this file and the evaluation report agree by
# construction. The real extractors run on real windows, nothing is simulated.
#
#   python -m pytest        # from the repo root with the seed42 env active

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
    stream = AudioStream(str(DATA / name), sr=SR, window=WINDOW, hop=1.0)
    next_at = WINDOW
    out = []
    for timestamp, samples in stream.segments():
        if timestamp > limit + 1e-6:
            break
        if timestamp + 1e-6 >= next_at:
            out.append((round(timestamp, 1), samples))
            next_at += STRIDE
    return out


def params(wins):
    return [pytest.param(samples, id="t=%g" % t) for t, samples in wins]


CLICKS = windows("synth_clicks_120bpm.wav")
C_MAJOR = windows("synth_chord_c_major.wav")
A_MINOR = windows("synth_chord_a_minor.wav")
NOISE = windows("synth_white_noise.wav")
SILENCE = windows("synth_silence.wav")
TRANSITION = windows("synth_transition_90_to_140bpm.wav", limit=math.inf)
BEFORE = [(t, s) for t, s in TRANSITION if t <= SWITCH]
AFTER = [(t, s) for t, s in TRANSITION if t >= SWITCH + WINDOW]


# --- Tempo steps, as evaluation.py scores them ---------------------------------


def tempo_steps(true_bpm):
    # The tracker can only report 60 * sr / (hop * lag) for a whole-number lag,
    # so a reading counts as correct if it is either step around the true tempo.
    lag = 60 * SR / (HOP * true_bpm)
    return [60 * SR / (HOP * l) for l in (math.floor(lag), math.ceil(lag))]


def on_step(tempo, true_bpm):
    return any(abs(tempo - s) < 0.1 for s in tempo_steps(true_bpm))


# --- The seven checks ----------------------------------------------------------


@pytest.mark.parametrize("samples", params(CLICKS))
def test_clicks_tempo_on_nearest_step(samples):
    tempo = rhythm.extract(samples, SR)["tempo"]
    assert on_step(tempo, 120), "tempo %.3f is not on a step around 120 BPM %s" % (
        tempo, [round(s, 3) for s in tempo_steps(120)])


@pytest.mark.parametrize("samples", params(C_MAJOR))
def test_c_major_chord_key_and_mode(samples):
    result = chroma.extract(samples, SR)
    assert (result["key"], result["mode"]) == ("C", "major")


@pytest.mark.parametrize("samples", params(A_MINOR))
def test_a_minor_chord_key_and_mode(samples):
    result = chroma.extract(samples, SR)
    assert (result["key"], result["mode"]) == ("A", "minor")


def test_white_noise_flatter_than_every_chord_window():
    noise = [spectral.extract(s, SR)["flatness"] for _, s in NOISE]
    chords = [spectral.extract(s, SR)["flatness"] for _, s in C_MAJOR + A_MINOR]
    assert min(noise) > max(chords), "noise min %.4f, chord max %.4f" % (
        min(noise), max(chords))


@pytest.mark.parametrize("samples", params(BEFORE))
def test_transition_before_change_90bpm_a_minor(samples):
    tempo = rhythm.extract(samples, SR)["tempo"]
    result = chroma.extract(samples, SR)
    assert on_step(tempo, 90), "tempo %.3f is not on a step around 90 BPM" % tempo
    assert (result["key"], result["mode"]) == ("A", "minor")


@pytest.mark.parametrize("samples", params(AFTER))
def test_transition_after_change_140bpm_f_sharp_major(samples):
    tempo = rhythm.extract(samples, SR)["tempo"]
    result = chroma.extract(samples, SR)
    assert on_step(tempo, 140), "tempo %.3f is not on a step around 140 BPM" % tempo
    assert (result["key"], result["mode"]) == ("F#", "major")


@pytest.mark.parametrize("samples", params(SILENCE))
def test_silence_runs_without_error(samples):
    # The three Phase 1 extractors must return on an all-zero window. The full
    # classify and mapping path on silence is covered by evaluate_stages.py.
    rhythm.extract(samples, SR)
    spectral.extract(samples, SR)
    chroma.extract(samples, SR)


# --- Known open defect ---------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="D-07: bass_energy divides by zero on an empty spectrum, fix in PR #23",
)
@pytest.mark.parametrize("samples", params(SILENCE))
def test_silence_spectral_has_no_nan(samples):
    # strict=True means this flips to a failure the moment D-07 merges, which is
    # the signal to delete the xfail marker and keep the test as a plain pass.
    result = spectral.extract(samples, SR)
    assert not any(math.isnan(v) for v in result.values()), result
    assert result["bass_energy"] == 0.0
