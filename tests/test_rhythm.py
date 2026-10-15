# D-09: audio with no clear beat must not be given a confident tempo.

from pathlib import Path

import pytest

from seed42_audio.features import rhythm
from seed42_audio.io.stream import AudioStream

DATA = Path(__file__).resolve().parent.parent / "data"
SR = 22050
WINDOW = 30.0


# The first 30 second window of a clip, loaded the way the pipeline loads it.
def first_window(name):
    path = DATA / name
    if not path.exists():
        pytest.skip("%s is missing from data/" % name)
    stream = AudioStream(str(path), sr=SR, window=WINDOW, hop=1.0)
    for timestamp, samples in stream.segments():
        if timestamp >= WINDOW:
            return samples


def test_clear_beat_keeps_its_tempo():
    out = rhythm.extract(first_window("synth_clicks_120bpm.wav"), SR)
    assert 110 < out["tempo"] < 130


@pytest.mark.parametrize("name", [
    "synth_white_noise.wav",
    "synth_chord_c_major.wav",
    "synth_chord_a_minor.wav",
    "synth_silence.wav",
])
def test_beatless_audio_gets_no_tempo(name):
    out = rhythm.extract(first_window(name), SR)
    assert out["tempo"] == 0.0