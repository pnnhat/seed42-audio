# Generate the synthetic test clips for the seed42 evaluation, each built so its
# correct tempo, key, mode or flatness is known in advance, plus an answer key.
# Uses numpy and the standard library only, so it runs in either environment.
#
#   python generate_synth_audio.py            # writes into data/
#   python generate_synth_audio.py some/dir   # writes somewhere else

import csv
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22050                         # the pipeline's sample rate
SECONDS = 60                       # long enough for several 30 second windows
RNG = np.random.default_rng(42)    # fixed seed, so the noise is reproducible

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note(name, octave):
    # Equal-tempered frequency in Hz, tuned to A4 = 440.
    semitones = _NAMES.index(name) - _NAMES.index("A") + (octave - 4) * 12
    return 440.0 * 2 ** (semitones / 12)


def chord(notes, seconds, amplitude):
    # Sustained sine-wave chord. notes is a list of (name, octave) pairs.
    t = np.arange(int(SR * seconds)) / SR
    mix = sum(np.sin(2 * np.pi * note(n, o) * t) for n, o in notes)
    return amplitude * mix / len(notes)


def clicks(bpm, seconds, amplitude):
    # A 30 ms decaying 1 kHz blip placed exactly on every beat.
    out = np.zeros(int(SR * seconds))
    t = np.arange(int(0.03 * SR)) / SR
    blip = np.sin(2 * np.pi * 1000 * t) * np.exp(-t / 0.005)
    for start in np.arange(0.0, seconds, 60.0 / bpm):
        i = int(start * SR)
        end = min(i + len(blip), len(out))
        out[i:end] += blip[: end - i]
    return amplitude * out


def write_wav(path, samples):
    # 16-bit mono PCM, clipped to [-1, 1] first.
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(data.tobytes())


# The two triads, each with its root doubled an octave down to anchor the key.
C_MAJOR = [("C", 3), ("C", 4), ("E", 4), ("G", 4)]
A_MINOR = [("A", 2), ("A", 3), ("C", 4), ("E", 4)]
F_SHARP_MAJOR = [("F#", 3), ("F#", 4), ("A#", 4), ("C#", 5)]


def build_clips():
    half = SECONDS * 0.75  # 45 s per side, so there are full windows on both sides
    transition = np.concatenate([
        chord(A_MINOR, half, 0.15) + clicks(90, half, 0.3),         # quiet
        chord(F_SHARP_MAJOR, half, 0.35) + clicks(140, half, 0.6),  # loud
    ])
    return {
        "synth_silence.wav": np.zeros(int(SR * SECONDS)),
        "synth_clicks_120bpm.wav": clicks(120, SECONDS, 0.6),
        "synth_chord_c_major.wav": chord(C_MAJOR, SECONDS, 0.3),
        "synth_chord_a_minor.wav": chord(A_MINOR, SECONDS, 0.3),
        "synth_white_noise.wav": 0.1 * RNG.standard_normal(int(SR * SECONDS)),
        "synth_transition_90_to_140bpm.wav": transition,
    }


# What a correct pipeline should report for each clip. "none" means the clip has
# no such property, so any value the pipeline reports is not meaningful.
# Tempo comes out in fixed steps (lag bins of the onset envelope), about 6 BPM
# apart near 120, so the nearest step counts as correct.
ANSWER_KEY = [
    ("synth_silence.wav", "none", "none", "none", "none",
     "Edge case. The pipeline must not crash. Key detection falls back to C "
     "major on silence, which is a default rather than a reading."),
    ("synth_clicks_120bpm.wav", "120", "none", "none", "none",
     "Tempo check. The nearest steps are 117.5 and 123.0, so either counts as "
     "correct. 60 or 240 would be an octave error."),
    ("synth_chord_c_major.wav", "none", "C", "major", "low",
     "Key and mode check. No beat, so any tempo reported is not meaningful."),
    ("synth_chord_a_minor.wav", "none", "A", "minor", "low",
     "Key and mode check. No beat, so any tempo reported is not meaningful."),
    ("synth_white_noise.wav", "none", "none", "none", "high",
     "Flatness check. Noise should read far flatter than the chords. Any key "
     "or tempo reported is not meaningful."),
    ("synth_transition_90_to_140bpm.wav", "90 then 140", "A then F#",
     "minor then major", "low",
     "Change at 45 s, quiet to loud. 90 reads as the step 89.1 and 140 as "
     "143.6. Windows that straddle the change show blended readings."),
]


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, samples in build_clips().items():
        write_wav(out / name, samples)
        print("wrote %s (%.0f s)" % (out / name, len(samples) / SR))
    key_path = out / "synth_answer_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tempo_bpm", "key", "mode", "flatness", "notes"])
        writer.writerows(ANSWER_KEY)
    print("wrote %s" % key_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
