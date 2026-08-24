"""The every-X-seconds loop: read segments, extract features, emit a classification."""

import importlib
from seed42_audio.io.stream import AudioStream
from seed42_audio.state.music_state import MusicState

# Each module exposes extract(samples, sr) -> dict, with keys matching MusicState
# fields. Modules still being written are skipped until they exist.
FEATURE_MODULES = [
    "seed42_audio.features.spectral",
    "seed42_audio.features.rhythm",
    "seed42_audio.features.chroma",
    "seed42_audio.mood.classify",
]


def load_extractors():
    extractors = []
    for name in FEATURE_MODULES:
        try:
            module = importlib.import_module(name)
            extractors.append((name, module.extract))
        except (ImportError, AttributeError):
            print(f"[pipeline] {name} not ready, skipping for now")
    return extractors


def classify(samples, sr, timestamp, extractors):
    state = MusicState(timestamp=round(timestamp, 1))
    for name, extract in extractors:
        try:
            state.update(extract(samples, sr))
        except Exception as error:
            print(f"[pipeline] {name} failed at {timestamp:.0f}s: {error}")
    return state


def run(paths, sr=22050, window=30.0, hop=1.0, emit_every=60.0):
    """Play the given files back to back and print one classification every
    `emit_every` seconds, each built from the trailing `window` seconds.
    """
    extractors = load_extractors()
    elapsed = 0.0
    next_emit = emit_every
    last = 0.0
    for path in paths:
        stream = AudioStream(path, sr=sr, window=window, hop=hop)
        for timestamp, samples in stream.segments():
            last = elapsed + timestamp
            if last >= next_emit:
                print(classify(samples, sr, last, extractors).to_json())
                next_emit += emit_every
        elapsed = last
