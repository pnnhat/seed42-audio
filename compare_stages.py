# Run one audio window through all three Stages and print the prompts side by
# side. This is the Model Evaluation figure: the same MusicState into Stage 1
# (rule table), Stage 2 (CLAP retrieval), and Stage 3 (local LLM), so the three
# methods can be compared on identical input, with each Stage's own latency.
#
#   python compare_stages.py                 # first window of data/test.wav
#   python compare_stages.py data/test.wav

import sys
import time

from seed42_audio.io.stream import AudioStream
from seed42_audio.pipeline import load_extractors, classify
from seed42_audio.stages import stage1, stage2, stage3


_STAGES = {
    "Stage 1 (rule table)": stage1,
    "Stage 2 (CLAP retrieval)": stage2,
    "Stage 3 (local LLM)": stage3,
}


def one_state(path, sr=22050, window=30.0, hop=1.0):
    # Build a real MusicState from the first full window of the clip, so all
    # three Stages see identical features and the same audio samples.
    extractors = load_extractors()
    stream = AudioStream(path, sr=sr, window=window, hop=hop)
    for timestamp, samples in stream.segments():
        return classify(samples, sr, timestamp, extractors)
    raise SystemExit("no audio window found in %s" % path)


def show(label, module, state):
    started = time.perf_counter()
    result = module.generate(state)
    elapsed = time.perf_counter() - started
    print("\n=== %s   (%.2f s) ===" % (label, elapsed))
    print("prompt          : %s" % result["prompt"])
    print("negative_prompt : %s" % result["negative_prompt"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test.wav"
    state = one_state(path)
    print(
        "MusicState: key %s %s, valence %.2f, arousal %.2f, energy %.2f, tempo %.0f BPM"
        % (
            state.key,
            state.mode,
            state.valence,
            state.arousal,
            state.energy,
            state.tempo,
        )
    )
    for label, module in _STAGES.items():
        show(label, module, state)


if __name__ == "__main__":
    main()
