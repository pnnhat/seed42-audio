# Start-up timing: seconds from launch to the first prompt, for each Stage.
#
# Answers the half of the sponsor's load-time question that needs no stream:
# how long the module takes, from being started, to have its first params
# ready to send. Each run starts a fresh Python process, so every import and
# model load is paid again, as on a real launch. The clock stops when the
# first window's params (Stage prompt plus mapping sliders) exist. Nothing is
# sent, so neither the mock nor a stream is needed.
#
# A file is read far faster than live audio arrives, so these times leave out
# the wait for the audio itself. Live, the first prompt also has to wait for
# one window of audio, or for the first cycle when the cycle is longer.
#
# A Stage that cannot run falls back to Stage 1 (Stage 2 without CLAP, Stage 3
# without Ollama), and its time is then really Stage 1's. Such runs are
# flagged, so a fallback time is never reported as that Stage's time.
#
#   python scripts/time_startup.py                         # default clip
#   python scripts/time_startup.py data/dark_calm_2.mp3    # another clip

import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP = "data/bright_calm_1.mp3"  # the shortest real track, so decoding adds little
WINDOW = 30.0  # seconds of audio behind the first prompt, as in the evaluation
RUNS = 3  # per Stage

# What each fresh process runs: load the Stage, play the clip up to its first
# window, build that window's params the way the orchestrator does, report,
# and stop. A Stage that falls back returns Stage 1's prompt word for word,
# which is how a fallback run is spotted.
CHILD = """
import importlib, sys
from seed42_audio.orchestrate import build_params
from seed42_audio.pipeline import run

stage_number, clip, window = sys.argv[1], sys.argv[2], float(sys.argv[3])
stage = importlib.import_module("seed42_audio.stages.stage" + stage_number)

def first_window(state):
    params = build_params(state, stage)
    print("FIRST PROMPT", flush=True)
    from seed42_audio.stages import stage1
    same = params["prompt"] == stage1.generate(state)["prompt"]
    print("FELL BACK", stage_number != "1" and same, flush=True)
    raise SystemExit(0)

run([clip], window=window, hop=1.0, emit_every=window, on_emit=first_window)
"""


# One launch of one Stage. Returns (seconds to the first prompt, fell back).
def time_once(stage, clip):
    started = time.monotonic()
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(stage), clip, str(WINDOW)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=ROOT,
    )
    seconds, fell_back, output = None, False, []
    for line in child.stdout:
        output.append(line.rstrip())
        if line.startswith("FIRST PROMPT"):
            seconds = time.monotonic() - started
        elif line.startswith("FELL BACK"):
            fell_back = line.split()[-1] == "True"
    child.wait()
    if seconds is None:
        print("\n".join(output[-20:]))
        sys.exit("Stage %d never produced a prompt, see its output above" % stage)
    return seconds, fell_back


def main(clip):
    print("Start-up timing, seconds from launch to the first prompt")
    print("Machine: %s, Python %s" % (platform.platform(), platform.python_version()))
    print("Clip: %s, %g s window, %d runs per Stage\n" % (clip, WINDOW, RUNS))
    for stage in (1, 2, 3):
        results = [time_once(stage, clip) for _ in range(RUNS)]
        times = "  ".join("%5.2f" % seconds for seconds, _ in results)
        median = statistics.median(seconds for seconds, _ in results)
        print("Stage %d   %s   median %.2f" % (stage, times, median), flush=True)
        fallbacks = [str(i) for i, (_, fell) in enumerate(results, 1) if fell]
        if fallbacks:
            print(
                "          runs that fell back to Stage 1: %s (not this Stage's time)"
                % ", ".join(fallbacks),
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CLIP)
