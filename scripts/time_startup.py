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
# Before timing on a machine for the first time, load Stage 2 once so that
# CLAP's checkpoint (630k-audioset-best.pt) and the roberta-base text model it
# uses are already downloaded, or the first Stage 2 run includes the
# downloads. This prints "[stage2] CLAP ready" when they are in place:
#   python -c "from seed42_audio.stages import stage2"
#
# Ollama keeps Stage 3's model in memory between requests, so only the first
# Stage 3 run would start cold. Before every Stage 3 run the script unloads
# it, the same as `ollama stop llama3.2`, and waits until Ollama no longer
# lists it, so every Stage 3 run starts cold.
#
# The first run of a Stage can still be the slowest, because the operating
# system keeps recently read files in memory. A first launch after a restart
# is the slowest case of all.
#
# A Stage that cannot run falls back to Stage 1 (Stage 2 without CLAP, Stage 3
# without Ollama), and its time is then really Stage 1's. Such runs are
# flagged, so a fallback time is never reported as that Stage's time.
#
# The results, with the device CLAP and Ollama ran on, are printed and saved
# to results/startup_timing_<date>_<time>.txt.
#
#   python scripts/time_startup.py                         # default clip
#   python scripts/time_startup.py data/dark_calm_2.mp3    # another clip

import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from seed42_audio.stages import stage3

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CLIP = "data/bright_calm_1.mp3"  # the shortest real track, so decoding adds little
WINDOW = 30.0  # seconds of audio behind the first prompt, as in the evaluation
RUNS = 3  # per Stage

# Ollama's list of loaded models, beside the endpoint Stage 3 calls.
OLLAMA_PS = stage3.OLLAMA_URL.replace("/api/generate", "/api/ps")

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


# Ollama's entry for Stage 3's model while it is loaded, or None.
def _loaded_stage3_model():
    name = stage3.OLLAMA_MODEL
    if ":" not in name:
        name += ":latest"  # how Ollama lists a model pulled without a tag
    response = requests.get(OLLAMA_PS, timeout=10)
    response.raise_for_status()
    for model in response.json().get("models") or []:
        if name in (model.get("name"), model.get("model")):
            return model
    return None


# Unload Stage 3's model so the next run starts cold. Ollama answers the
# unload before the model is out of memory, so wait until it is no longer
# listed. Returns False if Ollama is not running, in which case Stage 3 falls
# back and its runs are flagged anyway.
def unload_stage3_model():
    try:
        requests.post(
            stage3.OLLAMA_URL,
            json={"model": stage3.OLLAMA_MODEL, "keep_alive": 0},
            timeout=10,
        )
        for _ in range(60):
            if _loaded_stage3_model() is None:
                return True
            time.sleep(0.5)
    except requests.RequestException:
        return False
    sys.exit("%s was still loaded 30 s after the unload" % stage3.OLLAMA_MODEL)


# Where Ollama ran Stage 3's model. Read straight after the Stage 3 runs,
# while the model is still loaded.
def stage3_device():
    try:
        model = _loaded_stage3_model()
    except requests.RequestException:
        model = None
    if not model or not model.get("size"):
        return "unknown, the model was not loaded"
    gpu = min(round(100 * model.get("size_vram", 0) / model["size"]), 100)
    return "CPU only" if gpu == 0 else "%d%% GPU" % gpu


# The device CLAP runs on. laion_clap uses CUDA when torch can see it and the
# CPU otherwise (never Apple's GPU), so ask torch the same way. Checked after
# the runs, so loading torch here cannot speed up a Stage 2 run.
def clap_device():
    try:
        import torch
    except Exception:
        return "unknown, torch could not be loaded"
    if torch.cuda.is_available():
        return "GPU, " + torch.cuda.get_device_name(0)
    return "CPU only"


def main(clip):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / datetime.now().strftime("startup_timing_%Y%m%d_%H%M.txt")
    with open(path, "w", encoding="utf-8") as out:

        # Print a line and keep it in the results file.
        def report(line=""):
            print(line, flush=True)
            out.write(line + "\n")

        report("Start-up timing, seconds from launch to the first prompt")
        report("Run: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
        report(
            "Machine: %s, Python %s"
            % (platform.platform(), platform.python_version())
        )
        report("Clip: %s, %g s window, %d runs per Stage" % (clip, WINDOW, RUNS))
        report()
        for stage in (1, 2, 3):
            results, cold = [], 0
            for _ in range(RUNS):
                if stage == 3:
                    cold += unload_stage3_model()
                results.append(time_once(stage, clip))
            times = "  ".join("%5.2f" % seconds for seconds, _ in results)
            median = statistics.median(seconds for seconds, _ in results)
            report("Stage %d   %s   median %.2f" % (stage, times, median))
            fallbacks = [str(i) for i, (_, fell) in enumerate(results, 1) if fell]
            if fallbacks:
                report(
                    "          runs that fell back to Stage 1: %s"
                    " (not this Stage's time)" % ", ".join(fallbacks)
                )
            if cold == RUNS:
                report(
                    "          %s unloaded before each run, so every run started cold"
                    % stage3.OLLAMA_MODEL
                )
        ollama = stage3_device()  # first, while the model is still loaded
        report()
        report("CLAP (Stage 2) device: %s" % clap_device())
        report("Ollama (Stage 3) device: %s" % ollama)
    print("\nSaved to %s" % path.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CLIP)
