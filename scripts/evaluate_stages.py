# Run every evaluation clip in data/ through all three Stages and save one row
# per window per Stage to a CSV, for the D3 Model Evaluation and Performance
# Evaluation sections. Run it from the repo root in the seed42-clap environment
# with Ollama running, so Stage 2 and Stage 3 are both live.
#
#   python scripts/evaluate_stages.py                # every clip, every Stage
#   python scripts/evaluate_stages.py --stages 1,2   # skip Stage 3, much quicker
#
# Each window is timed. A Stage 2 or Stage 3 result identical to Stage 1's is
# recorded as a fallback, since falling back to Stage 1 is what both do when they
# cannot answer. Anything a Stage prints while it runs, such as why it fell back,
# goes in the notes column. Rows are written as they are produced, so a run that
# is interrupted still leaves usable results.

import argparse
import contextlib
import csv
import io
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from seed42_audio.io.stream import AudioStream
from seed42_audio.output import mapping
from seed42_audio.pipeline import classify, load_extractors

SR = 22050
WINDOW = 30.0      # the pipeline's trailing window, in seconds
STRIDE = 5.0       # evaluate one window every 5 seconds of audio
LIMIT = 60.0       # stop after 60 s of audio, except for the transition clip
DATA = Path("data")
OUT = Path("results")

FEATURES = ["tempo", "energy", "brightness", "flatness", "key", "mode",
            "valence", "arousal", "bass_energy"]
COLUMNS = (["file", "group", "timestamp"] + FEATURES
           + ["depth", "edge", "consistency", "t_index",
              "stage", "status", "latency_s", "prompt", "negative_prompt", "notes"])


def clips():
    # Every audio file in data/ except test.wav, which bright_energetic_1.wav copies.
    return sorted(
        p for p in DATA.iterdir()
        if p.suffix.lower() in (".wav", ".mp3", ".flac") and p.name != "test.wav"
    )


def group_of(path):
    # bright_calm_2 becomes bright_calm, and every synth_ clip is synthetic.
    if path.stem.startswith("synth_"):
        return "synthetic"
    return path.stem.rsplit("_", 1)[0]


def load_stages(wanted):
    # Stage 1 always runs, because it is the baseline the fallback check compares against.
    from seed42_audio.stages import stage1
    stages = {1: stage1}
    if 2 in wanted:
        from seed42_audio.stages import stage2
        stages[2] = stage2
    if 3 in wanted:
        from seed42_audio.stages import stage3
        stages[3] = stage3
    return stages


def windows(path, limit):
    # Trailing windows from the real AudioStream, one every STRIDE seconds.
    stream = AudioStream(str(path), sr=SR, window=WINDOW, hop=1.0)
    next_at = WINDOW
    for timestamp, samples in stream.segments():
        if timestamp > limit + 1e-6:
            break
        if timestamp + 1e-6 >= next_at:
            yield timestamp, samples
            next_at += STRIDE


def quietly(fn, *args):
    # Run fn with its printing captured, and return (result, printed lines).
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args)
    lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    return result, lines


def evaluate(path, stages, extractors, out):
    limit = math.inf if "transition" in path.stem else LIMIT
    count = 0
    for timestamp, samples in windows(path, limit):
        base = {"file": path.name, "group": group_of(path), "timestamp": round(timestamp, 1)}
        try:
            state, printed = quietly(classify, samples, SR, timestamp, extractors)
            # Keep only the pipeline's own warnings, such as a feature failing on
            # silence. The routine essentia fallback message is left out.
            feature_notes = [line for line in printed if line.startswith("[pipeline]")]
            for name in FEATURES:
                base[name] = getattr(state, name, "")
            sliders = mapping.to_params(state)
            scales = [cn["conditioning_scale"] for cn in sliders["controlnets"]]
            base.update(depth=scales[0], edge=scales[1], consistency=scales[2],
                        t_index=sliders["t_index_list"])
        except Exception as error:
            out.writerow(dict(base, stage="features", status="error",
                              notes="%s: %s" % (type(error).__name__, error)))
            continue

        baseline = None
        for number, module in stages.items():
            row = dict(base, stage=number)
            started = time.perf_counter()
            try:
                result, printed = quietly(module.generate, state)
                row["latency_s"] = round(time.perf_counter() - started, 4)
                row["prompt"] = result["prompt"]
                row["negative_prompt"] = result["negative_prompt"]
                if number == 1:
                    baseline = result
                    row["status"] = "ok"
                else:
                    row["status"] = "fallback" if result == baseline else "ok"
                row["notes"] = " | ".join(feature_notes + printed)
            except Exception as error:
                row["latency_s"] = round(time.perf_counter() - started, 4)
                row["status"] = "error"
                row["notes"] = "%s: %s" % (type(error).__name__, error)
            out.writerow(row)
        count += 1
    return count


def summarise(csv_path):
    # Per-Stage latency and fallback counts, printed for the report.
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["stage"] in ("1", "2", "3")]
    print("\nstage  windows  ok  fallback  error  median_s  max_s")
    for stage in ("1", "2", "3"):
        mine = [r for r in rows if r["stage"] == stage]
        if not mine:
            continue
        times = [float(r["latency_s"]) for r in mine if r["latency_s"]]
        by = {s: sum(r["status"] == s for r in mine) for s in ("ok", "fallback", "error")}
        print("%5s  %7d  %2d  %8d  %5d  %8.3f  %5.3f" % (
            stage, len(mine), by["ok"], by["fallback"], by["error"],
            statistics.median(times) if times else 0.0, max(times) if times else 0.0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate all Stages on data/ clips")
    parser.add_argument("--stages", default="1,2,3",
                        help="Stages to run, e.g. 1,2. Stage 1 always runs.")
    args = parser.parse_args()
    wanted = {int(s) for s in args.stages.split(",") if s.strip()}

    stages = load_stages(wanted)
    extractors, _ = quietly(load_extractors)
    OUT.mkdir(exist_ok=True)
    csv_path = OUT / ("evaluation_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M"))

    files = clips()
    print("evaluating %d clips with Stages %s" % (len(files), sorted(stages)))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=COLUMNS)
        out.writeheader()
        for path in files:
            started = time.perf_counter()
            try:
                n = evaluate(path, stages, extractors, out)
                print("  %-38s %2d windows  %6.1f s" % (path.name, n, time.perf_counter() - started))
            except Exception as error:
                print("  %-38s FAILED  %s: %s" % (path.name, type(error).__name__, error))
            f.flush()

    print("\nwrote %s" % csv_path)
    summarise(csv_path)


if __name__ == "__main__":
    main()
