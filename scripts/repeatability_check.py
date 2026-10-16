from pathlib import Path
from datetime import datetime
import contextlib
import csv
import io
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import evaluate_stages as ev
import evaluation as scoring
from seed42_audio.state.music_state import MusicState
from seed42_audio.stages import stage2


def run_quietly(state):
    buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        result = stage2.generate(state)
    elapsed = time.perf_counter() - start
    output = buffer.getvalue()
    prompt = result.get("prompt", "") if isinstance(result, dict) else ""
    negative = result.get("negative_prompt", "") if isinstance(result, dict) else ""

    if not prompt:
        status = "error"
    elif "fallback" in output.lower():
        status = "fallback"
    else:
        status = "ok"

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "latency_s": elapsed,
        "status": status,
    }


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"repeatability_{timestamp}.txt"
    csv_path = results_dir / f"repeatability_windows_{timestamp}.csv"

    real_clips = [
        path for path in ev.clips()
        if ev.group_of(path) in scoring.QUADRANTS
    ]

    rows = []
    repeated = None
    repeated_results = []

    for path in real_clips:
        limit = ev.LIMIT
        for audio_timestamp, samples in ev.windows(path, limit):
            state = MusicState(
                timestamp=float(audio_timestamp),
                samples=samples,
            )
            result = run_quietly(state)

            row = {
                "file": path.name,
                "group": ev.group_of(path),
                "timestamp": round(float(audio_timestamp), 1),
                "stage": "2",
                "status": result["status"],
                "latency_s": result["latency_s"],
                "prompt": result["prompt"],
                "negative_prompt": result["negative_prompt"],
            }
            rows.append(row)

            if repeated is None:
                repeated = (path.name, float(audio_timestamp), state)

    # Repeat the exact same in-memory window five times.
    # The first call above is also recorded in the full-window results.
    if repeated is not None:
        name, audio_timestamp, state = repeated
        repeated_results = [run_quietly(state) for _ in range(5)]

    prompts = [r["prompt"] for r in rows if r["status"] == "ok"]
    latencies = [r["latency_s"] for r in rows]
    ok_count = sum(r["status"] == "ok" for r in rows)
    fallback_count = sum(r["status"] == "fallback" for r in rows)
    error_count = sum(r["status"] == "error" for r in rows)

    arousal_right, arousal_total, ambiguous = scoring.stage2_arousal(rows)

    repeat_prompts = [r["prompt"] for r in repeated_results]
    repeat_identical = (
        len(repeat_prompts) == 5
        and all(p == repeat_prompts[0] for p in repeat_prompts)
    )

    try:
        mps_available = torch.backends.mps.is_available()
    except Exception:
        mps_available = False

    report = [
        "STAGE 2 REPEATABILITY AND PERFORMANCE CHECK",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"Apple MPS available: {mps_available}",
        "",
        "REPEATED-WINDOW TEST",
    ]

    if repeated is not None:
        report += [
            f"Window: {name} at {audio_timestamp:.1f}s",
            f"Identical prompt across five repeated calls: {repeat_identical}",
            f"Distinct prompts across those five calls: {len(set(repeat_prompts))}",
            f"Repeated-call statuses: {[r['status'] for r in repeated_results]}",
            f"Prompt: {repeat_prompts[0] if repeat_prompts else '(none)'}",
            f"Negative prompt: {repeated_results[0]['negative_prompt'] if repeated_results else '(none)'}",
        ]
    else:
        report.append("No real-track windows were found.")

    report += [
        "",
        "REAL-TRACK WINDOW RESULTS",
        f"Real clips: {len(real_clips)}",
        f"Windows processed: {len(rows)}",
        f"Successful own prompts: {ok_count}",
        f"Fallbacks: {fallback_count}",
        f"Errors: {error_count}",
        f"Distinct successful prompts: {len(set(prompts))} of {len(prompts)}",
        f"Median latency: {statistics.median(latencies):.3f}s" if latencies else "Median latency: N/A",
        f"Maximum latency: {max(latencies):.3f}s" if latencies else "Maximum latency: N/A",
        (
            f"Stage 2 atmosphere arousal: {arousal_right}/{arousal_total} "
            f"= {100 * arousal_right / arousal_total:.0f}% "
            f"({ambiguous} ambiguous excluded)"
            if arousal_total else
            f"Stage 2 atmosphere arousal: no scorable phrases ({ambiguous} ambiguous)"
        ),
        "",
        "PHASE B BASELINE (previously scored with the same evaluation script)",
        "Distinct Stage 2 prompts: 25 of 52",
        "Stage 2 atmosphere arousal: 44 of 50 = 88% (2 ambiguous excluded)",
        "Stage 2 median latency: 0.188s",
        "Stage 2 maximum latency: 0.249s",
        "",
        "CPU NOTE",
        "CUDA/MPS availability reports hardware capability, not the device",
        "actually selected by CLAP. Confirm the CLAP device from its startup log;",
        "do not claim CPU-only execution from these flags alone.",
    ]

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    fields = [
        "file", "group", "timestamp", "stage", "status",
        "latency_s", "prompt", "negative_prompt",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n".join(report))
    print(f"\nSaved report: {report_path}")
    print(f"Saved window CSV: {csv_path}")


if __name__ == "__main__":
    main()
