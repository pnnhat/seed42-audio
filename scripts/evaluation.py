# Score and plot an evaluation run. Reads the CSV that evaluate_stages.py writes,
# compares predictions with the known answers, and draws the figures.
#
#   python scripts/evaluation.py                     # newest CSV in results/
#   python scripts/evaluation.py results/evaluation_20260930_0220.csv
#
# Writes next to the CSV:
#   scores.txt               every number below, ready to quote in the report
#   synthetic_checks.csv     each synthetic clip against its known answer
#   confusion_matrix.csv     mood quadrant, actual against predicted
#   fig_mood_map.png, fig_confusion.png, fig_latency.png, fig_transition.png

import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SR = 22050
HOP = 512  # librosa's default hop, which sets the tempo steps
QUADRANTS = ["bright_energetic", "bright_calm", "dark_energetic", "dark_calm"]
NAMES = {
    "bright_energetic": "Bright energetic",
    "bright_calm": "Bright calm",
    "dark_energetic": "Dark energetic",
    "dark_calm": "Dark calm",
}

# Stage 2's atmosphere phrases grouped by the arousal they describe, fixed in
# advance. The last phrase of every Stage 2 prompt is from this group. Phrases
# not listed here count as ambiguous and are left out of the arousal score.
CALM_PHRASES = {
    "calm and meditative mood",
    "dreamy and weightless space",
    "sparse and lonely quiet",
    "warm and nostalgic haze",
}
ENERGETIC_PHRASES = {
    "aggressive and chaotic energy",
    "uplifting and euphoric tone",
    "dense and overwhelming intensity",
    "raw and industrial texture",
}

BUDGETS = {"2": 5.0, "3": 8.0}
SEND_LIMIT = 12.0
SWITCH, WINDOW = 45.0, 30.0


# --- Loading ------------------------------------------------------------------


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def windows(rows, real=None):
    # One row per window, taken from Stage 1 since features repeat on every Stage.
    out = [r for r in rows if r["stage"] == "1"]
    if real is True:
        out = [r for r in out if not r["file"].startswith("synth_")]
    return out


def clip(rows, name):
    return [r for r in rows if r["stage"] == "1" and r["file"] == name]


def quadrant(valence, arousal):
    return (
        ("bright" if valence >= 0 else "dark")
        + "_"
        + ("energetic" if arousal >= 0 else "calm")
    )


# --- Scoring ------------------------------------------------------------------


def tempo_steps(true_bpm):
    # The tracker can only report 60 * sr / (hop * lag) for a whole-number lag,
    # so a reading counts as correct if it is either step around the true tempo.
    lag = 60 * SR / (HOP * true_bpm)
    return [60 * SR / (HOP * l) for l in (math.floor(lag), math.ceil(lag))]


def on_step(tempo, true_bpm):
    return any(abs(num(tempo) - s) < 0.1 for s in tempo_steps(true_bpm))


def synthetic_checks(rows):
    checks = []

    def add(name, rs, test):
        checks.append((name, sum(1 for r in rs if test(r)), len(rs)))

    add(
        "120 BPM clicks, tempo on a nearest step",
        clip(rows, "synth_clicks_120bpm.wav"),
        lambda r: on_step(r["tempo"], 120),
    )
    add(
        "C major chord, key and mode",
        clip(rows, "synth_chord_c_major.wav"),
        lambda r: (r["key"], r["mode"]) == ("C", "major"),
    )
    add(
        "A minor chord, key and mode",
        clip(rows, "synth_chord_a_minor.wav"),
        lambda r: (r["key"], r["mode"]) == ("A", "minor"),
    )

    noise = [num(r["flatness"]) for r in clip(rows, "synth_white_noise.wav")]
    chords = [
        num(r["flatness"])
        for r in clip(rows, "synth_chord_c_major.wav")
        + clip(rows, "synth_chord_a_minor.wav")
    ]
    if noise and chords:
        checks.append(
            (
                "White noise flatter than every chord window",
                int(min(noise) > max(chords)),
                1,
            )
        )

    trans = clip(rows, "synth_transition_90_to_140bpm.wav")
    before = [r for r in trans if num(r["timestamp"]) <= SWITCH]
    after = [r for r in trans if num(r["timestamp"]) >= SWITCH + WINDOW]
    add(
        "Transition before the change, 90 BPM step and A minor",
        before,
        lambda r: on_step(r["tempo"], 90) and (r["key"], r["mode"]) == ("A", "minor"),
    )
    add(
        "Transition after the change, 140 BPM step and F# major",
        after,
        lambda r: on_step(r["tempo"], 140) and (r["key"], r["mode"]) == ("F#", "major"),
    )

    silence = [r for r in rows if r["file"] == "synth_silence.wav"]
    checks.append(
        (
            "Silence runs without an error row",
            int(bool(silence) and not any(r["status"] == "error" for r in silence)),
            1,
        )
    )
    return checks


def confusion(rows):
    real = windows(rows, real=True)
    matrix = np.zeros((4, 4), dtype=int)
    for r in real:
        if r["group"] in QUADRANTS:
            predicted = quadrant(num(r["valence"]), num(r["arousal"]))
            matrix[QUADRANTS.index(r["group"]), QUADRANTS.index(predicted)] += 1
    return matrix


def half_accuracy(rows):
    # Valence and arousal scored separately, as a dark or bright and a calm or
    # energetic call against the quadrant each track was chosen for.
    real = [r for r in windows(rows, real=True) if r["group"] in QUADRANTS]
    valence = sum(
        (num(r["valence"]) >= 0) == r["group"].startswith("bright") for r in real
    )
    arousal = sum(
        (num(r["arousal"]) >= 0) == r["group"].endswith("energetic") for r in real
    )
    return valence, arousal, len(real)


def stage2_arousal(rows):
    right = ambiguous = total = 0
    for r in rows:
        if (
            r["stage"] != "2"
            or r["file"].startswith("synth_")
            or r["group"] not in QUADRANTS
        ):
            continue
        phrase = r["prompt"].split(",")[-1].strip()
        if phrase in CALM_PHRASES:
            energetic = False
        elif phrase in ENERGETIC_PHRASES:
            energetic = True
        else:
            ambiguous += 1
            continue
        total += 1
        right += energetic == r["group"].endswith("energetic")
    return right, total, ambiguous


def stage_summary(rows):
    lines = []
    for stage in ("1", "2", "3"):
        mine = [r for r in rows if r["stage"] == stage]
        if not mine:
            continue
        times = [num(r["latency_s"]) for r in mine if r["latency_s"]]
        real = [r["prompt"] for r in mine if not r["file"].startswith("synth_")]
        counts = {
            s: sum(r["status"] == s for r in mine) for s in ("ok", "fallback", "error")
        }
        lines.append(
            "Stage %s  median %.3f s, max %.3f s, own prompt %d, fallback %d, error %d, "
            "distinct prompts on real tracks %d of %d"
            % (
                stage,
                statistics.median(times),
                max(times),
                counts["ok"],
                counts["fallback"],
                counts["error"],
                len(set(real)),
                len(real),
            )
        )
    return lines


def score(rows, folder):
    lines = ["SYNTHETIC CLIPS AGAINST THE ANSWER KEY"]
    checks = synthetic_checks(rows)
    for name, passed, total in checks:
        lines.append("  %-58s %d / %d" % (name, passed, total))
    with open(folder / "synthetic_checks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "passed", "total"])
        writer.writerows(checks)

    matrix = confusion(rows)
    with open(folder / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted"] + QUADRANTS)
        for name, row in zip(QUADRANTS, matrix):
            writer.writerow([name] + list(row))

    hits, total = int(np.trace(matrix)), int(matrix.sum())
    lines += [
        "",
        "MOOD QUADRANT, ACTUAL AGAINST PREDICTED (real tracks, windows)",
        "  overall %d / %d = %.0f%% (chance 25%%)"
        % (hits, total, 100 * hits / max(total, 1)),
    ]
    for i, name in enumerate(QUADRANTS):
        lines.append("  %-18s %d / %d" % (name, matrix[i, i], matrix[i].sum()))
    valence, arousal, n = half_accuracy(rows)
    lines.append(
        "  valence, dark or bright   %d / %d = %.0f%% (chance 50%%)"
        % (valence, n, 100 * valence / max(n, 1))
    )
    lines.append(
        "  arousal, calm or energetic %d / %d = %.0f%% (chance 50%%)"
        % (arousal, n, 100 * arousal / max(n, 1))
    )
    right, counted, ambiguous = stage2_arousal(rows)
    if counted:
        lines.append(
            "  Stage 2 arousal from its atmosphere phrase %d / %d = %.0f%% (%d ambiguous left out)"
            % (right, counted, 100 * right / counted, ambiguous)
        )

    lines += ["", "STAGES"] + ["  " + line for line in stage_summary(rows)]
    text = "\n".join(lines)
    (folder / "scores.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return matrix


# --- Figures ------------------------------------------------------------------

# Colours, the validated reference palette (slots 1 and 2) and its chart chrome.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TARGET = "#e6effb"  # light wash behind a panel's target quadrant
CONTEXT = "#c3c2b7"  # other tracks, shown for context only
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BAND = "#f0efec"  # neutral shading for the windows that span the change
# One-hue blue ramp, light to dark, for the confusion matrix counts.
RAMP = ["#f2f6fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "legend.frameon": False,
    }
)

STAGE_LABELS = {
    "1": "Stage 1\nrule table",
    "2": "Stage 2\nCLAP retrieval",
    "3": "Stage 3\nlocal LLM",
}
PANELS = [  # mood map panels, placed where each quadrant sits on the map
    ("dark_energetic", "Dark, energetic", (-1, 0), (0, 1)),
    ("bright_energetic", "Bright, energetic", (0, 1), (0, 1)),
    ("dark_calm", "Dark, calm", (-1, 0), (-1, 0)),
    ("bright_calm", "Bright, calm", (0, 1), (-1, 0)),
]


def seconds(value):
    if value < 0.01:
        return "under 0.01 s"
    return "%.2f s" % value


def fig_mood_map(rows, out):
    real = [r for r in windows(rows, real=True) if r["group"] in QUADRANTS]
    valence = np.array([num(r["valence"]) for r in real])
    arousal = np.array([num(r["arousal"]) for r in real])
    groups = np.array([r["group"] for r in real])

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), sharex=True, sharey=True)
    for ax, (group, title, (x0, x1), (y0, y1)) in zip(axes.flat, PANELS):
        ax.axvspan(
            x0, x1, ymin=(y0 + 1) / 2, ymax=(y1 + 1) / 2, color=TARGET, zorder=0, lw=0
        )
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        others = groups != group
        ax.scatter(
            valence[others], arousal[others], s=14, color=CONTEXT, lw=0, zorder=2
        )
        mine = ~others
        ax.scatter(
            valence[mine],
            arousal[mine],
            s=40,
            color=BLUE,
            edgecolor=SURFACE,
            lw=1,
            zorder=3,
        )
        inside = int(
            np.sum(
                mine
                & (valence >= x0)
                & (valence <= x1)
                & (arousal >= y0)
                & (arousal <= y1)
            )
        )
        ax.set_title("%s\n%d of %d in target" % (title, inside, int(np.sum(mine))))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("Valence  (dark  to  bright)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Arousal  (calm  to  energetic)")
    fig.suptitle("Mood readings by intended quadrant", y=0.985, fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(top=0.89)
    fig.savefig(out)
    plt.close(fig)


def fig_confusion(matrix, out):
    labels = [NAMES[q].replace(" ", "\n") for q in QUADRANTS]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", RAMP)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.pcolormesh(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=max(int(matrix.max()), 1),
        edgecolors=SURFACE,
        linewidth=2,
    )
    for i in range(4):
        for j in range(4):
            dark = matrix[i, j] > 0.55 * matrix.max()
            ax.text(
                j + 0.5,
                i + 0.5,
                matrix[i, j],
                ha="center",
                va="center",
                fontsize=12,
                color=SURFACE if dark else INK,
            )
    ax.set_xticks(np.arange(4) + 0.5, labels)
    ax.set_yticks(np.arange(4) + 0.5, labels)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    hits, total = int(np.trace(matrix)), int(matrix.sum())
    ax.set_title(
        "Mood quadrant, actual vs predicted (%d of %d correct)" % (hits, total),
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_latency(rows, out):
    rng = np.random.default_rng(0)  # fixed jitter so reruns draw the same figure
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    stages = [s for s in ("1", "2", "3") if any(r["stage"] == s for r in rows)]
    fell_back = False
    tick_labels = []
    for i, stage in enumerate(stages, start=1):
        mine = [
            r for r in rows if r["stage"] == stage and r["status"] in ("ok", "fallback")
        ]
        t = np.array([max(num(r["latency_s"]), 1e-4) for r in mine])
        back = np.array([r["status"] == "fallback" for r in mine])
        x = i + rng.uniform(-0.18, 0.18, len(t))
        ax.scatter(
            x[~back], t[~back], s=30, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3
        )
        if back.any():
            fell_back = True
            ax.scatter(
                x[back],
                t[back],
                s=34,
                facecolor="none",
                edgecolor=ORANGE,
                lw=1.6,
                zorder=5,
            )
        label = STAGE_LABELS[stage]
        if len(t):
            median = float(np.median(t))
            ax.plot([i - 0.28, i + 0.28], [median, median], color=INK, lw=2, zorder=4)
            label += "\nmedian %s" % seconds(median)
        tick_labels.append(label)
        if stage in BUDGETS:
            ax.plot(
                [i - 0.4, i + 0.4], [BUDGETS[stage]] * 2, color=MUTED, lw=1.2, ls="--"
            )
            ax.annotate(
                "budget %g s" % BUDGETS[stage],
                (i + 0.42, BUDGETS[stage]),
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
    ax.axhline(SEND_LIMIT, color=MUTED, lw=1, ls=":")
    ax.annotate(
        "send abort %g s" % SEND_LIMIT,
        (len(stages) + 0.85, SEND_LIMIT * 1.15),
        ha="right",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 40)
    ax.set_xlim(0.5, len(stages) + 0.9)
    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Seconds per window (log scale)")
    ax.set_title("Time per window by Stage", fontsize=13, pad=12)
    if fell_back:
        ax.scatter([], [], s=30, color=BLUE, label="own prompt")
        ax.scatter(
            [],
            [],
            s=34,
            facecolor="none",
            edgecolor=ORANGE,
            lw=1.6,
            label="fell back to Stage 1",
        )
        ax.legend(loc="center left", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_transition(rows, out):
    points = sorted(
        clip(rows, "synth_transition_90_to_140bpm.wav"),
        key=lambda r: num(r["timestamp"]),
    )
    if not points:
        print("no transition clip in this run, skipping fig_transition.png")
        return
    t = np.array([num(r["timestamp"]) for r in points])
    panels = [
        ("tempo", "Tempo (BPM)", (90, 140)),
        ("energy", "Energy (RMS)", None),
        ("arousal", "Arousal", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.4, 7.4), sharex=True)
    for ax, (column, label, guides) in zip(axes, panels):
        y = np.array([num(r[column]) for r in points])
        ax.axvspan(SWITCH, SWITCH + WINDOW, color=BAND, lw=0, zorder=0)
        ax.axvline(SWITCH, color=MUTED, lw=1.2, ls="--", zorder=1)
        if guides:
            for g in guides:
                ax.axhline(g, color=AXIS, lw=1, ls=":", zorder=1)
                ax.annotate(
                    "%d" % g, (t.max() + 1, g), va="center", fontsize=8.5, color=INK_2
                )
        ax.plot(
            t,
            y,
            color=BLUE,
            lw=2,
            marker="o",
            ms=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            zorder=3,
        )
        ax.set_ylabel(label)
    # Label the detected key only where it changes, so the blend is visible.
    top = axes[0]
    previous = None
    for r, x in zip(points, t):
        key = "%s %s" % (r["key"], r["mode"])
        if key != previous:
            top.annotate(
                key,
                (x, num(r["tempo"])),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
            previous = key
    top.annotate(
        "change",
        (SWITCH, 1.02),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=8.5,
        color=INK_2,
    )
    top.annotate(
        "mixed windows",
        (SWITCH + WINDOW / 2, 1.02),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=8.5,
        color=INK_2,
    )
    lo, hi = top.get_ylim()
    top.set_ylim(lo, hi + (hi - lo) * 0.15)
    axes[-1].set_xlabel("Window end (s)")
    fig.suptitle("Readings across a change at 45 s", y=0.985, fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.subplots_adjust(top=0.925)
    fig.savefig(out)
    plt.close(fig)


# --- Entry --------------------------------------------------------------------


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        found = sorted(Path("results").glob("evaluation_*.csv"))
        if not found:
            sys.exit("no results/evaluation_*.csv found, run evaluate_stages.py first")
        csv_path = found[-1]
    rows = load(csv_path)
    folder = csv_path.parent
    print("scoring %s\n" % csv_path)
    matrix = score(rows, folder)
    fig_mood_map(rows, folder / "fig_mood_map.png")
    fig_confusion(matrix, folder / "fig_confusion.png")
    fig_latency(rows, folder / "fig_latency.png")
    fig_transition(rows, folder / "fig_transition.png")
    print("\nwrote scores and four figures to %s" % folder)


if __name__ == "__main__":
    main()
