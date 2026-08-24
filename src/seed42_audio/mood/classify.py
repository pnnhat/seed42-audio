"""Genre / mood / valence-arousal classification.

Phase 1 fallback: estimate valence and arousal directly from this segment's
own brightness, energy and mode. The shared extract(samples, sr) -> dict
signature does not receive the other feature modules' outputs (every
extractor only ever sees raw samples and sr, never another module's
result), so this module computes its own lightweight versions of those
three signals instead of depending on spectral.py or chroma.py being
finished. That keeps a valence/arousal reading always available, even if
those files are still stubs, which is what the 30 August demo needs.

Upgrade path: swap the body of extract() for Essentia's pretrained mood
and valence/arousal models once that dependency is confirmed working
(essentia-tensorflow on Windows, plain essentia on Mac, Linux or WSL).
Keep this function's signature and output keys unchanged so the swap is a
drop-in for the rest of the pipeline.
"""

import numpy as np
import librosa

# Rough spectral centroid range, in Hz, covering dark or muffled timbre up
# to bright or harsh timbre. Only used to squash brightness into 0 to 1
# before it is combined with the other signals below.
_BRIGHTNESS_LOW = 500.0
_BRIGHTNESS_HIGH = 4000.0

# RMS amplitude range used to squash energy into 0 to 1. 0.3 is already a
# loud, near-clipping segment for normalised audio.
_ENERGY_LOW = 0.0
_ENERGY_HIGH = 0.3

# Krumhansl-Schmuckler major and minor key profiles, C-rooted. Correlating
# a chroma vector against every rotation of these and keeping the best
# major score versus the best minor score gives mode without needing a
# key estimate first.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Shortest segment worth analysing. Below this, chroma_cqt has too little
# audio for a stable estimate, so skip rather than return noise.
_MIN_SAMPLES_SECONDS = 0.5


def _normalise(value, low, high):
    """Clamp value to [low, high] then rescale it to 0 to 1."""
    value = max(low, min(high, value))
    return (value - low) / (high - low)


def _estimate_mode(samples, sr):
    """Major (1.0) or minor (0.0), from chroma correlated against the
    Krumhansl-Schmuckler profiles.
    """
    chroma_mean = librosa.feature.chroma_cqt(y=samples, sr=sr).mean(axis=1)

    major_score = max(
        np.corrcoef(chroma_mean, np.roll(_MAJOR_PROFILE, i))[0, 1] for i in range(12)
    )
    minor_score = max(
        np.corrcoef(chroma_mean, np.roll(_MINOR_PROFILE, i))[0, 1] for i in range(12)
    )
    return 1.0 if major_score >= minor_score else 0.0


def extract(samples, sr):
    """Fallback valence and arousal for one segment.

    Returns valence and arousal in [-1, 1], Russell's circumplex
    convention: negative valence is darker or sadder, negative arousal is
    calmer. Brightness and mode drive valence, energy drives arousal, with
    brightness giving arousal a smaller push too (a bright, loud segment
    reads as more energetic than a bright, quiet one).
    """
    if samples.size < int(sr * _MIN_SAMPLES_SECONDS):
        return {}

    rms = float(np.sqrt(np.mean(samples**2)))
    energy = _normalise(rms, _ENERGY_LOW, _ENERGY_HIGH)

    centroid = float(librosa.feature.spectral_centroid(y=samples, sr=sr).mean())
    brightness = _normalise(centroid, _BRIGHTNESS_LOW, _BRIGHTNESS_HIGH)

    mode = _estimate_mode(samples, sr)

    valence = 0.6 * (2 * brightness - 1) + 0.4 * (2 * mode - 1)
    arousal = 0.7 * (2 * energy - 1) + 0.3 * (2 * brightness - 1)

    return {
        "valence": round(max(-1.0, min(1.0, valence)), 3),
        "arousal": round(max(-1.0, min(1.0, arousal)), 3),
    }
