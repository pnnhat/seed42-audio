"""Genre / mood / valence-arousal classification.

Primary path: Essentia's pretrained DEAM model, MusiCNN embeddings feeding
a valence/arousal regression head trained on continuous human annotations
of music (Soleymani et al., DEAM dataset). Falls back automatically, on
any failure, essentia not installed, model files missing, a segment the
model cannot process, to the self-contained heuristic that estimates
valence and arousal from this segment's own brightness, energy and mode.
The heuristic was built first for the 30 August demo and is kept
permanently as a safety net rather than removed, so a classification
always exists even on a machine where essentia is not fully working.

Model files (not committed, large binaries, see .gitignore). Download
once per machine before the essentia path will actually run:
  mkdir -p models
  curl -L -o models/msd-musicnn-1.pb \
      https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb
  curl -L -o models/deam-msd-musicnn-2.pb \
      https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb
Without them, extract() silently uses the fallback for every segment.
"""

from pathlib import Path

import numpy as np
import librosa

# --- Essentia primary path --------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
_MUSICNN_MODEL = _MODELS_DIR / "msd-musicnn-1.pb"
_DEAM_MODEL = _MODELS_DIR / "deam-msd-musicnn-2.pb"
_ESSENTIA_SR = 16000  # fixed by the pretrained models, not tunable

try:
    from essentia.standard import TensorflowPredict2D, TensorflowPredictMusiCNN

    _ESSENTIA_IMPORTED = True
except ImportError:
    _ESSENTIA_IMPORTED = False

_embedding_model = None
_deam_model = None


def _load_essentia_models():
    """Load both TensorFlow graphs once and cache them for later calls.
    Raises if the model files are not present, extract() catches this the
    same as any other essentia failure and falls back.
    """
    global _embedding_model, _deam_model
    if _embedding_model is None:
        _embedding_model = TensorflowPredictMusiCNN(
            graphFilename=str(_MUSICNN_MODEL), output="model/dense/BiasAdd"
        )
    if _deam_model is None:
        _deam_model = TensorflowPredict2D(
            graphFilename=str(_DEAM_MODEL), output="model/Identity"
        )
    return _embedding_model, _deam_model


def _essentia_extract(samples, sr):
    """Valence and arousal from Essentia's pretrained DEAM model.

    Raises on any failure. extract() is the only caller and always wraps
    this in a try/except, falling back to _fallback_extract().
    """
    if not _ESSENTIA_IMPORTED:
        raise RuntimeError("essentia is not installed")

    embedding_model, deam_model = _load_essentia_models()

    # The models were trained on 16 kHz mono audio, resample this segment
    # to match rather than reading anything from disk again.
    audio_16k = librosa.resample(samples, orig_sr=sr, target_sr=_ESSENTIA_SR)
    audio_16k = audio_16k.astype(np.float32)

    embeddings = embedding_model(audio_16k)
    predictions = np.asarray(deam_model(embeddings))

    # DEAM predicts valence and arousal on a 1 to 9 scale. A segment longer
    # than the model's internal patch size comes back as one row per
    # patch, average them into a single reading for the segment.
    if predictions.ndim > 1:
        predictions = predictions.mean(axis=0)
    valence_raw, arousal_raw = predictions

    # Rescale 1..9 to -1..1 so the output matches the fallback's
    # convention regardless of which path produced it.
    valence = (valence_raw - 5.0) / 4.0
    arousal = (arousal_raw - 5.0) / 4.0

    return {
        "valence": round(max(-1.0, min(1.0, float(valence))), 3),
        "arousal": round(max(-1.0, min(1.0, float(arousal))), 3),
    }


# --- Fallback path (Phase 1, kept permanently as a safety net) -------------

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


def _fallback_extract(samples, sr):
    """Fallback valence and arousal for one segment, from this segment's
    own brightness, energy and mode.

    Returns valence and arousal in [-1, 1], Russell's circumplex
    convention: negative valence is darker or sadder, negative arousal is
    calmer. Brightness and mode drive valence, energy drives arousal, with
    brightness giving arousal a smaller push too (a bright, loud segment
    reads as more energetic than a bright, quiet one).
    """
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


# --- Public entry point ------------------------------------------------------


def extract(samples, sr):
    """Valence and arousal for one segment, essentia's pretrained DEAM
    model when it is available and working, the brightness/energy/mode
    heuristic otherwise.
    """
    if samples.size < int(sr * _MIN_SAMPLES_SECONDS):
        return {}

    try:
        return _essentia_extract(samples, sr)
    except Exception as error:
        print(f"[mood.classify] essentia path failed, using fallback: {error}")
        return _fallback_extract(samples, sr)
