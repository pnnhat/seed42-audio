import numpy as np
import librosa


def extract(samples, sr) -> dict:
    """Key and mode features from one audio segment (causal, Phase 1).

    From the segment's trailing audio, compute chroma (CQT), then estimate:
    - key (pitch class root)
    - mode (major vs minor)

    The chroma is reduced to one 12-bin vector by averaging across time frames,
    then correlated against rotated major and minor templates. The stronger
    match is returned.
    """
    # Ensure we treat the input as a 1D numpy array.
    samples = np.asarray(samples)

    # Guard against empty input, keep it simple.
    if samples.size == 0:
        return {"key": "C", "mode": "major"}

    # Compute chroma from CQT, shape is (12, frames).
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)

    # Reduce across time frames to a single 12-bin pitch-class vector.
    chroma_vec = np.mean(chroma, axis=1).astype(float)

    # Normalize for more stable template correlations.
    chroma_vec = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-12)

    # Major/minor pitch-class templates (classic Krumhansl-Schmuckler style).
    major = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=float,
    )
    minor = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=float,
    )

    # Build one template per possible key root by rotating the profiles.
    major_profiles = np.stack([np.roll(major, -root) for root in range(12)], axis=0)
    minor_profiles = np.stack([np.roll(minor, -root) for root in range(12)], axis=0)

    # Normalize templates so dot product behaves like cosine similarity.
    major_profiles = major_profiles / (
        np.linalg.norm(major_profiles, axis=1, keepdims=True) + 1e-12
    )
    minor_profiles = minor_profiles / (
        np.linalg.norm(minor_profiles, axis=1, keepdims=True) + 1e-12
    )

    # Correlate chroma against all roots for both modes.
    major_corr = major_profiles @ chroma_vec  # shape (12,)
    minor_corr = minor_profiles @ chroma_vec  # shape (12,)

    # Pick best matching root and mode, keep the stronger of major vs minor.
    major_root = int(np.argmax(major_corr))
    minor_root = int(np.argmax(minor_corr))

    major_strength = float(major_corr[major_root])
    minor_strength = float(minor_corr[minor_root])

    # Pitch class labels for roots (simple, sharp spellings only).
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Return the stronger mode match.
    if major_strength >= minor_strength:
        return {"key": pitch_classes[major_root], "mode": "major"}
    return {"key": pitch_classes[minor_root], "mode": "minor"}
