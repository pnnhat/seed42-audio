import numpy as np
import librosa


def extract(samples, sr) -> dict:
    """Key and mode features from one audio segment."""

    # Ensure input is a 1D numpy array.
    samples = np.asarray(samples).flatten()

    # Guard against empty input.
    if samples.size == 0:
        return {
            "chroma": np.zeros(12),
            "key": "C",
            "mode": "major",
        }

    # Compute chroma from CQT.
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)

    # Average across time frames.
    chroma_vec = np.mean(chroma, axis=1).astype(float)

    # Normalize chroma.
    chroma_vec = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-12)

    # Major and minor pitch-class profiles.
    major = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
         2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=float,
    )

    minor = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
         2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=float,
    )

    # Rotate templates so the tonic is at each possible pitch class.
    major_profiles = np.stack(
        [np.roll(major, root) for root in range(12)],
        axis=0,
    )

    minor_profiles = np.stack(
        [np.roll(minor, root) for root in range(12)],
        axis=0,
    )

    # Normalize templates.
    major_profiles = major_profiles / (
        np.linalg.norm(major_profiles, axis=1, keepdims=True) + 1e-12
    )

    minor_profiles = minor_profiles / (
        np.linalg.norm(minor_profiles, axis=1, keepdims=True) + 1e-12
    )

    # Correlate chroma against every possible key.
    major_corr = major_profiles @ chroma_vec
    minor_corr = minor_profiles @ chroma_vec

    # Find strongest match.
    major_root = int(np.argmax(major_corr))
    minor_root = int(np.argmax(minor_corr))

    major_strength = float(major_corr[major_root])
    minor_strength = float(minor_corr[minor_root])

    pitch_classes = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    # Keep whichever mode has the stronger match.
    if major_strength >= minor_strength:
        return {
            "chroma": chroma_vec,
            "key": pitch_classes[major_root],
            "mode": "major",
        }

    return {
        "chroma": chroma_vec,
        "key": pitch_classes[minor_root],
        "mode": "minor",
    }