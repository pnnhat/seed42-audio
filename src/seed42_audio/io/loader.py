"""Load a wav or mp3 file into audio samples for testing."""

import librosa


def load_audio(path, sr=22050):
    """Return (samples, sr) for an audio file, mono, at the given sample rate.

    samples is a 1D numpy array. sr is the sample rate actually used.
    """
    samples, sr = librosa.load(path, sr=sr, mono=True)
    return samples, sr
