"""Spectral centroid, flatness, RMS, bass energy."""
import librosa
import numpy as np

def extract(samples, sr) -> dict:
    # Brightness: average frequency "center of mass" of the segment
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    brightness = float(np.mean(centroid))

    # Loudness: root-mean-square energy of the waveform
    loudness = librosa.feature.rms(y=samples)
    loudness = float(np.mean(loudness))

    # Flatness: how noise-like (flat spectrum) vs tonal (peaked spectrum) the sound is
    flatness = librosa.feature.spectral_flatness(y=samples)
    flatness = float(np.mean(flatness))

    # Bass energy: share of total spectral energy below 250 Hz
    stft = np.abs(librosa.stft(samples))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 250
    bass_energy = float(np.sum(stft[bass_mask]) / np.sum(stft))

    return {
        "brightness": brightness,
        "loudness": loudness,
        "flatness": flatness,
        "bass_energy": bass_energy,
    }