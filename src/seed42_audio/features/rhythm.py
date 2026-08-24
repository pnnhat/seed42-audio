"""Rhythm features from one audio segment: tempo, onsets, 
    beats (causal, Phase 1)."""

import numpy as np
import librosa


def extract(samples, sr):
    """Return tempo (BPM), onset times and beat times for one segment.

    Times are seconds from the start of the segment, not the 
    whole track. MusicState keeps tempo; onset_times and 
    beat_times are for the mapping stage.
    """
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr)
    return {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "onset_times": librosa.frames_to_time(onset_frames,
                                               sr=sr).tolist(),
        "beat_times": librosa.frames_to_time(beat_frames,
                                              sr=sr).tolist(),
    }