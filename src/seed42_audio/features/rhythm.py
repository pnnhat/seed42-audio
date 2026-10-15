# Rhythm features from one audio segment: tempo, onsets, beats (causal, Phase 1).

import numpy as np
import librosa

# Detected beats must be at least this many times stronger than the average onset strength.
# Measured on the corpus: white noise 1.26, sustained chords 2.66, quietest real track 2.88,
# click track 15.4. Below this the audio has no clear beat, so no tempo is reported.
MIN_BEAT_RATIO = 2.75

# Return tempo (BPM), onset times and beat times for one segment.
#
# Times are seconds from the start of the segment, not the whole track.
# MusicState keeps tempo; onset_times and beat_times are for the mapping stage.
def extract(samples, sr):
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr)
    # Beat confidence: how much stronger the beats are than the average onset strength
    average_strength = float(np.mean(onset_env))
    if len(beat_frames) > 0 and average_strength > 0:
        beat_ratio = float(np.mean(onset_env[beat_frames]) / average_strength)
    else:
        beat_ratio = 0.0
    tempo_value = float(np.atleast_1d(tempo)[0])
    if beat_ratio < MIN_BEAT_RATIO:
        tempo_value = 0.0
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr)
    return {
        "tempo": tempo_value,
        "onset_times": librosa.frames_to_time(onset_frames,
                                               sr=sr).tolist(),
        "beat_times": librosa.frames_to_time(beat_frames,
                                              sr=sr).tolist(),
    }