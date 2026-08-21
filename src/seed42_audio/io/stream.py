"""Feeds audio in past-X-second segments only (causal). Never reads the whole file at once."""

import librosa


class AudioStream:
    """Simulates a live audio feed from a file.

    Yields the most recent `window` seconds of audio every `hop` seconds, so
    downstream code only ever sees the past, never the whole track and never
    future samples. Later this same interface can be backed by a real line-in feed.
    """

    def __init__(self, path, sr=22050, window=6.0, hop=1.0):
        self.path = path
        self.sr = sr
        self.window = window  # seconds of audio each segment holds
        self.hop = hop  # seconds between segments

    def segments(self):
        """Yield (timestamp, samples) for each window as the track plays.

        timestamp is the time in seconds of the segment's trailing edge.
        samples is a numpy array of the last `window` seconds of audio.
        """
        y, sr = librosa.load(self.path, sr=self.sr, mono=True)
        win_len = int(self.window * sr)
        hop_len = int(self.hop * sr)

        for end in range(win_len, len(y) + hop_len, hop_len):
            end = min(end, len(y))
            start = max(0, end - win_len)
            yield end / sr, y[start:end]
