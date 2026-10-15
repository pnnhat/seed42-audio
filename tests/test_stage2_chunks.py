import time

import numpy as np

from seed42_audio.stages import stage2


class FakeCLAP:
    """Record each audio chunk passed to the embedding model."""

    def __init__(self):
        self.chunk_lengths = []

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        self.chunk_lengths.append(x.shape[-1])
        return np.ones((1, 64), dtype=np.float32)


def test_thirty_second_audio_uses_three_ten_second_chunks(monkeypatch):
    fake = FakeCLAP()
    monkeypatch.setattr(stage2, "_CLAP", fake)

    sample_rate = 48000
    audio = np.zeros(sample_rate * 30, dtype=np.float32)

    embedding = stage2._embed_audio_window(audio, time.perf_counter())

    assert len(fake.chunk_lengths) == 3
    assert fake.chunk_lengths == [480000, 480000, 480000]
    assert embedding.shape == (64,)


def test_real_thirty_second_window_prepares_and_uses_three_chunks(
    monkeypatch,
):
    fake = FakeCLAP()
    monkeypatch.setattr(stage2, "_CLAP", fake)

    audio = np.zeros(22050 * 30, dtype=np.float32)

    prepared_audio = stage2._prepare_audio(audio)

    assert prepared_audio.dtype == np.float32
    assert abs(len(prepared_audio) - (48000 * 30)) <= 1

    embedding = stage2._embed_audio_window(
        prepared_audio, time.perf_counter()
    )

    assert len(fake.chunk_lengths) == 3
    assert fake.chunk_lengths == [480000, 480000, 480000]
    assert embedding.shape == (64,)
