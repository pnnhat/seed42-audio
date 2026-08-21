"""Entry point: play a song in segments and print each window as it arrives."""

from seed42_audio.io.stream import AudioStream

if __name__ == "__main__":
    stream = AudioStream("data/test.mp3", window=6.0, hop=1.0)
    for timestamp, segment in stream.segments():
        print(f"t={timestamp:5.1f}s   {len(segment)} samples")
