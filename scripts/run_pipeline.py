"""Entry point: play songs back to back and print a classification every 60 seconds."""

import os

os.chdir(r"C:\Users\phamn\Documents\Project\seed42")
print(os.getcwd())
print(os.path.exists("data/test.wav"))

from seed42_audio.pipeline import run

if __name__ == "__main__":
    songs = [
        "data/test.wav",
    ]
    run(songs, window=30.0, hop=1.0, emit_every=60.0)

run(["data/test.wav"], window=30.0, hop=1.0, emit_every=10.0)
