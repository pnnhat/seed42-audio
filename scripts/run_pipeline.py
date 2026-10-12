# Entry point: play songs back to back and print a classification every 60 seconds.

import os

os.chdir(r"C:\Users\phamn\Documents\Project\seed42")
print(os.getcwd())
print(os.path.exists("data/bright_energetic_3.mp3"))

from seed42_audio.pipeline import run

if __name__ == "__main__":
    songs = [
        "data/bright_energetic_3.mp3",
    ]
    run(songs, window=30.0, hop=1.0, emit_every=60.0)

run(["data/bright_energetic_3.mp3"], window=30.0, hop=1.0, emit_every=10.0)
