# Work log

One row per piece of work. Add your own rows, newest at the top. Keep entries to one line: what you did, not how. Put the commit hash in the last column once it is pushed, or the file name if it is not code.

| Date | Time | Name | What was done | Commit or file |
|------|------|------|---------------|----------------|
| 2026-08-22 | 17:00 | Hatim | Set up local environment (conda, Python 3.11), ran stream test, added egg-info and scratch to .gitignore, created this log | docs/WORKLOG.md |
| 2026-08-22 | 18:30 | Tatiana | Set up local environment (Anaconda base), ran stream test on Mac | scripts/run_stream.py |
| 2026-08-24 | 13:00 | Jayden | Added pipeline loop that reads causal segments and emits a MusicState classification every X seconds | src/seed42_audio/pipeline.py |
| 2026-08-24 | 13:30 | Jayden | Defined MusicState dataclass, the shared per-window state the features fill and the output reads | src/seed42_audio/state/music_state.py |
| 2026-08-24 | 13:45 | Jayden | Finished causal streaming loop, handling clips shorter than the window | src/seed42_audio/io/stream.py |
| 2026-08-24 | 19:50 | Tatiana | Implemented spectral features (centroid/brightness, RMS/loudness, flatness, bass energy) in features/spectral.py, tested end-to-end against AudioStream | src/seed42_audio/features/spectral.py |
| 2026-08-24 | 20:00 | Philo | Mood classification fallback: valence and arousal from brightness, energy and mode, computed self-contained in extract(samples, sr) since it does not receive other modules' state | src/seed42_audio/mood/classify.py |
| 2026-08-24 | 21:30 | Philo | Wired essentia's pretrained DEAM model (MusiCNN embeddings, valence/arousal regression head) as the primary path, brightness/energy/mode heuristic kept as an automatic fallback on any essentia failure. Added essentia-tensorflow to requirements.txt and gitignored models/ | src/seed42_audio/mood/classify.py |
| 2026-08-24 | 23:20 | Hatim | Implemented features/rhythm.py (tempo, onsets, beats), verified against 120 BPM click track, tempo reads 129.2 due to frame quantisation, noted for evaluation | src/seed42_audio/features/rhythm.py |
| 2026-08-24 | 23:45 | Charmin | Implemented chroma features extraction and automatic key detection in features/chroma.py, returning the estimated musical key, mode (major vs. minor), and mean chroma vector in the segment extraction output | src/seed42_audio/features/chroma.py
| 2026-08-25 | 18:20 | Danial | Found run_pipeline.py hardcodes a Windows path on line 5 and calls run() a second time outside the main guard at emit_every=10.0 instead of 60.0, and run_stream.py loads data/test.mp3 which is not in the repo, so both fail on a fresh pull | scripts/run_pipeline.py |
| 2026-08-25 | 18:10 | Danial | Verified causal streaming loop on Mac against test.wav after pull, 132300 samples per 6s window through to end of clip | scripts/run_stream.py |
| 2026-08-22 | 14:30 | Danial | Set up local environment (Anaconda base, Python 3.13), installed dependencies and package, ran stream test on Mac | scripts/run_stream.py |


 
