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