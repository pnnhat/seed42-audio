# seed42-audio

The audio side of seed42, built by COMP3850 Group 21. seed42 restyles a live
camera feed with an image model and projects the result. This module listens
to the music, describes what it hears, and steers the visuals: each cycle it
turns the last 30 seconds of audio into a text prompt and four slider values,
and sends them to seed42's Control API.

It reads audio files today. Live microphone or line-in input is in progress.

## How it works

1. `io/stream.py` plays an audio file as if it were live. Every second it hands
   on the trailing 30 seconds of audio, never anything later.
2. `features/` and `mood/` turn a window into a `MusicState`: tempo, energy,
   brightness, flatness, key and mode, valence and arousal, bass energy and
   beat times.
3. A Stage turns the `MusicState` into a prompt and a negative prompt:
   - Stage 1, a rule table.
   - Stage 2, CLAP retrieval from the phrase pairs in `data/phrase_bank.json`.
   - Stage 3, Llama 3.2 through Ollama, behind a guardrail.

   Stages 2 and 3 fall back to Stage 1 whenever they cannot answer.
4. `output/mapping.py` turns the same `MusicState` into the four sliders:
   Creativity (`t_index_list`), and Depth, Edge and Consistency (the three
   `controlnets` scales).
5. `orchestrate.py` merges the two, keeps only the fields this session drives,
   and hands them to `output/writer.py`. The writer refuses any field that
   would reload the stream, retries a stream that is still starting, drops an
   update that gets no reply within 12 seconds, and ends the session if the
   stream has gone.

By default one update is sent per 60 seconds of audio, so the first one comes
60 seconds in. A file is read as fast as the disk allows, not in real time.

The contract between these parts, and with seed42's API, is in
`docs/seam-contract.md`. The design is in `docs/design-document.md`.

## Setup

All three Stages run in one conda environment, `seed42-clap`, with Python 3.11:

```
conda create -n seed42-clap python=3.11 -y
conda activate seed42-clap
pip install "numpy==1.26.4" torch torchaudio torchvision laion-clap
pip install librosa soundfile scipy pyyaml requests matplotlib pytest
pip install -e .
python -c "from seed42_audio.stages import stage2"
```

The last line downloads CLAP's checkpoint and text model the first time, then
prints `[stage2] CLAP ready`. CLAP runs on the CPU on a Mac. On Windows with an
NVIDIA GPU, install PyTorch's CUDA build first, using the selector on
pytorch.org.

For Stage 3, install Ollama from ollama.com, open it, and run
`ollama pull llama3.2`.

Essentia is left out on purpose. Without it the mood reading uses its fallback
estimate, and each window prints a line saying so.

## Settings

Copy `.env.example` to `.env`, fill it in, and load it into the terminal
before a run. `.env` is gitignored, so real values never reach the repository.

```
set -a; source .env; set +a        # macOS or Linux
$env:SEED42_STAGE = '1'            # Windows PowerShell, one setting at a time
```

| Setting | What it does |
|---|---|
| `SEED42_BASE_URL` | Where the Control API is. The mock's address unless set; `https://seed42.app` to go live |
| `SEED42_EMAIL`, `SEED42_PASSWORD` | The account seed42 issues. Any non-empty values work with the mock |
| `SEED42_STAGE` | Which Stage writes the prompt: 1, 2 or 3. Default 1 |
| `SEED42_FIELDS` | Which fields this session drives, from `prompt`, `negative_prompt`, `t_index_list` and `controlnets`. Empty drives all four |

## Running it

Start the mock Control API in its own terminal and leave it running:

```
python scripts/mock-control-api.py --always-ready
```

Its other modes (`--not-ready 404`, `--not-ready 409`, `--async-create`,
`--unstable`, `--fail-rate 0.2`) imitate the ways the real API misbehaves.

In a second terminal, with the settings loaded, run the module on a clip
longer than a minute:

```
python -m seed42_audio.orchestrate data/dark_calm_2.mp3
```

It prints each window's `MusicState` and one line per HTTP call, and the mock
prints each update it receives. With no clip named it plays
`data/bright_energetic_3.mp3`, which at 59 seconds ends before the first
update.

Other entry points, all run from the repository folder:

| Command | What it does |
|---|---|
| `python compare_stages.py data/dark_calm_2.mp3` | One window through all three Stages, side by side, with timings |
| `python -m pytest` | The test suite |
| `python tests/harness.py` | The module against the mock in every failure mode. It starts its own mock, so stop yours first |
| `python scripts/time_startup.py` | Seconds from launch to the first prompt, for each Stage |
| `python scripts/evaluate_stages.py --stages 1,2` | Every evaluation clip through the chosen Stages, saved to `results/` |
| `python scripts/evaluation.py` | Scores and figures for the newest evaluation run |

## What is where

| Path | Contents |
|---|---|
| `src/seed42_audio/` | The module |
| `scripts/` | The mock, evaluation, timing and synthetic-audio scripts |
| `tests/` | The pytest checks and the mock harness |
| `data/` | Cleared test tracks, synthetic clips and the phrase bank. Each track's source and licence is in `data/corpus_manifest.csv` |
| `results/` | Evaluation runs, scores and figures |
| `docs/` | The seam contract and the design document |
| `.env.example` | The settings template |
| `AGENTS.md` | The rules for working in this repository |

## Working on it

Every change goes through a branch and a pull request with one approval, from
Jayden or Hatim. Run `python -m pytest` before opening one. `AGENTS.md` has the
rest, including the rule of no em dashes anywhere.
