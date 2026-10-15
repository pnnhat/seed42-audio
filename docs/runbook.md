# seed42 Audio Module: Runbook

First draft, 15 October 2026. Procedures for installing, running, checking and
fixing the module. Written against the mock Control API. The live API section
is a plan until seed42 issues access, and live input will be added once it is
merged.

For what the module does and how to use it in a session, see
`docs/user-manual.md`.

## 1. Install

### The environment

All three Stages run in one conda environment, `seed42-clap`, with Python 3.11.
From the repository folder:

```
conda create -n seed42-clap python=3.11 -y
conda activate seed42-clap
pip install "numpy==1.26.4" torch torchaudio torchvision laion-clap
pip install librosa soundfile scipy pyyaml requests matplotlib pytest
pip install -e .
```

On Windows with an NVIDIA GPU, install PyTorch's CUDA build first, using the
selector on pytorch.org, then the rest. Leave Essentia out on every system;
the mood reading uses its fallback estimate without it.

### Stage 3

Install Ollama from ollama.com and open it, then:

```
ollama pull llama3.2
```

### Check the install

Run these in order, in the environment, from the repository folder.

| Check | Command | Expected |
|---|---|---|
| CLAP loads | `python -c "from seed42_audio.stages import stage2"` | `[stage2] CLAP ready`. The first run downloads CLAP's checkpoint and the roberta-base text model, which takes several minutes |
| All three Stages answer | `python compare_stages.py data/dark_calm_2.mp3` | Three different prompts. Stage 2 reads as four phrases. Stage 3 needs Ollama open |
| Ollama has the model | `ollama list` | A line for `llama3.2` |
| Tests pass | `python -m pytest` | All passed |

### Install problems

| Error | Cause | Fix |
|---|---|---|
| `[stage2] CLAP unavailable: No module named 'torchvision'` | laion-clap needs torchvision but does not list it | `pip install torchvision` |
| An error mentioning numpy 2 | Another package upgraded numpy | `pip install "numpy==1.26.4"` |
| `No module named seed42_audio` | The package is not installed in this environment | `pip install -e .` from the repository folder |
| `conda` not recognised on Windows | conda is not set up for this shell | Use Anaconda Prompt, or run `conda init powershell` and reopen PowerShell |

Older setup notes also list a `weights_only` error and an unexpected
`position_ids` key. laion-clap 1.1.7 handles both, so neither should appear.

## 2. Before a session

1. Activate the environment: `conda activate seed42-clap`.
2. Load the settings from `.env` and check them, as in section 3 of the user
   manual.
3. For Stage 3, check Ollama is running: `ollama ps` should answer with a
   header row rather than a connection error.
4. For development, start the mock in its own terminal:
   `python scripts/mock-control-api.py --always-ready`.

## 3. Start, watch and stop

Start a session on a clip longer than a minute:

```
python -m seed42_audio.orchestrate data/dark_calm_2.mp3
```

It prints the fields it drives, then logs in, creates a stream and sends one
update per 60 seconds of audio. Stop it with **Ctrl+C**; the stream is deleted
on the way out.

For a quicker check, send an update every 10 seconds of audio instead. Updates
start once the first 30 seconds have played, and the first few come a second
apart while the schedule catches up:

```
python -c "from seed42_audio.orchestrate import run_dev; run_dev(['data/dark_calm_2.mp3'], emit_every=10)"
```

### Reading the log

Each call to seed42 prints one line:

```
[output.writer] {"op": "patch", "attempt": 1, "status": 200, "error": null, "t": 2497.211, "ms": 3}
```

| Field | Meaning |
|---|---|
| `op` | `login`, `create`, `poll` (waiting for a new stream), `patch` (an update) or `delete` |
| `attempt` | Which try this was, for a retried patch or a poll |
| `status` | The HTTP status, or `null` if no answer came |
| `error` | The error type when no answer came, such as `ReadTimeout` |
| `t` | When the call started, in seconds on a clock that only counts up. Only the gap between two lines means anything |
| `ms` | How long the call took |

Other lines worth knowing:

| Line | Meaning |
|---|---|
| `[orchestrate] this session drives: ...` | The fields chosen with `SEED42_FIELDS` |
| `[mood.classify] essentia path failed, using fallback` | Expected without Essentia, once per update |
| `[stage2] CLAP ready` | Stage 2 loaded CLAP |
| `[stage2] CLAP unavailable: ...` | Stage 2 could not load, so Stage 1 writes every prompt. The rest of the line says why |
| `[stage2] cycle budget exceeded (...); using Stage 1` | Stage 2 took over 5 seconds for one update |
| `[stage3] unavailable (...); using Stage 1` | Ollama could not be reached, or the model is missing |
| `[stage3] reply failed the guardrail; using Stage 1` | Llama's reply was empty, malformed, too long or a refusal |
| `[stage3] cycle budget exceeded; using Stage 1` | Llama took over 8 seconds |

## 4. When the API misbehaves

The writer follows the request discipline in `docs/seam-contract.md`. This is
what it does in each case, and what the log shows.

| Situation | What the writer does | What you see |
|---|---|---|
| The stream is still starting (409, or 404 saying not ready) | Retries, up to 3 attempts 2 seconds apart, then skips this update | `patch` lines with 409 or 404, then `stream not ready after 3 attempts, update dropped` if it never got ready |
| No answer within 12 seconds | Skips this update, without retrying or sending it late | A `patch` line with `ReadTimeout`, then `patch timed out after 12 seconds, update dropped` |
| The stream has ended (502) | Ends the session cleanly | `[orchestrate] stream ... has ended, a new one is needed; ending session` |
| A new stream answers 202 | Polls every 2 seconds until it is ready, for up to 30 seconds | `poll` lines, then the first `patch` |
| The delete fails | Ignores it, since seed42 closes abandoned sessions | A `delete` line with an error status |
| 400, 401 or any other error | Does not retry, and the session stops | The status in the log, then a traceback. A 401 usually means the account details are wrong |
| A field that would reload the stream | Refuses to send it | `ColdFieldError`. The shipped code never sends one, so this means a code change added one |
| The API cannot be reached at all | The session stops | A `ConnectionError` traceback. Handling this is planned |

## 5. Test against the mock

The mock imitates the real API on your computer, including its failures:

| Flag | What the mock does |
|---|---|
| `--always-ready` | Every stream is ready at once |
| `--not-ready 404` or `--not-ready 409` | A new stream answers not ready for 4 seconds, with that code |
| `--ready-after N` | Changes those 4 seconds to N |
| `--async-create` | Answers 202 on create, so the client must poll |
| `--unstable` | Streams end about 45 seconds after creation, and deletes fail |
| `--fail-rate 0.2` | Answers 502 to that share of creates and updates at random |

The harness runs the module against the mock in each of these modes and checks
the results:

```
python tests/harness.py
python tests/harness.py always-ready unstable
```

It starts its own mock on port 8787, so stop any mock you started first. Logs
go to `tests/logs/`. The `fail-rate` mode is informational and can fail at
random when the injected 502 lands on the create.

Run `python -m pytest` before every pull request.

## 6. Going live

To be confirmed once seed42 issues access.

1. In `.env`, set `SEED42_BASE_URL=https://seed42.app` and the account seed42
   issued, then load it.
2. Start with Stage 1 and a clip longer than a minute. In the log, check that
   `login` answers 200, `create` 201 or 202, and each `patch` 200.
3. Watch the stream in seed42 and confirm the prompt and sliders change at
   each update.
4. Check that the `delete` at the end answers 200. The writer sends the
   stream id as `stream_id`, which the sponsor's mock accepts, but the key the
   live API expects is not confirmed yet.
5. Run the load-time check (section 7).

Whether seed42 creates the stream in production and hands the module its id
is still open with the sponsor, and the module has no setting for a given
stream id yet. Note too that the live API never rejects a field that reloads
the stream, so the writer's check is the only guard, and it must stay in
place.

## 7. Measure

**Start-up.** Seconds from launch to the first prompt, for each Stage, three
fresh launches each:

```
python scripts/time_startup.py
```

Load Stage 2 once beforehand, so CLAP's downloads are not timed. The script
unloads Llama before every Stage 3 run, so each starts cold, and saves its
results to `results/startup_timing_<date>_<time>.txt` with the device each
model ran on. In a live session, the first update also waits for the first
60 second cycle of audio.

**Live load time.** With a live stream, the `t` values in the writer's log give
the time from login to a ready stream, and from there to the first update
that answers 200.

**Evaluation.** In the environment, with Ollama open for Stage 3:

```
python scripts/evaluate_stages.py --stages 1,2
python scripts/evaluation.py
python scripts/stage2_phrase_counts.py results/<the new evaluation CSV>
```

The first writes `results/evaluation_<date>_<time>.csv`. The second scores the
newest run and draws its figures. The third counts how often Stage 2 used each
phrase.

## 8. Live input

In progress. This section will cover choosing the input device and running a
session from a microphone or line-in once the code is merged.

## 9. Known issues

- With no clip named, `python -m seed42_audio.orchestrate` plays
  `data/bright_energetic_3.mp3`, which at 59 seconds ends before the first
  update, so it sends nothing.
- The update rate (60 seconds) and window (30 seconds) are not settings. Pass
  `emit_every` or `window` to `run_dev` to change them.
- A missing setting or an unreachable API stops the session with a traceback
  rather than a plain message.
