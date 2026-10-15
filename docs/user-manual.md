# seed42 Audio Module: User Manual

First draft, 15 October 2026. Written against the mock Control API, before
access to a live seed42 stream. Live microphone or line-in input is in
progress and will be covered here once it is merged.

For installing, testing and fixing problems, see `docs/runbook.md`.

## 1. What the module does

seed42 restyles a live camera feed with an image model and shows the result on
a projector. Its look is set by a text prompt and a handful of sliders, which
normally someone has to change by hand.

This module changes them for you, following the music. While a track plays,
it listens to the last 30 seconds of audio, works out how the music sounds
(its tempo, energy, brightness, key and mood), and turns that into a prompt
and four slider values. It sends them to seed42 once a minute, so the
visuals shift as the music does.

## 2. What you need

- The `seed42-clap` environment, set up as in the README.
- For Stage 3 only: Ollama, open, with `llama3.2` pulled.
- The audio, as a WAV or MP3 file. Live input is in progress.
- For a live stream: the seed42 account details and the live address. Until
  then, the mock Control API stands in for seed42 on your own computer.

## 3. Settings

Settings are environment variables. Copy `.env.example` to `.env`, fill it in,
and load it into each new terminal before a session:

| System | Load the settings | Check them |
|---|---|---|
| macOS or Linux | `set -a; source .env; set +a` | `env \| grep SEED42` |
| Windows PowerShell | one at a time, for example `$env:SEED42_STAGE = '2'` | `Get-ChildItem env:SEED42*` |

| Setting | What it does |
|---|---|
| `SEED42_BASE_URL` | Where seed42's Control API is. The mock unless set; `https://seed42.app` for the live service |
| `SEED42_EMAIL`, `SEED42_PASSWORD` | The account seed42 issues. Any non-empty values work with the mock |
| `SEED42_STAGE` | Which Stage writes the prompt, 1, 2 or 3 (section 4). Default 1 |
| `SEED42_FIELDS` | Which fields the module drives (section 5). Empty drives them all |

Keep real passwords in `.env` only. It is never committed.

## 4. Choosing a Stage

The Stage decides how the prompt is written. The sliders are the same for all
three.

| Stage | How it writes the prompt | Time per update | Good to know |
|---|---|---|---|
| 1 | Looks up phrases in a fixed table, by mood, major or minor, and energy | Instant | The most predictable, and the most repetitive |
| 2 | Matches the sound itself against a bank of written phrases with CLAP, an audio and text model | A fraction of a second | Picks one phrase each for subject, motion, lighting and atmosphere |
| 3 | Asks Llama 3.2, a language model running on your own computer | A few seconds | The most varied, and the least predictable |

Stages 2 and 3 also take longer to start, while their models load, and they
fall back to Stage 1 whenever they cannot answer, for example when CLAP is not
installed or Ollama is closed. The session carries on either way, and the
terminal says so.

## 5. Choosing what the module controls

By default the module drives four fields. You can hand any of them back to
yourself with `SEED42_FIELDS`. A field the module does not drive is never sent,
so seed42 keeps whatever value it has, including anything you set by hand.

| Field | seed42 control | What it changes |
|---|---|---|
| `prompt` | Prompt | What the visuals look like |
| `negative_prompt` | Negative | What the visuals avoid |
| `t_index_list` | Creativity | How far the image moves away from the camera. seed42 shows it as 19 minus the value sent |
| `controlnets` | Depth, Edge and Consistency | How closely the image keeps the camera's depth, edges and previous frame. These three always go together |

Some examples:

| You want to | Set |
|---|---|
| Let the module drive everything | `SEED42_FIELDS=` (or leave it out) |
| Write your own prompt and let the module move the sliders | `SEED42_FIELDS=t_index_list,controlnets` |
| Let the module write the prompts and set the sliders yourself | `SEED42_FIELDS=prompt,negative_prompt` |

A name that is not one of these four stops the module before it connects, and
the error lists the names it accepts.

## 6. Running a session

These steps use the mock. For a live stream, set `SEED42_BASE_URL` and the
account instead, and skip the first step.

1. In one terminal, from the repository folder, start the mock and leave it
   running:

   ```
   python scripts/mock-control-api.py --always-ready
   ```

2. In a second terminal, activate the environment and load your settings.
3. Start the module on a track longer than a minute:

   ```
   python -m seed42_audio.orchestrate data/dark_calm_2.mp3
   ```

4. Stop it at any time with **Ctrl+C**. It ends its seed42 session on the way
   out.

What you see in the module's terminal:

- `[orchestrate] this session drives: ...`, the fields from section 5.
- One `[output.writer]` line per call to seed42. A `status` of 200 or 201
  means it worked.
- Once a minute, a line of the music readings, then `sent 60.0s`. Without
  Essentia, each one also has a line saying the mood reading used its
  fallback estimate, which is expected.

The mock's terminal shows each update it receives, as a `patch #1 on ...`
line with the start of the prompt.

## 7. How quickly the visuals follow the music

Each update is based on the 30 seconds of audio before it, and updates go out
once a minute. The first one comes a minute into the track, and a change in
the music shows at the next update, so up to about a minute later. A track
shorter than a minute produces no update at all.

A file is read as fast as the computer allows, not in real time, so a
two-minute track finishes in seconds. With live input, the audio will arrive
in real time.

## 8. If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `KeyError: 'SEED42_EMAIL'` | The settings are not loaded in this terminal | Load them (section 3) and start again |
| `SEED42_FIELDS: ... not one of ...` | A field name is misspelt or not allowed | Use names from section 5 |
| `[stage2] CLAP unavailable` or `[stage3] unavailable` | That Stage cannot run, so Stage 1 wrote the prompt | See the runbook. The session is unaffected |
| `update dropped` | seed42 was slow or still starting, so that one update was skipped | Nothing. The next update goes out as normal |
| `ending session` | The seed42 stream has ended | Start a new session |
| A status of 401 | The account details are wrong | Check `.env` and start again |

The runbook explains each of these in more detail.

## 9. Limits of this version

- It reads audio files. Live input is in progress.
- The mood reading uses a simple estimate from brightness, major or minor,
  and loudness. It tells calm from energetic music far better than dark from
  bright.
- The update rate and the 30 second window are fixed in code, not settings.
- Depth, Edge and Consistency can only be handed back together, not one at a
  time.
- Whether seed42 will create the stream itself in production is still open
  with the sponsor. For now the module creates its own stream and ends it
  when it stops.
