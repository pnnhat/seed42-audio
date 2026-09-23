# seed42 audio module: seam contract and architecture

The single internal source of truth for the shapes we build to and for how our
module talks to seed42. The sponsor's full external spec is `control-api.md`. Where
any older doc disagrees with this file, this file is right.

## What we send, and what we do not

seed42 owns the renderer. Video runs over a connection that stays open for the whole
session, camera in and projector out, and we never touch it. Our module sends one
REST call each time our classification changes, and seed42 patches it onto the
running stream. The stream is never re-created for a parameter change.

In production the stream is created and the camera connected before our code runs,
and we are given a stream id to send against for the session. We create our own
stream only while developing against the mock.

## The pipeline, end to end

Every window, the pipeline does one pass:

```
audio window -> features -> MusicState -> Stage (prompt text) + mapping (sliders)
             -> orchestrator merge -> writer -> PATCH /api/streams
```

The front half (window, features, MusicState) is Phase 1 and already runs. The
back half (Stage, mapping, merge, writer) is the D3 baseline, and it now runs end
to end against the mock.

## The shapes everyone builds to

Do not change these without telling Jayden, because other people's code depends on
them.

**MusicState** (`state/music_state.py`). One dataclass per window holding
timestamp, tempo, energy, brightness, flatness, key, mode, valence, arousal,
bass_energy, onset_times and beat_times. A `samples` field is being added in Part 2
so the Stages that need the waveform can reach it. The Stage step and the mapping
step both read this object.

**Every Stage** (`stages/stage1.py`, `stage2.py`, `stage3.py`) has one signature:

```
def generate(state) -> dict   # {"prompt": ..., "negative_prompt": ...}
```

It takes a MusicState and returns prompt text. No API calls, no stored state.
Stage 1, Stage 2 and Stage 3 share this shape so they swap behind the seam.

**The mapping** (`output/mapping.py`) has one signature:

```
def to_params(state) -> dict  # slider fields only
```

It returns Creativity as `t_index_list` with the 19 minus t_index inversion
applied, and Depth, Edge and Consistency as the three `controlnets`
`conditioning_scale` values in fixed order.

**The writer** (`output/writer.py`) is dumb transport:

```
login() -> token
create_stream(params=None) -> stream_id
send(stream_id, params) -> response
delete_stream(stream_id)
```

`send` applies the hot-field allow-list and PATCHes `/api/streams`. It never merges
and never generates prompt text.

## The merge lives in the orchestrator

The orchestrator (`orchestrate.py`) is the one place that assembles a whole
window's params. It calls the Stage for prompt text, the mapping for sliders,
merges the two dicts into one params object, and hands that to the writer. The
writer receives an already merged params object. The writer does not merge.

## The session and the request

```
login          POST   /api/auth      once, returns a bearer token
create_stream  POST   /api/streams   once, returns the stream id
send           PATCH  /api/streams   every window, body is { id, params }
delete         DELETE /api/streams   once
```

Cold fields are set at stream creation and only there. Each window sends only the
hot fields that changed. A partial update merges with the stream's existing
configuration rather than replacing it, so a field we leave out keeps its current
value. This is why sending only what changed is safe.

## Hot fields and cold fields

Hot fields update the running stream instantly and are safe every window:
`prompt`, `negative_prompt`, `seed`, `t_index_list`, `guidance_scale`, `delta`,
`num_inference_steps`, and the three `controlnets` `conditioning_scale` values.

Cold fields trigger an approximately 30 second reload with no output at all, so they
are set once at stream creation and never sent live.

| Cold field | Notes |
|---|---|
| `model_id` | fixed to `stabilityai/sdxl-turbo` at stream creation |
| `width`, `height` | 384 to 1024, divisible by 64, set at creation |
| `controlnets[].model_id`, `controlnets[].preprocessor` | the three-ControlNet recipe is fixed for the session |
| `acceleration`, LoRA, `filters` | engine setup, set at creation |

Because a partial update merges, omitting a cold field leaves it unchanged. We never
need to send one to keep it, and sending one is what causes the blackout.

The writer's allow-list is the last safety gate before the PATCH, and it checks
inside each `controlnets` entry as well as the top-level keys, so a cold subfield
such as `model_id` hidden in a controlnet entry is caught rather than passed
through. Never put anything but `conditioning_scale` in a controlnet entry.

## Our labels and the fields behind them

These are seed42's own control labels. This table is the reference the mapping
builds to.

| Our label | Field | What it is |
|---|---|---|
| Prompt | `prompt` | free text, where the Stage output lands |
| Negative | `negative_prompt` | free text |
| Creativity | `t_index_list` | denoising start indices. Lower values diverge further from the camera, higher stay closer. The UI shows `19 - t_index`, so the shown number is not the number sent. See the format note below |
| Depth | `controlnets[0].conditioning_scale` | depth ControlNet, index 0, 0 to 1 |
| Edge | `controlnets[1].conditioning_scale` | canny ControlNet, index 1, 0 to 1 |
| Consistency | `controlnets[2].conditioning_scale` | tile ControlNet, index 2, feeds the previous output back, which is what holds frames steady |
| Steps | `num_inference_steps` | 1 to 100, also affects the output frame rate |
| Seed | `seed` | integer |

The `controlnets` array is positional, so send all three scales in the fixed order
depth, edge, consistency even when only one changes, each entry containing only
`conditioning_scale`.

**`t_index_list` format, to reconcile before live.** The current sponsor guide
gives this as a comma-separated string, for example `"6"`, with 1 to 4
non-decreasing integers. Our `mapping.py` currently emits a list, for example
`[6]`. The mock accepts either because it does not check the type, so this passes in
development and would only surface against the real API. The mapping must emit the
string form before the live run.

## Response codes

| Status | Meaning |
|---|---|
| 200 | applied to the running stream |
| 400 | a required field such as the id is missing |
| 401 | no valid seed42 session on the request |
| 404 or 409 with not-ready in the body | the stream exists but has not finished starting |
| other | passed through from the engine unchanged |

Authentication is a bearer token from login, there is no API key. Build against
`scripts/mock-control-api.py`. The mock prints a warning on a cold field and the
real API stays silent, so a clean run against the mock is the check that matters.

## Request discipline

The client must survive the API misbehaving. One request in flight per stream with
last-write-wins, so a value computed while a call is open replaces the pending
payload rather than queueing. Abort a call after 12 seconds. Retry only not-ready,
treating the 404 and 409 forms the same, up to 3 attempts 2 seconds apart. Retry
nothing else, a 400 fails forever and a 502 means the stream is gone. This is proven
against the mock modes `--not-ready 404`, `--not-ready 409`, `--async-create` and
`--unstable`.

## Available but not wired yet

These exist in the API and may be useful later. Nothing sends them today, and the
allow-list does not include them, so they are out of scope until someone builds for
them deliberately.

Weighted prompts and seeds. `prompt` can take an array of `[string, number]` tuples
instead of a plain string, which lets one window blend two descriptions rather than
switching between them. `seed` can take the same tuple form.

Image reference (`ip_adapter`). An earlier spec listed `ip_adapter` fields as hot,
for supplying a style image. The current sponsor guide does not list them at all, so
treat `ip_adapter` as unconfirmed and do not send it until the sponsor confirms its
status and shape.