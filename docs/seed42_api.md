# The seed42 Control API

A summary of the sponsor's Control API so the whole team builds to one contract.

Source: `docs/seed42-control-api.pdf`, seed42 integration preview, 20 August 2026.
This file covers what our module needs. Anything not here is in the sponsor PDF.

## What we send, and what we do not

seed42 owns the renderer. Video runs over a WebRTC connection that stays open for the
whole session, camera in and projector out. We never touch that.

Our module sends one REST call each time our classification changes, and seed42 patches
it onto the stream while it runs. The stream is never re-created for a parameter change.

The stream is created and the camera connected before our code runs. We are given a
`stream_id` and send against it for the length of the session.

## The request

One endpoint, `POST /api/prompts`, with `Content-Type: application/json`.

```json
{
  "stream_id": "<the running stream>",
  "body": {
    "prompt": "concrete hallway, hard shadows, deep red",
    "t_index_list": [20],
    "controlnets": [
      { "conditioning_scale": 0.6 },
      { "conditioning_scale": 0.4 },
      { "conditioning_scale": 0.8 }
    ]
  }
}
```

Two levels: `stream_id` sits beside `body`, not inside it.

Whatever is in `body` is forwarded to the engine as that stream's parameters. A partial
object merges with the existing configuration rather than replacing it, so fields we
leave out keep their current values. This is why sending only what changed is safe, and
it is what makes the rule below workable.

## The rule that matters most

A documented subset of fields updates on the running pipeline. Every other field
triggers a pipeline reload, documented at roughly thirty seconds, during which the
stream produces no output.

Never send a reload field on a live stream.

### Hot-swap fields, safe to send

| Field | Notes |
|---|---|
| `prompt` | String, or weighted tuples (see below) |
| `negative_prompt` | String |
| `t_index_list` | 1 to 4 integers, non-decreasing. We send one |
| `guidance_scale` | Number |
| `delta` | Number |
| `num_inference_steps` | Integer, 1 to 100. Also affects output frame rate |
| `seed` | Integer, or weighted tuples |
| `controlnets[].conditioning_scale` | 0 to 1. The scale only |
| `ip_adapter` | Scale, enabled, weight type, and the style image URL |

### Reload fields, never send

| Field | Notes |
|---|---|
| `model_id` | Fixed to `stabilityai/sdxl-turbo` at stream creation |
| `width`, `height` | 384 to 1024, divisible by 64. Set at stream creation |
| `controlnets[].model_id`, `controlnets[].preprocessor` | The three-ControlNet recipe is fixed for the session |
| `acceleration`, `lora`, `filters` | Engine setup, set at stream creation |

Because a partial update merges, omitting these leaves them unchanged. We do not need to
send them to keep them, and sending them is what causes the blackout.

## Our labels and the fields behind them

These are seed42's own control labels, which their interface already exposes. Worth
knowing when we decide which musical features drive which control.

| Our label | Field | What it is |
|---|---|---|
| Prompt | `prompt` | String or weighted tuples |
| Negative | `negative_prompt` | String |
| Creativity | `t_index_list` | Denoising start indices. Lower values diverge further from the camera, higher stay closer. seed42's interface shows this inverted, so the number a user sees is not the number sent |
| Depth | `controlnets[0].conditioning_scale` | Depth ControlNet, 0 to 1 |
| Edge | `controlnets[1].conditioning_scale` | Canny ControlNet, 0 to 1 |
| Consistency | `controlnets[2].conditioning_scale` | Tile ControlNet. Feeds the previous output back, which is what holds frames steady against each other |
| Image ref | `ip_adapter_style_image_url` | HTTPS URL or base64 data URI, 5MB maximum |
| Image ref strength | `ip_adapter.scale` | Only read when `ip_adapter.enabled` is true |
| Steps | `num_inference_steps` | 1 to 100 |
| Seed | `seed` | Integer or weighted tuples |

The three ControlNets are positional and fixed in the order depth, canny, tile for the
session. If we send the `controlnets` array we send all three entries in that order,
each containing only `conditioning_scale`. Entries containing only that key leave the
rest of each object unchanged.

## Weighted prompts and seeds

`prompt` takes either a plain string or an array of `[string, number]` tuples. `seed`
takes either an integer or an array of `[integer, number]` tuples. Both are hot in
either form.

```json
"prompt": [
  ["concrete hallway, hard shadows", 0.7],
  ["deep red haze, slow drift", 0.3]
]
```

This is worth knowing for the mapping stage: it lets one classification blend two
descriptions rather than switching between them. How the blending behaves is set at
stream creation and is not ours to change.

## Response codes

| Status | Meaning |
|---|---|
| 200 | Applied to the running stream |
| 400 | `stream_id` or `body` missing |
| 401 | No valid seed42 session on the request |
| 404 with "not ready" in the body | Stream exists but has not finished starting |
| other | Passed through from the engine unchanged |

Authentication is a seed42 session on the request. The engine key is held by seed42 and
never appears in anything we send.

## Timing behaviour

Documented in the sponsor spec and not yet implemented on our side, since we have no
live stream to send to. seed42 applies no rate limit to this endpoint.

- One request in flight per stream. A change arriving while a call is open replaces the
  pending payload rather than queueing behind it. Last write wins.
- Requests are aborted after 12 seconds.
- A 404 whose body contains "not ready" is retried up to 3 times, 2 seconds apart. No
  other status is retried.
- Parameters are sent only after the video connection is established. Calls before that
  return "not ready".
- A change applies to frames produced after it arrives, not to frames already in flight.

## What this means for our module

`output/writer.py` builds the request and refuses any field that is not on the hot list,
so a reload field cannot reach a live stream by accident. It does not decide what to
send. Turning a `MusicState` into a prompt and a set of parameter values is the mapping
stage, and it is not built yet.

Until we have a `stream_id`, `writer.py` prints the request it would send rather than
posting it.

## Open questions for the sponsor

1. How do we obtain a `stream_id`, and can we have a test stream to send against?
2. How does our module authenticate? A 401 means no valid seed42 session on the request,
   but the spec does not say how we establish one.
3. The field table lists `ip_adapter.*` as hot, while the mapping table gives
   `ip_adapter_style_image_url` as a flat field and `ip_adapter.scale` as nested. Which
   shape does the request expect?
