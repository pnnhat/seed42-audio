"""Client for the seed42 Control API: sends parameter updates to a running stream.

See docs/seed42_api.md. The one hard rule: only ever send hot-swap fields. Any
other field reloads the pipeline, which takes the stream off air for roughly
thirty seconds.
"""

import json

ENDPOINT = "/api/prompts"

# Fields that update on the running pipeline. Anything not in this set forces a
# reload, so build_body refuses it. ip_adapter_style_image_url is included
# alongside ip_adapter because the sponsor spec gives both shapes and we have
# not yet confirmed which the request expects. See docs/seed42_api.md.
HOT_FIELDS = frozenset({
    "prompt",
    "negative_prompt",
    "t_index_list",
    "guidance_scale",
    "delta",
    "num_inference_steps",
    "seed",
    "controlnets",
    "ip_adapter",
    "ip_adapter_style_image_url",
})


class ReloadFieldError(ValueError):
    """Raised when a field would trigger a pipeline reload."""


def controlnet_scales(depth, edge, consistency):
    """Return the controlnets list for a partial update.

    The three ControlNets are positional and fixed in the order depth, canny,
    tile, so all three scales are always sent together. Building the list by
    hand risks putting a scale on the wrong ControlNet.
    """
    return [
        {"conditioning_scale": depth},
        {"conditioning_scale": edge},
        {"conditioning_scale": consistency},
    ]


def build_body(**fields):
    """Return the body of a partial update, rejecting any field that is not hot.

    Fields left out keep their current values, because seed42 merges a partial
    object rather than replacing the configuration. So we send only what changed.
    """
    rejected = set(fields) - HOT_FIELDS
    if rejected:
        raise ReloadFieldError(
            "these fields would reload the pipeline and cannot be sent live: "
            + ", ".join(sorted(rejected))
        )
    return fields


def build_request(stream_id, **fields):
    """Return the full payload for POST /api/prompts."""
    return {"stream_id": stream_id, "body": build_body(**fields)}


def send(stream_id, **fields):
    """Print the request that would be sent, and return it.

    We have no live stream_id yet, so nothing is posted. Returning the request
    lets a caller check the shape.
    """
    request = build_request(stream_id, **fields)
    print("POST " + ENDPOINT)
    print(json.dumps(request, indent=2))
    return request