# Orchestrator: wire one window's MusicState through a chosen Stage and the
# mapping, merge the two into one params dict, and PATCH it to the seed42 mock
# through the writer. The merge lives here on purpose, so the writer stays a
# dumb transport and each Stage stays a pure function of one window.

from .pipeline import run
from .output import mapping, writer

# The fields the module drives, as build_params makes them. Sponsor feedback
# F1 asked for a per-session choice, so SEED42_FIELDS can name a subset. The
# three controlnets scales (Depth, Edge, Consistency) count as one field,
# because the array is positional and always goes as a whole.
FIELDS = ("prompt", "negative_prompt", "t_index_list", "controlnets")


# Stage selection: 1 rule-based, 2 CLAP retrieval, 3 local LLM. Stages are
# imported lazily so a Stage 1 run never loads CLAP or reaches for Ollama.
def _load_stage(stage):
    if stage == 1:
        from .stages import stage1

        return stage1
    if stage == 2:
        from .stages import stage2

        return stage2
    if stage == 3:
        from .stages import stage3

        return stage3
    raise ValueError("unknown stage %r, expected 1, 2 or 3" % (stage,))


def build_params(state, stage_module):
    # Merge the Stage's prompt text with the mapping's slider params. The two
    # dicts have disjoint keys, so a plain merge is enough, and every key is a
    # hot field so the writer's allow-list passes it through.
    prompt = stage_module.generate(state)
    sliders = mapping.to_params(state)
    return {**prompt, **sliders}


# The fields one session drives, from SEED42_FIELDS, a comma-separated list.
# Left empty, the session drives all of FIELDS. A name outside FIELDS raises
# before the session starts, so a typo or a cold field stops the run instead
# of being dropped without a word.
def parse_fields(setting):
    chosen = tuple(name.strip() for name in setting.split(",") if name.strip())
    unknown = [name for name in chosen if name not in FIELDS]
    if unknown:
        raise ValueError(
            "SEED42_FIELDS: %s not one of %s"
            % (", ".join(unknown), ", ".join(FIELDS))
        )
    return chosen or FIELDS


def run_dev(paths, stage=1, fields=FIELDS, **kwargs):
    # kwargs (sr, window, hop, emit_every) pass straight through to pipeline.run.
    # fields is the part of FIELDS this session drives (sponsor F1). A field
    # left out is never sent, so the stream keeps whatever value it has.
    print("[orchestrate] this session drives: %s" % ", ".join(fields))
    stage_module = _load_stage(stage)
    writer.login()
    stream_id = writer.create_stream()

    def on_emit(state):
        params = build_params(state, stage_module)
        writer.send(stream_id, {k: v for k, v in params.items() if k in fields})
        print("  sent %.1fs" % getattr(state, "timestamp", 0.0))

    try:
        run(paths, on_emit=on_emit, **kwargs)
    except writer.StreamGoneError as gone:
        # A 502 means the stream ended. In development we cannot be handed a
        # new one, so end the session cleanly rather than crash.
        print("[orchestrate] %s; ending session" % gone)
    finally:
        # Teardown is allowed to fail and must not mask the run's result.
        try:
            writer.delete_stream(stream_id)
        except Exception:
            pass


if __name__ == "__main__":
    import os
    import sys

    # Stage from the SEED42_STAGE env var, default 1, and the fields this
    # session drives from SEED42_FIELDS, default all. Paths from the argv tail.
    stage = int(os.environ.get("SEED42_STAGE", "1"))
    fields = parse_fields(os.environ.get("SEED42_FIELDS", ""))
    run_dev(
        sys.argv[1:] or ["data/bright_energetic_3.mp3"], stage=stage, fields=fields
    )
