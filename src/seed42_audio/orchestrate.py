# Orchestrator: wire one window's MusicState through a chosen Stage and the
# mapping, merge the two into one params dict, and PATCH it to the seed42 mock
# through the writer. The merge lives here on purpose, so the writer stays a
# dumb transport and each Stage stays a pure function of one window.

from .pipeline import run
from .output import mapping, writer


# Stage selection: 1 rule-based, 2 CLAP retrieval, 3 local LLM. Stages are
# imported lazily so a Stage 1 run never loads CLAP or reaches for Ollama.
# A window with energy below this counts as silence. Energy is the RMS level:
# 0.0 for pure silence, about 0.04 for the quietest real track in the corpus.
SILENCE_ENERGY = 0.005


def is_silent(state):
    return state.energy < SILENCE_ENERGY

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


def run_dev(paths, stage=1, **kwargs):
    # kwargs (sr, window, hop, emit_every) pass straight through to pipeline.run.
    stage_module = _load_stage(stage)
    writer.login()
    stream_id = writer.create_stream()
    last_sent = None

    def on_emit(state):
        nonlocal last_sent
        timestamp = getattr(state, "timestamp", 0.0)
        # Silence hold: keep the last prompt and warn instead of sending a new one.
        if is_silent(state) and last_sent is not None:
            print("[orchestrate] warning: silent window at %.1fs, keeping the last prompt" % timestamp)
            return
        params = build_params(state, stage_module)
        writer.send(stream_id, params)
        last_sent = params
        print("  sent %.1fs" % timestamp)

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

    # Stage from the SEED42_STAGE env var, default 1. Paths from the argv tail.
    stage = int(os.environ.get("SEED42_STAGE", "1"))
    run_dev(sys.argv[1:] or ["data/bright_energetic_3.mp3"], stage=stage)
