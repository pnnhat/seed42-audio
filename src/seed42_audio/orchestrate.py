from .pipeline import run
from .stages import stage1
from .output import mapping, writer


def build_params(state):
    prompt = stage1.generate(state)
    sliders = mapping.to_params(state)
    return {**prompt, **sliders}


def run_dev(paths, **kwargs):
    writer.login()
    stream_id = writer.create_stream()

    def on_emit(state):
        writer.send(stream_id, build_params(state))
        print("  sent %.1fs" % getattr(state, "timestamp", 0.0))

    try:
        run(paths, on_emit=on_emit, **kwargs)
    finally:
        try:
            writer.delete_stream(stream_id)
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    run_dev(sys.argv[1:] or ["data/test.wav"])
