# Silence hold: a silent window keeps the last prompt and logs a warning.

from types import SimpleNamespace

from seed42_audio import orchestrate


def state(name, energy, timestamp):
    return SimpleNamespace(name=name, energy=energy, timestamp=timestamp)


# Run run_dev over the given windows with the writer and pipeline replaced,
# and return the params that would have been sent.
def run_with(monkeypatch, states):
    sent = []

    def fake_run(paths, on_emit=None, **kwargs):
        for s in states:
            on_emit(s)

    monkeypatch.setattr(orchestrate, "run", fake_run)
    monkeypatch.setattr(orchestrate.writer, "login", lambda: None)
    monkeypatch.setattr(orchestrate.writer, "create_stream", lambda: "stream")
    monkeypatch.setattr(orchestrate.writer, "send",
                        lambda stream_id, params: sent.append(params))
    monkeypatch.setattr(orchestrate.writer, "delete_stream", lambda stream_id: None)
    monkeypatch.setattr(orchestrate, "build_params",
                        lambda s, stage_module: {"prompt": s.name})
    orchestrate.run_dev(["unused.wav"], stage=1)
    return sent


def test_silent_window_keeps_last_prompt(monkeypatch, capsys):
    sent = run_with(monkeypatch, [
        state("first", 0.05, 30.0),
        state("quiet", 0.0, 60.0),
        state("third", 0.05, 90.0),
    ])
    assert sent == [{"prompt": "first"}, {"prompt": "third"}]
    assert "silent window at 60.0s" in capsys.readouterr().out


def test_silent_first_window_is_still_sent(monkeypatch):
    sent = run_with(monkeypatch, [state("quiet", 0.0, 30.0)])
    assert sent == [{"prompt": "quiet"}]