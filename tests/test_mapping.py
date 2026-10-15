from types import SimpleNamespace

from seed42_audio.output.mapping import to_params


def make_state(arousal):
    return SimpleNamespace(bass_energy=0.5, beat_times=[], energy=0.5, arousal=arousal)


def test_calm_music_moves_t_index():
    # Before the fix every negative arousal gave the same t_index of 19
    calm = int(to_params(make_state(-1.0))["t_index_list"])
    mild = int(to_params(make_state(-0.5))["t_index_list"])
    assert calm == 19
    assert mild < calm


def test_t_index_covers_full_range():
    assert int(to_params(make_state(-1.0))["t_index_list"]) == 19
    assert int(to_params(make_state(0.0))["t_index_list"]) == 10
    assert int(to_params(make_state(1.0))["t_index_list"]) == 0