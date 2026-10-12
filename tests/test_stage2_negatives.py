# D-04: Stage 2's negative phrase is the written opposite of its positive phrase.
#
# CLAP is replaced by a stand-in so the test runs without the model. Each phrase
# gets its own one-hot text embedding, and each window's audio embedding is the
# sum of the four positive phrases recorded for it in the 30 September run. The
# real retrieval code therefore picks the same positives as that run, and the
# test checks which negatives it now pairs with them, on the same 100 windows.
#
#   python -m pytest tests/test_stage2_negatives.py

import csv
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation_20260930_0220.csv"
GROUPS = ("subject", "motion", "lighting", "atmosphere")


class _FakeClap:
    index = {}

    def __init__(self, **kwargs):
        pass

    def load_ckpt(self):
        pass

    def get_text_embedding(self, phrases, use_tensor=False):
        for phrase in phrases:
            self.index.setdefault(phrase, len(self.index))
        return np.stack([self.one_hot(p) for p in phrases])

    @classmethod
    def one_hot(cls, phrase):
        vector = np.zeros(64, dtype=np.float32)
        vector[cls.index[phrase]] = 1.0
        return vector


@pytest.fixture(scope="module")
def stage2():
    fake = types.ModuleType("laion_clap")
    fake.CLAP_Module = _FakeClap
    saved = sys.modules.get("laion_clap")
    sys.modules["laion_clap"] = fake
    sys.modules.pop("seed42_audio.stages.stage2", None)
    try:
        module = importlib.import_module("seed42_audio.stages.stage2")
        assert module._CLAP_READY
        yield module
    finally:
        sys.modules.pop("seed42_audio.stages.stage2", None)
        if saved is None:
            sys.modules.pop("laion_clap", None)
        else:
            sys.modules["laion_clap"] = saved


def test_every_phrase_has_a_written_opposite(stage2):
    for group, phrases in stage2._PHRASE_BANK.items():
        opposites = stage2._OPPOSITES[group]
        assert len(opposites) == len(phrases)
        for i, j in enumerate(opposites):
            assert j != i
            assert opposites[j] == i


def test_negatives_on_30_september_windows(stage2):
    with open(RESULTS, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file)
                if r["stage"] == "2" and r["status"] == "ok"]

    before = after = total = 0
    for row in rows:
        positive = dict(zip(GROUPS, row["prompt"].split(", ")))
        negative = dict(zip(GROUPS, row["negative_prompt"].split(", ")))
        audio = sum(_FakeClap.one_hot(positive[g]) for g in GROUPS)

        new_positive, new_negative = stage2._retrieve_phrases(audio)

        for group in GROUPS:
            phrases = stage2._PHRASE_BANK[group]
            chosen = phrases.index(positive[group])
            written_opposite = phrases[stage2._OPPOSITES[group][chosen]]
            assert new_positive[group] == positive[group]
            total += 1
            before += negative[group] == written_opposite
            after += new_negative[group] == written_opposite

    print(f"\nwritten opposite: before {before} / {total}, after {after} / {total}")
    assert total == 400
    assert before == 78
    assert after == 400
