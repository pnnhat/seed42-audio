"""Stage 2: CLAP retrieval for visual prompt generation.

The Stage 2 interface is:

    generate(state) -> {
        "prompt": "...",
        "negative_prompt": "..."
    }

The phrase bank is embedded once when this module starts.
The current audio window in state.samples is embedded once per cycle.

If CLAP is unavailable or the Stage 2 cycle exceeds the time budget,
Stage 1 is used as the fallback.
"""

import json
import time
from pathlib import Path

import numpy as np


# Project paths and timing
PHRASE_BANK_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "phrase_bank.json"
)

PIPELINE_SAMPLE_RATE = 22050
CLAP_SAMPLE_RATE = 48000

# Maximum time allowed for one Stage 2 cycle.
CYCLE_BUDGET_SECONDS = 5.0


# CLAP and phrase embeddings are loaded once at startup.
_CLAP = None
_PHRASE_BANK = {}
_PHRASE_EMBEDDINGS = {}
_CLAP_READY = False


def _normalise(embeddings):
    """Normalise embeddings for cosine similarity."""
    values = np.asarray(embeddings, dtype=np.float32)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    norms = np.linalg.norm(values, axis=1, keepdims=True)

    return values / np.maximum(norms, 1e-12)


def _load_phrase_bank():
    """Load the phrase bank from JSON."""
    with open(PHRASE_BANK_PATH, "r", encoding="utf-8") as file:
        phrase_bank = json.load(file)

    if not isinstance(phrase_bank, dict):
        raise ValueError("Phrase bank must contain phrase groups.")

    return phrase_bank


def _initialise():
    """Load CLAP and embed the phrase bank once at startup."""
    global _CLAP
    global _PHRASE_BANK
    global _PHRASE_EMBEDDINGS
    global _CLAP_READY

    try:
        import laion_clap

        # Load the phrase bank once.
        _PHRASE_BANK = _load_phrase_bank()

        # Load CLAP once.
        _CLAP = laion_clap.CLAP_Module(enable_fusion=False)
        _CLAP.load_ckpt()

        # Embed every phrase once.
        for group, phrases in _PHRASE_BANK.items():
            if not isinstance(phrases, list) or not phrases:
                continue

            embeddings = _CLAP.get_text_embedding(
                phrases,
                use_tensor=False,
            )

            _PHRASE_EMBEDDINGS[group] = _normalise(embeddings)

        _CLAP_READY = bool(_PHRASE_EMBEDDINGS)

        if _CLAP_READY:
            print("[stage2] CLAP ready")
        else:
            print("[stage2] No phrase embeddings available")

    except Exception as error:
        _CLAP = None
        _PHRASE_BANK = {}
        _PHRASE_EMBEDDINGS = {}
        _CLAP_READY = False

        print(f"[stage2] CLAP unavailable: {error}")


# Initialise once when Stage 2 is imported.
_initialise()


def _fallback(state):
    """Return the Stage 1 result unchanged."""
    from seed42_audio.stages.stage1 import generate as stage1_generate

    return stage1_generate(state)


def _prepare_audio(samples):
    """Convert the current audio window to mono 48 kHz audio."""
    import librosa

    audio = np.asarray(samples, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio window is empty.")

    # Convert stereo/multi-channel audio to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Convert the pipeline's 22050 Hz audio to CLAP's 48000 Hz rate.
    if PIPELINE_SAMPLE_RATE != CLAP_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=PIPELINE_SAMPLE_RATE,
            target_sr=CLAP_SAMPLE_RATE,
        )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio window is empty after resampling.")

    return audio


def _retrieve_phrases(audio_embedding):
    """Retrieve the closest phrase from each group.

    The closest phrase becomes part of the positive prompt.
    The least-similar phrase becomes part of the negative prompt.
    """
    audio_embedding = _normalise(audio_embedding)[0]

    positive = {}
    negative = {}

    for group, phrase_embeddings in _PHRASE_EMBEDDINGS.items():
        phrases = _PHRASE_BANK[group]

        if not phrases:
            continue

        similarities = phrase_embeddings @ audio_embedding

        closest_index = int(np.argmax(similarities))
        furthest_index = int(np.argmin(similarities))

        positive[group] = phrases[closest_index]
        negative[group] = phrases[furthest_index]

    return positive, negative


def _assemble_prompt(phrases):
    """Assemble retrieved phrases into one comma-separated prompt."""
    groups = ("subject", "motion", "lighting", "atmosphere")

    return ", ".join(
        phrases[group]
        for group in groups
        if group in phrases
    )


def generate(state) -> dict:
    """Generate prompt and negative prompt from the current audio window."""
    # Fall back immediately if CLAP was not available at startup.
    if not _CLAP_READY or _CLAP is None:
        return _fallback(state)

    # Stage 2 requires the waveform added to MusicState.
    samples = getattr(state, "samples", None)

    if samples is None:
        return _fallback(state)

    started = time.perf_counter()

    try:
        # Prepare the current audio window.
        audio = _prepare_audio(samples)

        # CLAP expects audio as a batch.
        audio_input = audio.reshape(1, -1)

        # Embed the current audio window once for this cycle.
        audio_embedding = _CLAP.get_audio_embedding_from_data(
            x=audio_input,
            use_tensor=False,
        )

        # Check the cycle budget after audio embedding.
        elapsed = time.perf_counter() - started

        if elapsed > CYCLE_BUDGET_SECONDS:
            print(
                f"[stage2] cycle budget exceeded "
                f"({elapsed:.2f}s); using Stage 1"
            )
            return _fallback(state)

        # Retrieve one closest and one least-similar phrase per group.
        positive, negative = _retrieve_phrases(audio_embedding)

        # Check the complete retrieval cycle.
        elapsed = time.perf_counter() - started

        if elapsed > CYCLE_BUDGET_SECONDS:
            print(
                f"[stage2] cycle budget exceeded "
                f"({elapsed:.2f}s); using Stage 1"
            )
            return _fallback(state)

        prompt = _assemble_prompt(positive)
        negative_prompt = _assemble_prompt(negative)

        # Retrieval must produce both outputs.
        if not prompt or not negative_prompt:
            return _fallback(state)

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }

    except Exception as error:
        print(f"[stage2] retrieval failed: {error}")
        return _fallback(state)
