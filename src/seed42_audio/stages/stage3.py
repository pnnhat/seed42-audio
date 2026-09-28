# Stage 3: local LLM prompt generation via Ollama.
#
# Same seam as the other stages: generate(state) -> {"prompt", "negative_prompt"}.
# It asks a local Ollama model to describe the visuals the music suggests and
# return them as JSON. A guardrail checks the reply is usable before it leaves
# this module. Anything malformed, slow, or unreachable falls back to Stage 1,
# so Stage 3 can never put bad output on the stream.

import json
import re
import time

import requests


# Ollama's local HTTP API. Nothing here reaches the internet.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # any locally pulled model, e.g. llama3.2 or mistral

# Budgets and guardrail limits.
CYCLE_BUDGET_SECONDS = 8.0  # give up and fall back if the model is slow
MAX_PROMPT_CHARS = 300  # longer replies are treated as junk
MIN_PROMPT_CHARS = 8

# Phrases that mean the model hedged or refused instead of answering.
_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "as an ai",
    "i am unable",
)


def _fallback(state):
    # Reuse the rule-based baseline whenever Stage 3 cannot produce a clean reply.
    from seed42_audio.stages.stage1 import generate as stage1_generate

    return stage1_generate(state)


def _describe(state):
    # Turn the MusicState numbers into a short brief for the model.
    return (
        "tempo %.0f BPM, energy %.2f, brightness %.0f Hz, key %s %s, "
        "valence %.2f, arousal %.2f"
        % (
            getattr(state, "tempo", 0.0),
            getattr(state, "energy", 0.0),
            getattr(state, "brightness", 0.0),
            getattr(state, "key", "") or "unknown",
            getattr(state, "mode", "") or "",
            getattr(state, "valence", 0.0),
            getattr(state, "arousal", 0.0),
        )
    )


def _build_request(state):
    # One instruction, asking for strict JSON so the reply is easy to validate.
    brief = _describe(state)
    instruction = (
        "You turn a piece of music into a short visual prompt for an abstract "
        "audio-reactive video generator. The music right now is: "
        + brief
        + ". Reply with only a JSON object with two keys, prompt and "
        "negative_prompt. Each is a short comma-separated list of visual "
        "descriptions, no more than 20 words. Do not explain."
    )
    return {
        "model": OLLAMA_MODEL,
        "prompt": instruction,
        "stream": False,
        "format": "json",  # ask Ollama to constrain the output to JSON
        "options": {"temperature": 0.7},
    }


def _extract_json(text):
    # The model should return pure JSON, but pull the first {...} block out in
    # case it wrapped the answer in prose or code fences.
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def _clean(value):
    # A field is usable only if it is a non-empty, sensible-length string that
    # is not a refusal. Returns the cleaned string, or None if it fails.
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if len(text) < MIN_PROMPT_CHARS or len(text) > MAX_PROMPT_CHARS:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return None
    return text


def _guardrail(reply):
    # Accept only a dict carrying two clean string fields. Anything else is junk
    # and the caller falls back to Stage 1.
    if not isinstance(reply, dict):
        return None
    prompt = _clean(reply.get("prompt"))
    negative = _clean(reply.get("negative_prompt"))
    if prompt is None or negative is None:
        return None
    return {"prompt": prompt, "negative_prompt": negative}


def generate(state) -> dict:
    started = time.perf_counter()
    try:
        # Ask the local model.
        response = requests.post(
            OLLAMA_URL,
            json=_build_request(state),
            timeout=CYCLE_BUDGET_SECONDS,
        )
        response.raise_for_status()

        # Ollama returns the model's text under "response".
        text = response.json().get("response", "")

        # Give up if the call blew the cycle budget.
        if time.perf_counter() - started > CYCLE_BUDGET_SECONDS:
            print("[stage3] cycle budget exceeded; using Stage 1")
            return _fallback(state)

        # Parse the reply and run it through the guardrail.
        checked = _guardrail(_extract_json(text))
        if checked is None:
            print("[stage3] reply failed the guardrail; using Stage 1")
            return _fallback(state)

        return checked

    except Exception as error:
        # Ollama not running, model not pulled, network error, bad JSON: any of
        # these means Stage 3 cannot answer, so fall back rather than crash.
        print("[stage3] unavailable (%s); using Stage 1" % error)
        return _fallback(state)
