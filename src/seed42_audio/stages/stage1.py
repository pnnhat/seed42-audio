"""Stage 1: rule-based prompt dictionary.

Every Stage shares one signature, generate(state) -> dict with "prompt" and
"negative_prompt" keys (see the D3 seam). No API calls, no stored state: each
call looks only at the MusicState it is given.

The dictionary is keyed on three signals read straight off MusicState:
  - mood quadrant, from valence and arousal (Russell's circumplex, the same
    convention mood/classify.py already uses: negative valence is darker or
    sadder, negative arousal is calmer)
  - mode, major or minor, from chroma.py
  - energy band, low, mid or high, from the segment's RMS energy

Key (the pitch class letter) is available on MusicState but is not used here.
Twelve keys times two modes times four quadrants times three energy bands is
a lot of table for a rule-based baseline, and mode alone already carries the
light versus dark distinction key would add. Flagging this as an assumption
per AGENTS.md: if the seam contract from Friday's meeting wants key in the
mix too, this is the file to extend.

Kept as three small per-signal tables joined into one line, rather than one
nested table with an entry per quadrant/mode/band combination, so adding or
tuning a phrase touches one line instead of several near-duplicate ones.
"""

# --- Energy banding ----------------------------------------------------

# Same RMS range mood/classify.py normalises energy against, so a "high
# energy" reading means the same thing in both places.
_ENERGY_LOW = 0.0
_ENERGY_HIGH = 0.3


def _energy_band(energy):
    """Low, mid or high, from raw RMS energy. Mid if energy is not known yet."""
    if energy is None:
        return "mid"
    normalised = (max(_ENERGY_LOW, min(_ENERGY_HIGH, energy)) - _ENERGY_LOW) / (
        _ENERGY_HIGH - _ENERGY_LOW
    )
    if normalised < 1 / 3:
        return "low"
    if normalised < 2 / 3:
        return "mid"
    return "high"


# --- Mood quadrant -------------------------------------------------------


def _quadrant(valence, arousal):
    """One of four quadrants from valence and arousal, both in [-1, 1].
    Treated as neutral (0.0) when not known yet, rather than erroring.
    """
    valence = 0.0 if valence is None else valence
    arousal = 0.0 if arousal is None else arousal
    if arousal >= 0:
        return "bright_energetic" if valence >= 0 else "dark_energetic"
    return "bright_calm" if valence >= 0 else "dark_calm"


# --- The rule tables -------------------------------------------------------

# Each value is (prompt phrase, negative prompt phrase).
QUADRANT_PHRASES = {
    "bright_energetic": (
        "radiant colour bursts, rapid shifting patterns, sharp highlights",
        "muddy colours, dull static shapes, murky haze",
    ),
    "dark_energetic": (
        "jagged glitch textures, harsh strobing shadows, aggressive fractures",
        "soft pastel light, smooth gentle curves, calm stillness",
    ),
    "dark_calm": (
        "deep shadow, slow drifting fog, muted cold tones",
        "bright flashing light, sharp edges, chaotic motion",
    ),
    "bright_calm": (
        "soft warm glow, gentle flowing light, airy open space",
        "harsh dark shadows, jagged edges, violent motion",
    ),
}

# Mode only colours the prompt side, major and minor are not opposites of
# each other in a way that makes a clean negative phrase.
MODE_PHRASES = {
    "major": "warm golden light, uplifting colour palette",
    "minor": "cool blue and violet tones, melancholic undertone",
}

ENERGY_PHRASES = {
    "low": (
        "gentle slow movement, minimal motion, sparse detail",
        "frantic fast motion, dense chaotic detail",
    ),
    "mid": (
        "steady flowing movement, moderate detail",
        "",
    ),
    "high": (
        "rapid intense motion, dense layered detail, high contrast",
        "static stillness, flat empty space",
    ),
}


def generate(state) -> dict:
    """Prompt and negative prompt for one MusicState, from the rule tables above."""
    quadrant = _quadrant(state.valence, state.arousal)
    mode = state.mode if state.mode in MODE_PHRASES else "major"
    band = _energy_band(state.energy)

    quadrant_prompt, quadrant_negative = QUADRANT_PHRASES[quadrant]
    energy_prompt, energy_negative = ENERGY_PHRASES[band]

    prompt = ", ".join([quadrant_prompt, MODE_PHRASES[mode], energy_prompt])
    negative_prompt = ", ".join(p for p in (quadrant_negative, energy_negative) if p)

    return {"prompt": prompt, "negative_prompt": negative_prompt}
