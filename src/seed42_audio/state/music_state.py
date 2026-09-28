# The shared per-window state object: one MusicState per emit.
import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional


@dataclass
class MusicState:
    timestamp: float
    samples: Optional[object] = None
    tempo: float = 0.0
    energy: float = 0.0
    brightness: float = 0.0
    flatness: float = 0.0
    key: str = ""
    mode: str = ""
    valence: float = 0.0
    arousal: float = 0.0
    bass_energy: float = 0.0
    onset_times: list = field(default_factory=list)
    beat_times: list = field(default_factory=list)

    def update(self, features):
        """Copy known feature keys onto this state, ignoring any key that is
        not a declared field so a stray key never crashes the pipeline."""
        known = {f.name for f in fields(self)}
        for key, value in (features or {}).items():
            if key in known:
                setattr(self, key, value)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        # Keep the emitted line compact. The two raw event lists are for
        # downstream consumers, the mapping and later stages, not the log.
        d = {
            k: v
            for k, v in self.to_dict().items()
            if k not in ("onset_times", "beat_times")
        }
        return json.dumps(d)
