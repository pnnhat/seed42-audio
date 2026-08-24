"""The shared music-state object: one per window, filled by the feature modules."""

import json
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class MusicState:
    """One reading of the music at a point in time.

    Feature modules return a dict whose keys match these field names. Unknown
    keys are ignored, so a module can return only the fields it produces.
    """

    timestamp: float
    tempo: Optional[float] = None
    energy: Optional[float] = None
    brightness: Optional[float] = None
    flatness: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None

    def update(self, features):
        """Merge a feature dict into this state, keeping only known fields."""
        allowed = {f.name for f in fields(self)}
        for name, value in features.items():
            if name in allowed:
                setattr(self, name, value)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self))
