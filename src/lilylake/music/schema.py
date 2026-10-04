"""Validated continuous-time schema. Expression remains unknown by default."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .instruments import INSTRUMENTS


@dataclass
class Note:
    pitch: int
    onset: float
    offset: float
    velocity: int = 80
    confidence: dict[str, float] = field(default_factory=dict)
    articulation: str | None = None
    expression: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.pitch, int) or not 0 <= self.pitch <= 127:
            raise ValueError("MIDI pitch must be an integer in 0..127")
        if (
            not all(math.isfinite(v) for v in [self.onset, self.offset])
            or not 0 <= self.onset < self.offset
        ):
            raise ValueError("Note times require finite 0 <= onset < offset")
        if not isinstance(self.velocity, int) or not 1 <= self.velocity <= 127:
            raise ValueError("Velocity must be 1..127")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.confidence.values()):
            raise ValueError("Confidence must be finite in 0..1")
        if self.articulation not in {None, "staccato", "tenuto", "accent", "marcato", "legato"}:
            raise ValueError("Unsupported articulation")


@dataclass
class Tempo:
    time: float
    bpm: float

    def __post_init__(self):
        if (
            not math.isfinite(self.time)
            or self.time < 0
            or not math.isfinite(self.bpm)
            or self.bpm <= 0
        ):
            raise ValueError("Invalid tempo")


@dataclass
class Meter:
    time: float = 0
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self):
        if (
            self.time < 0
            or not math.isfinite(self.time)
            or self.numerator <= 0
            or self.denominator not in [1, 2, 4, 8, 16, 32]
        ):
            raise ValueError("Invalid meter")


@dataclass
class Key:
    time: float = 0
    tonic: str = "c"
    mode: str = "major"

    def __post_init__(self):
        if (
            self.time < 0
            or not math.isfinite(self.time)
            or self.tonic
            not in [
                "c",
                "cis",
                "des",
                "d",
                "dis",
                "ees",
                "e",
                "f",
                "fis",
                "ges",
                "g",
                "gis",
                "aes",
                "a",
                "ais",
                "bes",
                "b",
            ]
            or self.mode not in ["major", "minor"]
        ):
            raise ValueError("Invalid key")


@dataclass
class Pedal:
    time: float
    value: float
    controller: int = 64

    def __post_init__(self):
        if (
            not math.isfinite(self.time)
            or self.time < 0
            or not math.isfinite(self.value)
            or not 0 <= self.value <= 1
            or self.controller not in [64, 66, 67]
        ):
            raise ValueError("Invalid pedal event")


@dataclass
class Part:
    id: str
    instrument: str
    notes: list[Note] = field(default_factory=list)
    pedals: list[Pedal] = field(default_factory=list)

    def __post_init__(self):
        if not self.id or self.instrument not in INSTRUMENTS:
            raise ValueError(f"Unknown instrument or empty part ID: {self.instrument}")


@dataclass
class Piece:
    parts: list[Part] = field(default_factory=list)
    tempo_map: list[Tempo] = field(default_factory=lambda: [Tempo(0, 120)])
    meter_map: list[Meter] = field(default_factory=lambda: [Meter()])
    key_map: list[Key] = field(default_factory=lambda: [Key()])
    title: str = "LilyLake transcription"
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported schema version")
        if len({p.id for p in self.parts}) != len(self.parts):
            raise ValueError("Part IDs must be unique")
        for name in ["tempo_map", "meter_map", "key_map"]:
            entries = getattr(self, name)
            if (
                not entries
                or entries[0].time != 0
                or any(a.time >= b.time for a, b in zip(entries, entries[1:]))
            ):
                raise ValueError(f"{name} must start at zero and be strictly ordered")

    @property
    def duration(self):
        return max((n.offset for p in self.parts for n in p.notes), default=0.0)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["parts"] = [
            Part(
                p["id"],
                p["instrument"],
                [Note(**n) for n in p.get("notes", [])],
                [Pedal(**v) for v in p.get("pedals", [])],
            )
            for p in d.get("parts", [])
        ]
        for name, typ in [("tempo_map", Tempo), ("meter_map", Meter), ("key_map", Key)]:
            if name in d:
                d[name] = [typ(**x) for x in d[name]]
        return cls(**d)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
