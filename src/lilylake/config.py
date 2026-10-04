"""Versionable model, audio, training and event decoder settings."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .music.instruments import INSTRUMENTS


@dataclass
class Config:
    sample_rate: int = 16000
    n_fft: int = 4096
    hop: int = 160
    harmonics: int = 6
    width: int = 24
    depth: int = 4
    instruments: list[str] = field(default_factory=lambda: ["grand_piano", "violin"])
    bucket_batches: bool = True
    batch_size: int = 4
    learning_rate: float = 0.002
    scheduler_step: int = 100
    scheduler_gamma: float = 0.7
    epochs: int = 30
    seed: int = 42
    threads: int = 2
    workers: int = 0
    device: str = "auto"
    mixed_precision: bool = False
    accumulation: int = 1
    gradient_clip: float = 3.0
    cache_mb: int = 512
    onset_threshold: float = 0.5
    frame_threshold: float = 0.5
    offset_threshold: float = 0.5
    release_frames: int = 3
    min_note_seconds: float = 0.03
    chunk_seconds: float = 20.0
    validation_interval: int = 5
    curriculum_gate: float = 0.8

    def __post_init__(self):
        if (
            not 8000 <= self.sample_rate <= 192000
            or self.n_fft < 256
            or self.n_fft % 2
            or not 0 < self.hop <= self.n_fft
        ):
            raise ValueError("Invalid sample rate, FFT or hop")
        if (
            self.width < 4
            or self.width % 4
            or not 1 <= self.depth <= 12
            or not 1 <= self.harmonics <= 16
        ):
            raise ValueError("Invalid network width/depth/harmonics")
        if (
            not self.instruments
            or len(set(self.instruments)) != len(self.instruments)
            or any(x not in INSTRUMENTS for x in self.instruments)
        ):
            raise ValueError("Invalid instrument taxonomy")
        if (
            self.scheduler_step < 1
            or not 0 < self.scheduler_gamma <= 1
            or self.batch_size < 1
            or self.epochs < 1
            or self.learning_rate <= 0
            or self.accumulation < 1
            or self.threads < 1
            or self.workers < 0
            or self.cache_mb < 0
        ):
            raise ValueError("Invalid training settings")
        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValueError("Invalid device")
        if (
            any(
                not 0 < v < 1
                for v in [self.onset_threshold, self.frame_threshold, self.offset_threshold]
            )
            or self.release_frames < 1
            or self.min_note_seconds <= 0
            or self.chunk_seconds < 1
            or self.validation_interval < 1
        ):
            raise ValueError("Invalid decoder settings")

    @property
    def frame_seconds(self):
        return self.hop / self.sample_rate

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
