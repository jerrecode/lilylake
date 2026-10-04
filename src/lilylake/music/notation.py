"""Derived rational-beat notation; continuous performance is never mutated."""

from dataclasses import dataclass, field
from fractions import Fraction

from .schema import Piece, Tempo


def seconds_to_beats(seconds, tempos):
    beats = 0.0
    for i, t in enumerate(tempos):
        end = tempos[i + 1].time if i + 1 < len(tempos) else seconds
        if seconds <= t.time:
            break
        beats += (min(seconds, end) - t.time) * t.bpm / 60
    return beats


def beats_to_seconds(beats, tempos):
    consumed = 0.0
    for i, t in enumerate(tempos):
        if i + 1 == len(tempos):
            return t.time + (beats - consumed) * 60 / t.bpm
        width = (tempos[i + 1].time - t.time) * t.bpm / 60
        if beats <= consumed + width:
            return t.time + (beats - consumed) * 60 / t.bpm
        consumed += width
    return 0.0


@dataclass
class ScoreNote:
    pitches: tuple[int, ...]
    onset: Fraction
    duration: Fraction
    velocity: int
    articulation: str | None = None
    quantization_error_sec: float = 0.0


@dataclass
class ScorePart:
    id: str
    instrument: str
    notes: list[ScoreNote] = field(default_factory=list)


@dataclass
class Score:
    performance: Piece
    parts: list[ScorePart]
    end: Fraction


def quantize(piece: Piece, subdivisions=(1, 2, 3, 4, 6, 8), bpm=None):
    """Nearest supported grid, with explicit deterministic simple-grid tie preference."""
    tempos = piece.tempo_map if bpm is None else [Tempo(0, bpm)]

    def snap(b):
        candidates = [Fraction(round(b * s), s) for s in subdivisions]
        return min(candidates, key=lambda c: (round(abs(float(c) - b), 10), c.denominator))

    parts = []
    end = Fraction(0)
    for p in piece.parts:
        notes = []
        for n in sorted(p.notes, key=lambda n: (n.onset, n.pitch, n.offset)):
            on = snap(seconds_to_beats(n.onset, tempos))
            off = snap(seconds_to_beats(n.offset, tempos))
            off = max(off, on + Fraction(1, max(subdivisions)))
            notes.append(
                ScoreNote(
                    (n.pitch,),
                    on,
                    off - on,
                    n.velocity,
                    n.articulation,
                    abs(beats_to_seconds(float(on), tempos) - n.onset),
                )
            )
            end = max(end, off)
        # Merge only exactly aligned events. Different offsets remain independent voices.
        groups = {}
        for n in notes:
            key = (n.onset, n.duration, n.articulation)
            if key in groups:
                old = groups[key]
                old.pitches = tuple(sorted(old.pitches + n.pitches))
                old.velocity = max(old.velocity, n.velocity)
            else:
                groups[key] = n
        parts.append(
            ScorePart(
                p.id, p.instrument, sorted(groups.values(), key=lambda n: (n.onset, n.pitches))
            )
        )
    return Score(piece, parts, end)


def assign_voices(notes):
    voices = []
    for n in sorted(notes, key=lambda x: (x.onset, x.pitches)):
        available = [i for i, v in enumerate(voices) if v[-1].onset + v[-1].duration <= n.onset]
        if available:
            i = min(
                available,
                key=lambda i: abs(
                    sum(voices[i][-1].pitches) / len(voices[i][-1].pitches)
                    - sum(n.pitches) / len(n.pitches)
                ),
            )
            voices[i].append(n)
        else:
            voices.append([n])
    return voices
