"""Failure-driven, independently seeded difficult examples; no regression score reuse."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import split_for
from .music.schema import Note, Part, Piece, Tempo
from .rendering import render_piece

MODES = ["unison", "violin_solo", "piano_dense", "crossing"]


def compose_hard(seed, mode="unison"):
    if mode not in MODES:
        raise ValueError("Unknown hard-example mode")
    rng = np.random.default_rng(seed)
    bpm = int(rng.choice([90, 120, 150]))
    beat = 60 / bpm
    # C4 is reserved from these unison compositions for the named regression case.
    pitches = [p for p in range(55, 85) if p != 60]
    piano = []
    violin = []
    for i in range(4):
        pitch = int(rng.choice(pitches))
        on = i * beat
        if mode == "unison":
            piano.append(Note(pitch, on, on + beat * 0.5, int(rng.integers(70, 121))))
            violin.append(Note(pitch, on, on + beat * 0.9, int(rng.integers(40, 91))))
        elif mode == "violin_solo":
            violin.append(Note(pitch, on, on + beat * 0.8, int(rng.integers(35, 121))))
        elif mode == "piano_dense":
            root = int(rng.integers(40, 70))
            for semitone in rng.choice(
                np.arange(0, 20), size=int(rng.integers(4, 10)), replace=False
            ):
                piano.append(
                    Note(root + int(semitone), on, on + beat * 0.75, int(rng.integers(35, 121)))
                )
        else:
            piano.append(Note(pitch, on, on + beat * 0.8, int(rng.integers(50, 111))))
            violin.append(
                Note(
                    int(rng.choice(pitches)),
                    on + beat * 0.25,
                    on + beat,
                    int(rng.integers(40, 101)),
                )
            )
    parts = []
    if piano:
        parts.append(Part("piano", "grand_piano", piano))
    if violin:
        parts.append(Part("violin", "violin", violin))
    pid = hashlib.sha256(f"hard-v1:{seed}:{mode}".encode()).hexdigest()[:20]
    return Piece(
        parts,
        tempo_map=[Tempo(0, bpm)],
        title=f"Hard {mode} seed {seed}",
        metadata={
            "composition_id": pid,
            "seed": seed,
            "mode": mode,
            "generator": "hard-v1",
            "reserved_unison_pitch": 60,
        },
    )


def generate_hard_dataset(directory, count=32, seed=6000, engine="procedural", soundfont=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        mode = MODES[i % len(MODES)]
        piece = compose_hard(seed + i, mode)
        identity = piece.metadata["composition_id"]
        folder = directory / identity
        render_piece(piece, folder, engine, soundfont=soundfont, seed=seed + i)
        row = {
            "id": identity,
            "composition_id": identity,
            "split": split_for(identity),
            "seed": seed + i,
            "generator": "hard-v1",
            "mode": mode,
            "engine": engine,
            "audio": f"{identity}/audio.wav",
            "events": f"{identity}/events.json",
            "license": "original-generated",
        }
        rows.append(row)
        print(f"hard example {i + 1}/{count}: {mode} {row['split']}", flush=True)
    (directory / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows
