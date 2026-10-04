"""Explicit acoustic challenge suites; generation is not a claim of recognition."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .data import augment, compose
from .music.schema import Note, Part, Pedal, Piece, Tempo
from .rendering import render_piece


def critical_cases():
    def piano(notes):
        return Piece([Part("piano", "grand_piano", notes)])

    yield "one_key", piano([Note(60, 0, 0.5)]), None
    yield (
        "all_88_keys",
        piano([Note(p, (p - 21) * 0.25, (p - 21) * 0.25 + 0.2) for p in range(21, 109)]),
        None,
    )
    yield (
        "adjacent_intervals",
        piano(
            [
                Note(p + j, (p - 21) * 0.25, (p - 21) * 0.25 + 0.2)
                for p in range(21, 108)
                for j in [0, 1]
            ]
        ),
        None,
    )
    yield (
        "chord_families",
        piano(
            [
                Note(48 + j, i * 0.75, i * 0.75 + 0.6)
                for i, steps in enumerate(
                    [[0, 4, 7], [0, 3, 7], [0, 3, 6], [0, 4, 8], [0, 4, 7, 11], [0, 3, 7, 10]]
                )
                for j in steps
            ]
        ),
        None,
    )
    yield "dense_chords", piano([Note(p, 0, 1) for p in range(48, 73)]), None
    yield "fast_repeated", piano([Note(60, i * 0.1, i * 0.1 + 0.075) for i in range(20)]), None
    pedal = piano([Note(60, 0, 0.3), Note(64, 0.4, 0.7), Note(67, 0.8, 1.1)])
    pedal.parts[0].pedals = [Pedal(0, 1), Pedal(1.5, 0)]
    yield "sustain_pedal", pedal, None
    yield "arpeggio", compose(4, 4, ["grand_piano"], "arpeggio"), None
    yield (
        "independent_hands",
        piano(
            [Note(36, i * 0.5, i * 0.5 + 1) for i in range(6)]
            + [Note(72 + i % 5, i * 0.25, i * 0.25 + 0.2) for i in range(12)]
        ),
        None,
    )
    yield "violin", Piece([Part("v", "violin", [Note(60, 0, 1), Note(67, 1, 2)])]), None
    # Procedural bowed synth includes vibrato; labels explicitly do not pretend to capture a physical bow.
    yield (
        "violin_vibrato",
        Piece(
            [
                Part(
                    "v",
                    "violin",
                    [Note(69, 0, 2, expression={"vibrato_source": "procedural fixed 5.5Hz"})],
                )
            ]
        ),
        None,
    )
    for name, instruments in [
        ("piano_violin", ["grand_piano", "violin"]),
        ("piano_cello", ["grand_piano", "cello"]),
        ("piano_drums", ["grand_piano", "drum_kit"]),
        ("guitar_flute", ["acoustic_guitar", "flute"]),
        ("ensemble", ["grand_piano", "violin", "cello", "flute", "trumpet"]),
    ]:
        yield name, compose(12, 6, instruments, "counterpoint"), None
    for name, vel in [("soft", 15), ("loud", 125)]:
        yield name, piano([Note(60, 0, 1, vel), Note(64, 0, 1, vel)]), None
    base = compose(13, 6, ["grand_piano", "violin"], "progression")
    yield "noise", base, "noise"
    yield "heavy_reverb", base, "reverb"
    changing = piano([Note(60 + i % 8, i * 0.3, i * 0.3 + 0.2) for i in range(12)])
    changing.tempo_map = [Tempo(0, 120), Tempo(1.5, 90)]
    yield "tempo_change", changing, None
    yield "tuplets", piano([Note(60 + i % 5, i / 6, (i + 1) / 6) for i in range(12)]), None
    yield "dense_overlap", piano([Note(48 + i % 16, i * 0.1, i * 0.1 + 1) for i in range(24)]), None
    yield (
        "same_pitch_sources",
        Piece(
            [Part("p", "grand_piano", [Note(60, 0, 1)]), Part("v", "violin", [Note(60, 0, 1.5)])]
        ),
        None,
    )


def generate_suite(directory, engine="procedural", soundfont=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, piece, augmentation) in enumerate(critical_cases()):
        folder = directory / name
        render_piece(piece, folder, engine, soundfont=soundfont, seed=i)
        if augmentation:
            audio, sr = sf.read(folder / "audio.wav")
            audio, meta = augment(audio, sr, 1000 + i)
            if augmentation == "noise":
                audio += np.random.default_rng(i).normal(0, 0.02, audio.shape)
            else:
                delay = int(0.17 * sr)
                for repeat in range(1, 7):
                    shift = delay * repeat
                    if shift < len(audio):
                        audio[shift:] += audio[:-shift].copy() * 0.4**repeat
            sf.write(folder / "audio.wav", np.clip(audio, -0.98, 0.98), sr)
        rows.append(
            {
                "id": name,
                "composition_id": f"critical:{name}",
                "split": "test",
                "audio": f"{name}/audio.wav",
                "events": f"{name}/events.json",
                "engine": engine,
                "license": "procedural-original",
                "challenge": name,
            }
        )
        print(f"suite {i + 1}: {name}", flush=True)
    (directory / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "limitations.json").write_text(
        json.dumps(
            {
                "glissando": "Not yet rendered/labeled as continuous pitch trajectory; omitted rather than simulated as discrete notes",
                "physical_bow_direction": "Not identifiable without supervision",
                "vibrato": "Procedural synth includes fixed modulation; no vibrato estimator yet",
            },
            indent=2,
        )
        + "\n"
    )
    return rows
