"""Seeded musical curricula, explicit coverage and piece-before-augmentation splits."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from .music.instruments import INSTRUMENTS
from .music.schema import Note, Part, Piece, Tempo
from .rendering import render_piece

STRATEGIES = ["scale", "arpeggio", "progression", "counterpoint", "cluster", "repeated"]


def composition_id(seed, level, instruments=None, strategy=None):
    config = {
        "generator_version": 1,
        "seed": seed,
        "level": level,
        "instruments": instruments,
        "strategy": strategy,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]


def split_for(identity):
    bucket = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def compose(seed, level=6, instruments=None, strategy=None):
    if not 1 <= level <= 10:
        raise ValueError("Curriculum level must be 1..10")
    rng = np.random.default_rng(seed)
    instruments = list(
        instruments
        or (
            ["grand_piano"]
            if level < 5
            else ["grand_piano", "violin"]
            if level < 7
            else ["grand_piano", "violin", "cello", "flute", "drum_kit"]
        )
    )
    if any(i not in INSTRUMENTS for i in instruments):
        raise ValueError("Unknown instrument")
    strategy = strategy or STRATEGIES[seed % len(STRATEGIES)]
    if strategy not in STRATEGIES:
        raise ValueError("Unknown composition strategy")
    bpm = int(rng.choice([60, 90, 120, 150, 180]))
    seconds = 60 / bpm
    root = int(rng.integers(48, 73))
    scale = [0, 2, 4, 5, 7, 9, 11, 12]
    parts = []
    for j, name in enumerate(instruments):
        inst = INSTRUMENTS[name]
        notes = []
        if level == 1:
            pitch = 21 + seed % 88 if inst.keyboard else int(rng.integers(inst.low, inst.high + 1))
            notes = [Note(pitch, 0.1, 0.1 + seconds, int(rng.integers(35, 121)))]
        else:
            count = 8 if level < 4 else 12
            step = seconds * (0.5 if strategy not in ["progression", "cluster"] else 1)
            for i in range(count):
                start = i * step + (0 if j == 0 else 0.25 * seconds)
                if inst.percussion:
                    pitches = [int(rng.choice([36, 38, 42, 46, 49]))]
                elif strategy == "scale":
                    pitches = [root + scale[(i + j * 2) % 8]]
                elif strategy == "arpeggio":
                    pitches = [root + [0, 4, 7, 12][(i + j) % 4]]
                elif strategy == "progression":
                    chord = root + [0, 5, 7, 0][(i // 3) % 4]
                    pitches = (
                        [chord + x for x in [0, 4, 7]]
                        if level >= 3 and inst.keyboard
                        else [chord + 12 + j * 2]
                    )
                elif strategy == "cluster":
                    pitches = list(
                        rng.choice(
                            np.arange(max(inst.low, root - 12), min(inst.high, root + 24) + 1),
                            size=min(16, 2 + level) if inst.keyboard and level >= 3 else 1,
                            replace=False,
                        )
                    )
                elif strategy == "repeated":
                    pitches = [root + j * 7]
                else:
                    pitches = [root + scale[(count - 1 - i if j else i) % 8] + (12 if j else 0)]
                duration = step * rng.choice([0.5, 0.75, 1.0, 1.5] if level >= 3 else [0.75, 1.0])
                for pitch in pitches:
                    pitch = int(np.clip(pitch, inst.low, inst.high))
                    notes.append(
                        Note(
                            pitch,
                            start,
                            start + duration,
                            int(rng.integers(35, 121)),
                            articulation="staccato" if i % 7 == 0 and level >= 4 else None,
                        )
                    )
            # Boundary keys appear with musical context rather than disappearing from the distribution.
            if inst.keyboard and level >= 4:
                notes.extend(
                    [Note(21 + seed % 88, 0, seconds), Note(108 - seed % 88, seconds, 2 * seconds)]
                )
        parts.append(Part(f"part_{j}", name, notes))
    pid = composition_id(seed, level, instruments, strategy)
    return Piece(
        parts,
        tempo_map=[Tempo(0, bpm)],
        title=f"LilyLake {strategy} seed {seed}",
        metadata={
            "composition_id": pid,
            "seed": seed,
            "level": level,
            "strategy": strategy,
            "generator_version": 1,
        },
    )


def augment(audio, sample_rate, seed):
    rng = np.random.default_rng(seed)
    x = audio.astype(np.float32, copy=True)
    gain = float(rng.uniform(0.4, 1.4))
    noise = float(rng.uniform(0, 0.002))
    pole = float(rng.uniform(0, 0.6))
    x = lfilter([1 - pole], [1, -pole], x, axis=0).astype(np.float32)
    x *= gain
    x += rng.normal(0, noise, x.shape).astype(np.float32)
    delay = int(sample_rate * rng.uniform(0.01, 0.08))
    decay = float(rng.uniform(0.05, 0.3))
    if delay < len(x):
        x[delay:] += x[:-delay].copy() * decay
    np.clip(x, -0.98, 0.98, out=x)
    return x, {
        "seed": seed,
        "gain": gain,
        "noise_std": noise,
        "eq_pole": pole,
        "echo_delay_samples": delay,
        "echo_gain": decay,
        "timing_shift": 0,
    }


def generate_dataset(
    directory,
    count=16,
    seed=0,
    level=6,
    engine="procedural",
    sample_rate=16000,
    soundfont=None,
    randomize=False,
    instruments=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    coverage = Counter()
    for i in range(count):
        # Balance composition strategies before sampling augmentation domains.
        strategy = min(
            STRATEGIES,
            key=lambda s: (
                coverage["strategy:" + s],
                (STRATEGIES.index(s) - seed) % len(STRATEGIES),
            ),
        )
        p = compose(seed + i, level, instruments, strategy)
        pid = p.metadata["composition_id"]
        identity = hashlib.sha256(f"{pid}:{engine}:{randomize}:{sample_rate}".encode()).hexdigest()[
            :20
        ]
        folder = directory / identity
        render_piece(p, folder, engine, sample_rate, soundfont, seed + i)
        if randomize:
            x, sr = sf.read(folder / "audio.wav")
            x, augmentation = augment(x, sr, seed + i + 100000)
            sf.write(folder / "audio.wav", x, sr, subtype="PCM_16")
        else:
            augmentation = None
        row = {
            "id": identity,
            "composition_id": pid,
            "split": split_for(pid),
            "seed": seed + i,
            "level": level,
            "strategy": strategy,
            "engine": engine,
            "sample_rate": sample_rate,
            "audio": f"{identity}/audio.wav",
            "events": f"{identity}/events.json",
            "augmentation": augmentation,
            "license": "procedural-original",
            "generator_version": 1,
        }
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata.update(row)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(row)
        coverage["strategy:" + strategy] += 1
        for part in p.parts:
            coverage["instrument:" + part.instrument] += 1
            for n in part.notes:
                coverage[f"pitch:{part.instrument}:{n.pitch}"] += 1
        print(f"generated {i + 1}/{count}: {identity} {row['split']}", flush=True)
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest)
    )
    (directory / "coverage.json").write_text(json.dumps(dict(coverage), indent=2) + "\n")
    return manifest
