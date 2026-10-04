"""Actual LilyPond compilation and two explicit audio engines."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .lilypond import serialize
from .music.instruments import INSTRUMENTS
from .music.midi import read_midi
from .music.schema import Piece


def compile_score(source, output, timeout=90, require_midi=True):
    """Compile a trusted LilyPond source. LilyPond can execute Scheme code."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    binary = shutil.which("lilypond")
    if not binary:
        raise RuntimeError("LilyPond missing: install lilypond and put it on PATH")
    # Remove outputs from earlier invocations so missing output cannot masquerade as success.
    for suffix in [".midi", ".mid", ".pdf"]:
        Path(str(output) + suffix).unlink(missing_ok=True)
    command = [binary, "-dno-point-and-click", "-o", str(output), str(source)]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        Path(str(output) + ".compiler.log").write_text(str(e))
        raise RuntimeError("LilyPond compile timed out") from e
    log = result.stdout + result.stderr
    Path(str(output) + ".compiler.log").write_text(log)
    if result.returncode:
        raise RuntimeError(f"LilyPond failed ({result.returncode}): {log[-4000:]}")
    midi = Path(str(output) + ".midi")
    if not midi.exists():
        midi = Path(str(output) + ".mid")
    if require_midi and not midi.exists():
        raise RuntimeError("LilyPond produced no MIDI: score needs a \\midi block")
    return {
        "midi": midi,
        "pdf": Path(str(output) + ".pdf"),
        "log": Path(str(output) + ".compiler.log"),
    }


def synthesize(piece: Piece, sample_rate=16000, seed=0, tuning_cents=0.0, tail=0.5):
    """Seeded additive educational synth; not a physically accurate instrument model."""
    if not 8000 <= sample_rate <= 192000:
        raise ValueError("Sample rate must be 8000..192000")
    rng = np.random.default_rng(seed)
    out = np.zeros((int((piece.duration + tail) * sample_rate) + 1, 2), dtype=np.float32)
    for pi, part in enumerate(piece.parts):
        inst = INSTRUMENTS[part.instrument]
        pan = 0.5 if len(piece.parts) == 1 else 0.2 + 0.6 * pi / max(1, len(piece.parts) - 1)
        weights = np.array([np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)], dtype=np.float32)
        for n in part.notes:
            release = n.offset
            if inst.keyboard:
                sustain = [p for p in part.pedals if p.controller == 64 and p.time <= n.offset]
                if sustain and sustain[-1].value >= 0.5:
                    release = next(
                        (
                            p.time
                            for p in part.pedals
                            if p.controller == 64 and p.time > n.offset and p.value < 0.5
                        ),
                        piece.duration,
                    )
            start = int(round(n.onset * sample_rate))
            hold = max(1 / sample_rate, release - n.onset)
            count = min(len(out) - start, int((hold + tail) * sample_rate))
            t = np.arange(count, dtype=np.float32) / sample_rate
            freq = 440 * 2 ** ((n.pitch - 69 + tuning_cents / 100) / 12)
            signal = np.zeros(count, dtype=np.float32)
            bowed = part.instrument in ["violin", "viola", "cello", "double_bass"]
            if inst.percussion:
                noise = rng.standard_normal(count).astype(np.float32)
                if n.pitch in [35, 36]:
                    signal = np.sin(2 * np.pi * (55 * t + 8 * (1 - np.exp(-t * 30))))
                else:
                    signal = noise
                envelope = np.exp(-t * (20 if n.pitch in [35, 36, 38, 40] else 8))
            else:
                for h in range(1, 13):
                    if h * freq >= sample_rate / 2:
                        break
                    strength = 1 / h if bowed else 1 / h**2 if inst.keyboard else 1 / h**1.5
                    # Harmonic envelopes and timbre differ independently of note pitch.
                    phase = 2 * np.pi * h * freq * t
                    if bowed:
                        phase += 0.3 * h * np.sin(2 * np.pi * 5.5 * t)
                    signal += strength * np.sin(phase + rng.uniform(-0.03, 0.03))
                attack = 0.03 if bowed else 0.003
                envelope = np.minimum(t / attack, 1)
                if inst.keyboard:
                    envelope *= np.exp(-t * (1.5 + 0.4 * freq / 1000))
                envelope *= np.exp(-np.maximum(0, t - hold) / (0.08 if inst.keyboard else 0.05))
            signal *= envelope * (n.velocity / 127) * 0.22
            out[start : start + count] += signal[:, None] * weights
    # Fixed gain preserves relative velocity; only emergency normalization clips extreme clusters.
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def render_midi(
    midi, audio, engine="procedural", sample_rate=16000, soundfont=None, seed=0, tuning_cents=0.0
):
    audio = Path(audio)
    audio.parent.mkdir(parents=True, exist_ok=True)
    truth = read_midi(midi)
    metadata = {
        "engine": engine,
        "sample_rate": sample_rate,
        "seed": seed,
        "tuning_cents": tuning_cents,
    }
    if engine == "procedural":
        sf.write(
            audio, synthesize(truth, sample_rate, seed, tuning_cents), sample_rate, subtype="PCM_16"
        )
    elif engine == "fluidsynth":
        binary = shutil.which("fluidsynth")
        if not binary:
            raise RuntimeError("FluidSynth missing; install fluidsynth")
        if not soundfont or not Path(soundfont).is_file():
            raise ValueError("FluidSynth requires an existing --soundfont .sf2")
        metadata["soundfont_sha256"] = hashlib.sha256(Path(soundfont).read_bytes()).hexdigest()
        command = [
            binary,
            "-ni",
            "-R",
            "0",
            "-C",
            "0",
            "-g",
            "0.6",
            "-r",
            str(sample_rate),
            "-T",
            "wav",
            "-O",
            "s16",
            "-F",
            str(audio),
            str(soundfont),
            str(midi),
        ]
        r = subprocess.run(
            command, capture_output=True, text=True, timeout=max(90, int(truth.duration * 5))
        )
        audio.with_suffix(".synth.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"FluidSynth failed: {r.stderr}")
    else:
        raise ValueError(f"Unknown renderer: {engine}")
    x, sr = sf.read(audio)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError("Renderer produced invalid audio")
    metadata["audio_duration"] = len(x) / sr
    return truth, metadata


def render_piece(piece, directory, engine="procedural", sample_rate=16000, soundfont=None, seed=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    piece.save(directory / "composition.json")
    (directory / "score.ly").write_text(serialize(piece))
    result = compile_score(directory / "score.ly", directory / "score")
    truth, meta = render_midi(
        result["midi"], directory / "audio.wav", engine, sample_rate, soundfont, seed
    )
    truth.metadata.update({"label_source": "LilyPond compiled MIDI", "renderer": meta})
    truth.save(directory / "events.json")
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {**result, "audio": directory / "audio.wav", "events": directory / "events.json"}
