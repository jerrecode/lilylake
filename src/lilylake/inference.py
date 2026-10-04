"""Audio-only neural inference and instrument-independent temporal event decoding."""

import json
from pathlib import Path

import numpy as np
import torch

from .audio import features, load_audio
from .config import Config
from .lilypond import serialize
from .models import EventModel
from .music.instruments import INSTRUMENTS
from .music.schema import Note, Part, Piece, Tempo


def select_device(request="auto"):
    if request == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return request


def decode(prediction, config: Config, duration=None):
    frame = prediction["frame"]
    onset = prediction["onset"]
    offset = prediction["offset"]
    velocity = prediction["velocity"]
    frames = frame.shape[0]
    parts = []
    dt = config.frame_seconds
    for i, name in enumerate(config.instruments):
        inst = INSTRUMENTS[name]
        notes = []
        for pitch in range(inst.low, inst.high + 1):
            start = None
            release = 0

            def close(stop):
                if start is None:
                    return
                on = start * dt
                off = min(stop * dt, duration if duration is not None else frames * dt)
                if off - on < config.min_note_seconds:
                    return
                sl = slice(start, max(start + 1, stop))
                conf = {
                    "pitch": float(frame[sl, i, pitch].mean()),
                    "onset": float(onset[start, i, pitch]),
                    "offset": float(offset[min(stop, frames - 1), i, pitch]),
                    "instrument": float(frame[sl, i, pitch].mean()),
                    "articulation": 0.0,
                    "source_assignment": float(frame[sl, i, pitch].mean()),
                }
                vel = int(
                    np.clip(
                        round(
                            float(velocity[start : min(start + 3, frames), i, pitch].mean()) * 127
                        ),
                        1,
                        127,
                    )
                )
                notes.append(
                    Note(
                        pitch,
                        on,
                        off,
                        vel,
                        conf,
                        expression={
                            "bow_direction": None,
                            "pedal_key_release_distinction": "unknown",
                            "confidence_calibrated": False,
                        },
                    )
                )

            active_onset = onset[:, i, pitch] >= config.onset_threshold
            edges = np.diff(np.pad(active_onset.astype(np.int8), (1, 1)))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            peaks = {int(a + np.argmax(onset[a:b, i, pitch])) for a, b in zip(starts, stops)}
            for t in range(frames):
                # One strike per connected onset region; new separated regions permit re-strikes.
                if t in peaks:
                    if start is not None:
                        close(t)
                    start = t
                    release = 0
                if start is not None and t > start:
                    release = release + 1 if frame[t, i, pitch] < config.frame_threshold else 0
                    if (
                        offset[t, i, pitch] >= config.offset_threshold
                        and frame[t, i, pitch] < config.frame_threshold
                    ):
                        close(t)
                        start = None
                        release = 0
                    elif release >= config.release_frames:
                        close(t - release + 1)
                        start = None
                        release = 0
            if start is not None:
                close(frames)
        notes.sort(key=lambda n: (n.onset, n.pitch))
        parts.append(Part(name, name, notes))
    return Piece(
        parts,
        metadata={
            "timing_resolution_sec": dt,
            "expression_identifiability": "unknown unless supervised",
            "confidence_calibrated": False,
        },
    )


def predict_audio(model, audio, config, device="cpu"):
    """Chunk with receptive-field context; concatenate frames before event decoding."""
    model.eval()
    hop = config.hop
    chunk_frames = max(1, int(config.chunk_seconds / config.frame_seconds))
    total_frames = len(audio) // hop + 1
    context = (config.n_fft // 2) // hop + 2 + sum(2 ** (i % 4) for i in range(config.depth)) + 2
    outputs = {k: [] for k in ["onset", "offset", "frame", "velocity"]}
    with torch.inference_mode():
        for start in range(0, total_frames, chunk_frames):
            stop = min(total_frames, start + chunk_frames)
            left = max(0, start - context)
            right = min(total_frames, stop + context)
            segment = audio[left * hop : min(len(audio), right * hop)]
            pred = model(features(segment, config).unsqueeze(0).to(device))
            a = start - left
            b = a + stop - start
            for key in outputs:
                outputs[key].append(pred[key][0, a:b].sigmoid().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}


def transcribe_audio(audio_file, checkpoint, output, tempo=120.0, validate=True):
    from .rendering import compile_score
    from .training import load_checkpoint

    state = load_checkpoint(checkpoint)
    config = Config(**state["config"])
    torch.set_num_threads(config.threads)
    device = select_device(config.device)
    model = EventModel(config).to(device)
    model.load_state_dict(state["model"])
    audio = load_audio(audio_file, config.sample_rate)
    prediction = predict_audio(model, audio, config, device)
    piece = decode(prediction, config, len(audio) / config.sample_rate)
    piece.tempo_map = [Tempo(0, tempo)]
    piece.metadata.update(
        {
            "source_audio": Path(audio_file).name,
            "checkpoint": Path(checkpoint).name,
            "tempo_source": "user/default assumption",
            "meter_source": "4/4 assumption",
            "key_source": "C major placeholder, not inferred",
        }
    )
    output = Path(output)
    if output.suffix != ".ly":
        output = output / "score.ly"
    output.parent.mkdir(parents=True, exist_ok=True)
    piece.save(output.parent / "events.json")
    output.write_text(serialize(piece))
    confidence = {
        "calibrated": False,
        "tempo": None,
        "meter": None,
        "key": None,
        "parts": {p.id: [n.confidence for n in p.notes] for p in piece.parts},
    }
    (output.parent / "confidence.json").write_text(json.dumps(confidence, indent=2) + "\n")
    if validate:
        compile_score(output, output.with_suffix(""))
    return piece, prediction
