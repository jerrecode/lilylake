"""Explicitly licensed local dataset imports; never download copyrighted audio implicitly."""

import hashlib
import json
from pathlib import Path

from .music.midi import read_midi


def merge_manifests(manifests, output):
    rows = []
    splits = {}
    seen = set()
    for manifest in manifests:
        manifest = Path(manifest)
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["composition_id"]
            split = row["split"]
            if pid in splits and splits[pid] != split:
                raise ValueError(f"Composition leakage: {pid} appears in {splits[pid]} and {split}")
            splits[pid] = split
            for key in ["audio", "events"]:
                row[key] = str((manifest.parent / row[key]).resolve())
            identity = hashlib.sha256((row["audio"] + row["events"]).encode()).hexdigest()[:20]
            if identity in seen:
                continue
            seen.add(identity)
            row["id"] = identity
            rows.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def import_maestro(root, metadata, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(metadata).read_text())
    if isinstance(data, dict):
        keys = list(next(iter(data.values())).keys())
        data = [{column: values[k] for column, values in data.items()} for k in keys]
    rows = []
    splits = {}
    for record in data:
        audio = (root / record["audio_filename"]).resolve()
        midi = (root / record["midi_filename"]).resolve()
        if not audio.is_relative_to(root) or not midi.is_relative_to(root):
            raise ValueError("Dataset path points outside supplied root")
        if not audio.is_file() or not midi.is_file():
            raise FileNotFoundError(f"Missing aligned pair: {audio}, {midi}")
        canonical = (
            (record["canonical_composer"] + "|" + record["canonical_title"]).strip().casefold()
        )
        pid = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        split = record["split"]
        if split not in ["train", "validation", "test"]:
            raise ValueError("Unknown MAESTRO split")
        if pid in splits and splits[pid] != split:
            raise ValueError(f"Composition leakage in source metadata: {canonical}")
        splits[pid] = split
        identity = hashlib.sha256(record["midi_filename"].encode()).hexdigest()[:20]
        piece = read_midi(midi)
        piece.metadata.update(
            {
                "dataset": "maestro-v3",
                "license": "CC-BY-NC-SA-4.0",
                "source": "https://magenta.withgoogle.com/datasets/maestro",
                "original_metadata": record,
                "label_semantics": "aligned MIDI key release, including pedal controls",
            }
        )
        events = output / "events" / f"{identity}.json"
        piece.save(events)
        rows.append(
            {
                "id": identity,
                "composition_id": pid,
                "split": split,
                "audio": str(audio),
                "events": str(events),
                "dataset": "maestro-v3",
                "license": "CC-BY-NC-SA-4.0",
                "source": "https://magenta.withgoogle.com/datasets/maestro",
                "commercial_use": False,
                "midi_sha256": hashlib.sha256(midi.read_bytes()).hexdigest(),
            }
        )
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows
