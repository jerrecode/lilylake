import json
from pathlib import Path

import pytest

from lilylake.adapters import import_maestro, merge_manifests
from lilylake.music.midi import write_midi
from lilylake.music.schema import Note, Part, Piece


def test_merge_preserves_domain_composition_split_and_rejects_leakage(tmp_path):
    files = []
    for i in range(2):
        folder = tmp_path / str(i)
        folder.mkdir()
        row = {
            "id": str(i),
            "composition_id": "same-piece",
            "split": "train",
            "audio": "audio.wav",
            "events": "events.json",
        }
        path = folder / "manifest.jsonl"
        path.write_text(json.dumps(row))
        files.append(path)
    rows = merge_manifests(files, tmp_path / "combined.jsonl")
    assert len(rows) == 2 and all(Path(r["audio"]).is_absolute() for r in rows)
    row["split"] = "test"
    files[1].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="leakage"):
        merge_manifests(files, tmp_path / "bad.jsonl")


def test_maestro_local_adapter_records_license_and_safe_paths(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    write_midi(Piece([Part("p", "grand_piano", [Note(60, 0, 0.5)])]), root / "p.mid")
    (root / "p.wav").write_bytes(b"audio retained, not copied")
    metadata = [
        {
            "canonical_composer": "Original composer",
            "canonical_title": "Original piece",
            "split": "train",
            "audio_filename": "p.wav",
            "midi_filename": "p.mid",
        }
    ]
    (root / "maestro.json").write_text(json.dumps(metadata))
    rows = import_maestro(root, root / "maestro.json", tmp_path / "imported")
    assert rows[0]["license"] == "CC-BY-NC-SA-4.0" and Path(rows[0]["events"]).exists()
    metadata[0]["midi_filename"] = "../outside.mid"
    (root / "maestro.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="outside"):
        import_maestro(root, root / "maestro.json", tmp_path / "bad")


def test_merge_rejects_generator_seed_variants_across_splits(tmp_path):
    files = []
    for i, split in enumerate(["train", "test"]):
        path = tmp_path / f"{i}.jsonl"
        row = {
            "id": str(i),
            "composition_id": f"level-variant-{i}",
            "seed": 42,
            "generator_version": 1,
            "level": 5 + i,
            "split": split,
            "audio": "audio.wav",
            "events": "events.json",
        }
        path.write_text(json.dumps(row))
        files.append(path)
    with pytest.raises(ValueError, match="leakage"):
        merge_manifests(files, tmp_path / "bad.jsonl")


@pytest.mark.parametrize("other_split", ["validation", "test"])
def test_direct_manifest_load_rejects_seed_family_leakage(tmp_path, other_split):
    from lilylake.config import Config
    from lilylake.training import MusicDataset

    rows = [
        {
            "composition_id": str(i),
            "generator_version": 1,
            "seed": 42,
            "split": split,
            "audio": "a.wav",
            "events": "e.json",
        }
        for i, split in enumerate(["train", other_split])
    ]
    manifest = tmp_path / "direct.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        MusicDataset(manifest, Config())
