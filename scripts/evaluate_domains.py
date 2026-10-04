"""Evaluate included weights and optionally re-render held-out pieces with another font."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lilylake.config import Config
from lilylake.models import EventModel
from lilylake.music.midi import write_midi
from lilylake.music.schema import Piece
from lilylake.rendering import render_midi
from lilylake.training import MusicDataset, evaluate_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="examples/development.pt")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", default="outputs/domain-evaluation")
    parser.add_argument("--unseen-soundfont")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    state = load_checkpoint(checkpoint)
    config = Config(**state["config"])
    torch.set_num_threads(config.threads)
    model = EventModel(config)
    model.load_state_dict(state["model"])
    dataset = MusicDataset(args.manifest, config, args.split)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    report = {**evaluate_model(model, dataset, config), **metadata}
    (root / "heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.unseen_soundfont:
        font = Path(args.unseen_soundfont).resolve()
        rows = []
        domain = None
        for row in dataset.rows:
            folder = root / "unseen-font" / row["composition_id"]
            folder.mkdir(parents=True, exist_ok=True)
            source = (dataset.root / row["audio"]).resolve().parent / "score.midi"
            truth = Piece.load(dataset.root / row["events"])
            if not source.is_file():
                source = folder / "source.midi"
                write_midi(truth, source)
            mapping = None
            original = (dataset.root / row["audio"]).resolve().parent / "composition.json"
            if original.is_file():
                from lilylake.music.instruments import INSTRUMENTS

                mapping = {
                    INSTRUMENTS[p.instrument].program: p.instrument
                    for p in Piece.load(original).parts
                    if not INSTRUMENTS[p.instrument].percussion
                }
            rendered, domain = render_midi(
                source,
                folder / "audio.wav",
                engine="fluidsynth",
                sample_rate=config.sample_rate,
                soundfont=font,
                instrument_map=mapping,
            )
            rendered.save(folder / "events.json")
            rows.append(
                {
                    **row,
                    "audio": f"{row['composition_id']}/audio.wav",
                    "events": f"{row['composition_id']}/events.json",
                    "engine": "fluidsynth",
                    "augmentation": None,
                    "evaluation_only": True,
                }
            )
        manifest = root / "unseen-font/manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = {
            **evaluate_model(model, MusicDataset(manifest, config, args.split), config),
            **metadata,
            "domain": {**domain, "font_name": font.name, "evaluation_only": True},
        }
        (root / "unseen-font.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
