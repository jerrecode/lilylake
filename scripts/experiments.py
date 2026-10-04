"""Reproduce the small CPU experiment families. Run from the repository root.

This produces honest measurements; it does not impose optimistic metric gates.
All sample generation invokes LilyPond. Existing output paths are overwritten.
"""

import argparse
import json
from pathlib import Path

from lilylake.adapters import merge_manifests
from lilylake.config import Config
from lilylake.data import generate_dataset
from lilylake.evaluation import evaluate_notes
from lilylake.inference import transcribe_audio
from lilylake.models import EventModel
from lilylake.music.schema import Piece
from lilylake.rendering import render_midi
from lilylake.training import MusicDataset, evaluate_model, load_checkpoint, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/reproduce")
    parser.add_argument("--soundfont")
    parser.add_argument("--piano-epochs", type=int, default=100)
    parser.add_argument("--mixture-epochs", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = Config(
        width=16,
        depth=2,
        harmonics=4,
        batch_size=8,
        epochs=args.piano_epochs,
        validation_interval=10,
    )
    generate_dataset(root / "piano", 96, 1000, 1, instruments=["grand_piano"])
    train(root / "piano/manifest.jsonl", root / "piano-run", config)
    model = EventModel(config)
    model.load_state_dict(load_checkpoint(root / "piano-run/best.pt")["model"])
    reports = {}
    reports["piano_heldout"] = evaluate_model(
        model, MusicDataset(root / "piano/manifest.jsonl", config, "test"), config
    )
    generate_dataset(root / "mixture", 32, 2000, 6, randomize=True)
    manifests = [root / "piano/manifest.jsonl", root / "mixture/manifest.jsonl"]
    if args.soundfont:
        generate_dataset(
            root / "sampled",
            16,
            3000,
            6,
            engine="fluidsynth",
            soundfont=args.soundfont,
            randomize=True,
        )
        manifests.append(root / "sampled/manifest.jsonl")
    merge_manifests(manifests, root / "combined.jsonl")
    config = Config(**{**config.to_dict(), "epochs": args.mixture_epochs})
    train(
        root / "combined.jsonl", root / "mixture-run", config, initialize=root / "piano-run/best.pt"
    )
    mixed = EventModel(config)
    mixed.load_state_dict(load_checkpoint(root / "mixture-run/best.pt")["model"])
    reports["mixture_heldout"] = evaluate_model(
        mixed, MusicDataset(root / "combined.jsonl", config, "test"), config
    )
    row = next(
        json.loads(line)
        for line in (root / "combined.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "test"
    )
    p, _ = transcribe_audio(row["audio"], root / "mixture-run/best.pt", root / "roundtrip/score.ly")
    render_midi(root / "roundtrip/score.midi", root / "roundtrip/reconstruction.wav")
    reports["roundtrip"] = evaluate_notes(Piece.load(row["events"]), p)
    (root / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
