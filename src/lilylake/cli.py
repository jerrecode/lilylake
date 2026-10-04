"""Reproducible end-to-end command line entrypoint."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from .config import Config


def doctor():
    import platform

    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                memory[line.split(":")[0]] = int(line.split()[1]) * 1024
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cuda_devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "tools": {x: shutil.which(x) for x in ["lilypond", "fluidsynth", "ffmpeg"]},
    }


def parser():
    root = argparse.ArgumentParser(
        prog="lilylake", description="Trainable audio -> continuous events -> LilyPond"
    )
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="inspect compute and rendering dependencies")
    g = sub.add_parser("generate-data", help="render deterministic aligned musical examples")
    g.add_argument("--output", required=True)
    g.add_argument("--count", type=int, default=16)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--level", type=int, default=6)
    g.add_argument("--instruments", nargs="+")
    g.add_argument("--engine", choices=["procedural", "fluidsynth"], default="procedural")
    g.add_argument("--soundfont")
    g.add_argument("--sample-rate", type=int, default=16000)
    g.add_argument("--randomize", action="store_true")
    t = sub.add_parser("train", help="train or resume the multi-task neural model")
    t.add_argument("--manifest")
    t.add_argument(
        "--initialize", help="initialize weights for a new dataset without restoring its optimizer"
    )
    t.add_argument("--output", required=True)
    t.add_argument("--config")
    t.add_argument("--epochs", type=int)
    t.add_argument("--resume", nargs="?", const="auto")
    t.add_argument(
        "--sanity-overfit",
        action="store_true",
        help="explicitly use identical training/validation data for a pipeline sanity check",
    )
    t.add_argument("--online-examples", type=int, default=0)
    t.add_argument("--curriculum", action="store_true")
    e = sub.add_parser("evaluate", help="score a checkpoint against held-out aligned audio")
    e.add_argument("--manifest", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", choices=["train", "validation", "test"], default="test")
    e.add_argument("--output", required=True)
    a = sub.add_parser("transcribe", help="infer events from audio and engrave a score")
    a.add_argument("audio")
    a.add_argument("--checkpoint", help="defaults to the included development checkpoint")
    a.add_argument("-o", "--output", required=True)
    a.add_argument("--tempo", type=float, help="override heuristic tempo estimation")
    a.add_argument("--no-validate", action="store_true")
    a.add_argument("--diagnostics", action="store_true")
    v = sub.add_parser("validate", help="compile a trusted LilyPond source and save diagnostics")
    v.add_argument("score")
    v.add_argument("--output", default="outputs/validation/score")
    r = sub.add_parser("render", help="compile LilyPond and synthesize its MIDI")
    r.add_argument("score")
    r.add_argument("--output", required=True)
    r.add_argument("--engine", choices=["procedural", "fluidsynth"], default="procedural")
    r.add_argument("--soundfont")
    r.add_argument("--sample-rate", type=int, default=16000)
    r.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("benchmark", help="profile feature extraction and model inference")
    b.add_argument("--config")
    b.add_argument("--seconds", type=float, default=5)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--output", default="outputs/benchmark.json")
    s = sub.add_parser("regression-suite", help="generate the explicit critical evaluation cases")
    s.add_argument("--output", required=True)
    s.add_argument("--engine", choices=["procedural", "fluidsynth"], default="procedural")
    s.add_argument("--soundfont")
    merge = sub.add_parser(
        "merge-data", help="combine aligned manifests while enforcing composition splits"
    )
    merge.add_argument("manifests", nargs="+")
    merge.add_argument("--output", required=True)
    external = sub.add_parser(
        "import-maestro", help="import a locally licensed MAESTRO copy using official metadata"
    )
    external.add_argument("--root", required=True)
    external.add_argument("--metadata", required=True)
    external.add_argument("--output", required=True)
    hard = sub.add_parser(
        "generate-hard-data", help="render independent unison, solo-string and dense-piano failures"
    )
    hard.add_argument("--output", required=True)
    hard.add_argument("--count", type=int, default=32)
    hard.add_argument("--seed", type=int, default=6000)
    hard.add_argument("--engine", choices=["procedural", "fluidsynth"], default="procedural")
    hard.add_argument("--soundfont")
    return root


def _dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
        elif args.command == "generate-hard-data":
            from .hard_examples import generate_hard_dataset

            if args.count < 1:
                raise ValueError("--count must be positive")
            generate_hard_dataset(args.output, args.count, args.seed, args.engine, args.soundfont)
        elif args.command == "generate-data":
            from .data import generate_dataset

            if args.count < 1:
                raise ValueError("--count must be positive")
            generate_dataset(
                args.output,
                args.count,
                args.seed,
                args.level,
                args.engine,
                args.sample_rate,
                args.soundfont,
                args.randomize,
                args.instruments,
            )
        elif args.command == "train":
            from .training import load_checkpoint, train

            if not args.manifest and not args.online_examples:
                raise ValueError("Training requires --manifest or --online-examples")
            checkpoint = Path(args.output) / "last.pt" if args.resume == "auto" else args.resume
            config = (
                Config.load(args.config)
                if args.config
                else Config(**load_checkpoint(checkpoint)["config"])
                if checkpoint and Path(checkpoint).exists()
                else Config()
            )
            if args.epochs:
                config = Config(**{**config.to_dict(), "epochs": args.epochs})
            train(
                args.manifest,
                args.output,
                config,
                args.sanity_overfit,
                args.resume,
                args.online_examples,
                args.curriculum,
                args.initialize,
            )
        elif args.command == "evaluate":
            from .inference import select_device
            from .models import EventModel
            from .training import MusicDataset, evaluate_model, load_checkpoint

            state = load_checkpoint(args.checkpoint)
            config = Config(**state["config"])
            torch.set_num_threads(config.threads)
            device = select_device(config.device)
            model = EventModel(config).to(device)
            model.load_state_dict(state["model"])
            report = evaluate_model(
                model, MusicDataset(args.manifest, config, args.split), config, device
            )
            report["checkpoint"] = Path(args.checkpoint).name
            report["split"] = args.split
            _dump(args.output, report)
            print(json.dumps({k: v for k, v in report.items() if k != "composition_ids"}, indent=2))
        elif args.command == "transcribe":
            from .inference import transcribe_audio

            piece, prediction = transcribe_audio(
                args.audio, args.checkpoint, args.output, args.tempo, not args.no_validate
            )
            if args.diagnostics:
                from .diagnostics import piano_roll_svg

                output = Path(args.output)
                folder = output.parent if output.suffix == ".ly" else output
                piano_roll_svg(piece, folder / "piano_roll.svg")
            print(
                json.dumps(
                    {
                        "notes": sum(len(p.notes) for p in piece.parts),
                        "parts": [p.instrument for p in piece.parts],
                        "output": args.output,
                    }
                )
            )
        elif args.command == "validate":
            from .rendering import compile_score

            result = compile_score(args.score, args.output, require_midi=False)
            print(json.dumps({k: str(v) for k, v in result.items()}))
        elif args.command == "render":
            from .rendering import compile_score, render_midi

            result = compile_score(args.score, Path(args.output) / "score")
            piece, metadata = render_midi(
                result["midi"],
                Path(args.output) / "audio.wav",
                args.engine,
                args.sample_rate,
                args.soundfont,
                args.seed,
            )
            piece.save(Path(args.output) / "events.json")
            _dump(Path(args.output) / "metadata.json", metadata)
            print(json.dumps(metadata, indent=2))
        elif args.command == "benchmark":
            import numpy as np

            from .audio import features
            from .models import EventModel

            config = Config.load(args.config) if args.config else Config()
            torch.set_num_threads(config.threads)
            model = EventModel(config).eval()
            audio = np.zeros(int(config.sample_rate * args.seconds), dtype=np.float32)
            durations = []
            with torch.inference_mode():
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    x = features(audio, config)
                    f = time.perf_counter() - start
                    start = time.perf_counter()
                    model(x[None])
                    m = time.perf_counter() - start
                    durations.append(
                        {
                            "feature_seconds": f,
                            "model_seconds": m,
                            "realtime_factor": (f + m) / args.seconds,
                        }
                    )
            report = {
                "environment": doctor(),
                "config": config.to_dict(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "runs": durations,
            }
            _dump(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "merge-data":
            from .adapters import merge_manifests

            print(json.dumps({"examples": len(merge_manifests(args.manifests, args.output))}))
        elif args.command == "import-maestro":
            from .adapters import import_maestro

            print(
                json.dumps(
                    {
                        "examples": len(import_maestro(args.root, args.metadata, args.output)),
                        "license": "CC-BY-NC-SA-4.0",
                    }
                )
            )
        elif args.command == "regression-suite":
            from .suite import generate_suite

            generate_suite(args.output, args.engine, args.soundfont)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"lilylake: {error}", file=sys.stderr)
        return 1
    return 0
