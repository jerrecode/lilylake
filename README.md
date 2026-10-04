# LilyLake

A real, trainable **audio → continuous instrument-aware note events → LilyPond** research system.

LilyLake uses an actual PyTorch acoustic model, not a text generator or a pitch-per-frame monophonic shortcut. Independent instrument/pitch heads permit arbitrary polyphony and same-pitch piano/violin events. LilyPond itself compiles scores and produces MIDI. Two explicit synthesis backends produce paired training audio. **This is a development baseline, not a professionally trained transcription service.** See [measured experiments](reports/) and [capability status](docs/status.md).

## Install

Python 3.11+, PyTorch, NumPy, SciPy, SoundFile and Mido. On Debian/Ubuntu:

```bash
sudo apt-get install lilypond fluidsynth fluid-soundfont-gm ffmpeg
python -m venv .venv
. .venv/bin/activate
# CPU PyTorch avoids downloading CUDA dependencies on CPU-only machines.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
python -m lilylake doctor
python -m pytest
```

These commands install the compiler and synthesis tools, create an isolated Python environment, install CPU PyTorch and the project, then inspect dependencies and run tests. Tests requiring LilyPond skip if unavailable; CI installs it and runs them. Windows/macOS users should install LilyPond and FFmpeg separately and ensure their executables are on PATH.

## Generate, train, evaluate, transcribe

```bash
python -m lilylake generate-data --output outputs/data --count 128 --seed 1000 --level 6
python -m lilylake train --manifest outputs/data/manifest.jsonl --config configs/cpu.json --output outputs/run
python -m lilylake train --manifest outputs/data/manifest.jsonl --output outputs/run --resume --epochs 60
python -m lilylake evaluate --manifest outputs/data/manifest.jsonl --checkpoint outputs/run/best.pt --split test --output outputs/test.json
python -m lilylake transcribe recording.wav --checkpoint outputs/run/best.pt --output outputs/transcription/score.ly --tempo 120 --diagnostics
python -m lilylake render outputs/transcription/score.ly --output outputs/playback
```

The sequence creates paired examples, trains, resumes from the last epoch, evaluates unseen compositions, transcribes audio and renders the generated notation. Transcription writes `score.ly`, `events.json` and `confidence.json`; validation also creates a PDF, MIDI and compiler log. The tempo option is currently an explicit notation assumption. Continuous seconds in the event JSON are preserved independently of quantization. Diagnostic SVG shows time horizontally and MIDI pitch vertically for each instrument.

For sample-based synthesis use `--engine fluidsynth --soundfont /path/to/FluidR3_GM.sf2` with generation/render commands. The SoundFont SHA-256 is recorded. The procedural renderer is a deterministic educational additive synth, **not a realistic grand piano or bowed-string simulator**. Choosing FluidSynth never silently falls back to it.

```bash
python -m lilylake train --online-examples 32 --curriculum --output outputs/online --config configs/cpu.json
python -m lilylake regression-suite --output outputs/challenges
python -m lilylake benchmark --output outputs/benchmark.json
python -m lilylake validate examples/piano_violin.ly
```

Online training generates seeded examples on demand through the actual compiler, using a bounded cache. Curriculum advances only when validation note-with-offset F1 exceeds the configured gate. The regression suite generates 24 acoustic challenges; generating a case does not mean the model can recognize it.

## Architecture

Harmonic STFT magnitudes and positive temporal differences feed a compact pitch-equivariant residual temporal CNN. Four heads produce onset, frame, offset and velocity tensors with independent instrument and MIDI pitch axes. Instrument masks include A0–C8 for piano. Weighted BCE, frame Dice and masked velocity regression provide supervision. A connected-region onset decoder plus offset/frame hysteresis reconstructs continuous notes; rational-grid quantization, interval voice allocation, staff assignment and deterministic LilyPond engraving follow.

See [specification](docs/superpowers/specs/2026-10-04-lilylake-design.md), [implementation plan](docs/superpowers/plans/2026-10-04-lilylake.md), [research](docs/research.md), [schema and extension guide](docs/architecture.md) and [data licenses](configs/datasets.json).

## Honest limits

A renderable instrument mapping is not an accurate trained instrument recognizer. The small baseline requires substantially more diverse data and training for real recordings, dense mixtures and reliable 88-key coverage. Sustain controls are represented/rendered but there is no learned pedal head yet. Bow direction, bow speed, articulation, continuous bends, glissando, soft pedal recognition, calibrated uncertainty, robust tempo/meter/key inference and source separation remain research work. Quantization favors rational durations and can still produce awkward engraving. Concert-pitch notation is used for transposing instruments.

Arbitrary LilyPond can execute Scheme: compile only sources you trust or run external scores inside an OS sandbox. Checkpoints use PyTorch's safe weights-only loader. External recordings and SoundFonts are not bundled. No copyrighted corpus is silently downloaded.
