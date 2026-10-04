# Measured CPU experiments

These are development experiments on short, generated pieces, not evidence of reliable real-recording transcription. All labels come from the same LilyPond MIDI used to render each example. Note matching requires the correct instrument and pitch, onset within 50 ms, and offset within max(50 ms, 20% of reference duration). Scores below are micro-averaged; supports are reference notes. Timing and velocity errors are measured on onset-matched pairs and therefore exclude missed notes.

| Experiment | Pieces / notes | Onset F1 | Onset + offset F1 | Evidence |
|---|---:|---:|---:|---|
| Tiny piano sanity overfit | Same four training examples | 1.000 | 1.000 | Pipeline check, not generalization |
| Single piano notes, unseen compositions | 9 / 9 | 1.000 | 1.000 | `reports/piano-heldout.json` |
| Piano weights, unseen FluidR3 renderer | 9 / 9 | See report | 0.200 | `reports/piano-unseen-renderer.json` |
| Piano + violin development domains, mixture weights | 12 / 111 | 0.606 | 0.447 | `reports/mixed-heldout.json` |
| Expanded domains, hard-example weights | 16 / 139 | 0.625 | 0.442 | `reports/hard-heldout.json` |
| Same 16 pieces, unseen TimGM6mb SoundFont | 16 / 139 | 0.556 | 0.367 | `reports/unseen-soundfont.json` |

The default hard-example checkpoint has per-instrument note-with-offset F1 of 0.391 for grand piano and 0.525 for violin on the 16-piece development test set. Its timing errors among onset matches are 10.44 ms onset MAE and 80.50 ms offset MAE. Confidence scores have not been calibrated. Sample sizes are small; no statistical confidence intervals or real-recording claims are implied.

The named piano + violin challenge reconstructs both parts and compiles/renders, with onset F1 0.742 and note-with-offset F1 0.516. The same-pitch C4 challenge recovers both true source onsets but adds an extra violin note and misses the piano offset: onset F1 0.800, note-with-offset F1 0.400. The dense-chord challenge still has note-with-offset F1 **0.000**. All-88 sequential key challenge onset F1 is 0.876 and note-with-offset F1 0.734; this is not proof of arbitrary dense 88-key recognition. Per-case JSON round trips and the full critical-suite report expose failures, including unsupported instrument families.

## Training performed

CPU only: two training threads, width 16, depth 2, four harmonics, 16 kHz, 10 ms hop. The largest training pool contains 176 generated compositions before split: 96 single piano examples, 32 randomized procedural piano/violin pieces, 16 randomized FluidR3 pieces, and 32 separately seeded difficult unison/violin/dense/crossing examples. Training progressed through 160 tiny-overfit epochs, 100 piano epochs, 60 mixture epochs and 20 hard-example epochs. Validation selected the exported weights; the export manifest lists SHA-256 hashes and selected epochs. Best and last checkpoints, optimizer/scheduler state, history and RNG are produced by every training run.

The mixture run switched from random batches to duration buckets after epoch 31, reducing epochs from approximately 14–16 seconds to about 6 seconds. This changes sample ordering; the historical run is documented rather than presented as bit-exactly reproducible from current defaults. Current CPU tests establish exact interruption/resume under the same configuration. Reports record configurations and dependency versions; newer training runs also fingerprint source contents and both train/validation audio and events. Online curriculum families now have stable seed-based splits; manifest merging and training reject seed variants that cross cohorts. The published static pools use non-overlapping seed ranges and are unaffected. Older development exports predate data-content fingerprints and support only identity checks when resumed; initialization into a new experiment is preferable.

## Reproduce data and evaluate exported weights

Install LilyPond and the package as described in README. SoundFonts are supplied locally and excluded from Git. FluidR3 training usage and TimGM evaluation usage, hashes, sources and distribution-declared licenses are recorded in `configs/datasets.json`.

```bash
python -m lilylake generate-data --output outputs/piano/data --count 96 --seed 1000 --level 1 --instruments grand_piano
python -m lilylake generate-data --output outputs/mixture/data --count 32 --seed 2000 --level 6 --randomize
python -m lilylake generate-data --output outputs/fluidsynth-data --count 16 --seed 3000 --level 6 --randomize --engine fluidsynth --soundfont /path/to/FluidR3_GM.sf2
python -m lilylake generate-hard-data --output outputs/hard/data --count 32 --seed 6000
python -m lilylake merge-data outputs/piano/data/manifest.jsonl outputs/mixture/data/manifest.jsonl outputs/fluidsynth-data/manifest.jsonl outputs/hard/data/manifest.jsonl --output outputs/hard-combined/manifest.jsonl
python scripts/evaluate_domains.py --manifest outputs/hard-combined/manifest.jsonl --checkpoint examples/development.pt --unseen-soundfont /path/to/TimGM6mb.sf2
```

This compares **the same held-out compositions** across two rendering domains. Do not merge the unseen-font manifest into a training pool: it deliberately reuses test composition IDs. FluidSynth/LilyPond version changes can alter audio or MIDI semantics, so reproduce reported versions and font hashes for close comparisons. `scripts/experiments.py` generates and retrains the first two larger experiment stages. To train the hard stage:

```bash
python -m lilylake train --manifest outputs/hard-combined/manifest.jsonl --config configs/hard-development.json --initialize examples/mixture-development.pt --output outputs/hard-combined/run
python -m lilylake regression-suite --output outputs/challenges
python -m lilylake evaluate --manifest outputs/challenges/manifest.jsonl --checkpoint examples/development.pt --split test --output outputs/challenges.json
```

No external real corpus was used to train these weights. MAESTRO has a local importer; MAESTRO and Slakh remain optional separately licensed sources. Learned pedal and articulation prediction, continuous string bends, source separation, advanced notation, and reliable generalization require substantial additional research, data and compute.
