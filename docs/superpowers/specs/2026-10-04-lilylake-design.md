# LilyLake specification

## Goal and scientific boundary
An executable, trainable audio-to-events-to-LilyPond pipeline. Continuous acoustic events remain authoritative; notation is a derived representation. Real-world accuracy is an experimental outcome, never implied by tensor support. Key release and acoustic decay are distinct; bow direction is unknown unless separately supervised. No source separation or expression claim without evidence.

## Environment assessment
Initial environment: 9 logical CPUs, approximately 9 GB available memory, 30 GB free disk, FFmpeg; no initial PyTorch, LilyPond, or FluidSynth. Empty repository. CPU-first development with accelerator detection. Dependencies will be installed where permitted; unavailable engines produce actionable errors, not fake validation.

## Architecture and alternatives
Choose a compact shared spectral encoder and temporal convolution with independent instrument x 128 MIDI-pitch heads for onset, offset, frame and velocity. Piano uses a 21..108 validity mask. Configurable taxonomy enables same-pitch simultaneous sources without a polyphony cap. Compare this tractable baseline against Onsets and Frames' explicit temporal supervision and MT3's sequence decoding (see research.md). Large transformers and learned source separation require substantial training and are deferred until this baseline is measured.

Modules: music/schema (validated performance and notation dataclasses), music/notation (tempo conversion, quantization, interval voices), lilypond (deterministic absolute-pitch serializer), rendering (compile LilyPond to MIDI, parse compiler MIDI into actual rendered-event labels, FluidSynth and independent procedural MIDI synthesizer), data (seeded composers, coverage, augmentation, manifests), audio (STFT), models (multi-task encoder), training (atomic resumable checkpoints), inference (hysteresis/re-onset decode, chunked files), evaluation (one-to-one instrument-aware matching), CLI and diagnostics.

## Renderer and exact labels
Canonical full route is score.ly -> LilyPond -> score.midi -> synthesis -> audio.wav. Compiler MIDI is the ground truth for rendered acoustic targets: this prevents score articulation, velocity and quantization mismatch. Store both original composition and compiled performance. A procedural MIDI synth allows independent generation without SoundFonts and held-out engine tests; it is explicitly a synthetic timbre, not realistic instrument emulation. FluidSynth supports user-provided, provenance-recorded fonts. Never silently substitute an engine. Library Scheme evaluation means arbitrary external .ly files must be trusted; subprocess argument arrays and timeouts are used.

## Representation and notation
Performance: Piece, parts with unique IDs, MIDI pitch, seconds onset/offset, velocity, per-field confidence, expressive attributes defaulting unknown, pedal controls, tempo/meter/key maps. Notation: rational beat intervals, staves and interval-colored voices, chords, rests, ties split at bar boundaries; dotted and triplet durations, meter/tempo changes and pedal control tracks. MIDI sounding pitches are canonical, transposing instruments use concert pitch initially and this is documented. Confidence is persisted separately from engraving. No premature quantization.

## Generation
Seeded structured melody, arpeggio, progression, counterpoint, dense clusters and boundary composers. Curriculum stages single keys -> phrases -> chords -> two instruments -> ensemble -> randomized domains. Coverage scheduler preferentially chooses underrepresented pitch/polyphony/instrument/strategy bins; no impossible Cartesian product. Stable composition IDs precede augmentation and split assignment. Labels use MIDI events, seeds/configurations/engine/font hashes are recorded. Domain randomization: deterministic gain, noise, EQ, tuning for procedural engine, stereo panning and reverb; timing remains aligned. Coverage is descriptive, not evidence of recognition.

## Training
STFT log magnitude plus positive temporal difference features; independent output heads with pitch masks. Weighted BCE for imbalance and masked velocity regression. CPU/CUDA/MPS selection, gradient clipping, optional autocast, accumulation, seeded online examples or manifests, held-out validation, learning rate scheduling and best/last checkpoints. Atomic save uses safe weights-only loading. Capture config, revision, versions, history, RNG/optimizer/scheduler state and dataset identities. Validate gradient flow and tiny-set overfit before broader jobs. Curriculum increases only after validation gates; difficulty need not advance on poor metrics.

## Evaluation
Frame P/R/F1; onset and onset+offset one-to-one bipartite note matching with same instrument/pitch, 50 ms onset tolerance and max(50 ms,20% duration) offset tolerance. Per-instrument counts and macro/pooled metrics; onset/offset MAE and velocity MAE/RMSE. Empty/empty does not imply acoustic success. Compile/render success and score quantization timing errors separate from acoustics. Held-out compositions and renderer domains. Round-trip ground score -> audio -> neural predictions -> engraved score -> compiled audio. Publish measured results and failed criteria.

## Acceptance criteria
1. Compiler verifies actual generated scores and synthesizer creates finite, nonempty WAV plus aligned JSON/MIDI.
2. All 88 piano pitches serialize/compile; synth targets include all keys and multi-instrument same pitches independently.
3. Meaningful unit and integration tests cover boundaries, tempo, ties, percussion, deterministic generation, shapes/masks, re-onset decoding, evaluation matching and checkpoint continuation.
4. A real neural model receives gradients, reduces tiny-set loss and produces measured note metrics; held-out results may remain poor and must be reported.
5. CLI generates, trains/resumes, transcribes audio, evaluates, validates, renders and benchmarks. Predictions use only audio, no sidecar truth.
6. Inference produces events.json, confidence.json and score.ly, with compiler validation when requested and no hidden fallback.
7. Reproducible experiment scripts, configuration, limitations, data licenses and checkpoint instructions are committed.
8. Reliable 88-key recognition, piano+violin separation and real-audio generalization are quality targets marked achieved only when held-out measurements support them.

## Risks
Finite CPU training cannot establish professional transcription. Dense same-pitch mixtures can be non-identifiable. Fixed FFT compromises bass/time resolution; configurable multi-resolution extension is necessary for serious full-range training. Meter and key are hypotheses. Synthetic-to-real shift needs legally sourced real data and held-out fonts. Full LilyPond language parsing is delegated to its compiler; embedded project metadata is not acoustic inference.
