# Representations and extension contracts

## Performance schema v1
`Piece` has unique-ID parts, strictly ordered tempo/meter/key maps beginning at zero, title and metadata. Every `Note` contains integer MIDI pitch 0..127, continuous seconds onset/offset, MIDI velocity 1..127, per-field confidence, optional articulation and expression dictionary. `Pedal` supports CC64 sustain, CC66 sostenuto and CC67 soft pedal. Notes must have positive finite duration. `Part.instrument` selects a taxonomy entry, not a quality guarantee. Serialization to JSON preserves original performance timing.

`quantize` produces a separate `Score` with rational quarter-note beats. Exactly aligned note durations merge into chords; interval coloring separates different overlaps. Piano pitch below MIDI60 enters bass staff. The serializer writes readable standard durations/dots where possible and exact scaled duration notation otherwise. Full tuplet grouping and harmonic spelling remain improvements. All original continuous events remain available regardless of notation.

## Acoustic tensors
Features: batch x 2*harmonics x frames x 128. Log STFT amplitudes are sampled at each pitch's harmonics; above-Nyquist harmonics are masked. Outputs: batch x frames x instruments x 128 for each head. Separate axes allow piano C4 and violin C4 concurrently. Piano valid pitch mask is 21..108. No top-k cap limits polyphony. Fixed FFT compromises frequency vs time resolution; change `n_fft`, `hop` or add a multi-resolution frontend for serious bass work.

## Labels and playback
`composition.json` is the pre-engraving composition. `events.json` is read from the exact compiler MIDI used for synthesis. This captures LilyPond's timing/velocity/articulation semantics. It represents MIDI note-off/key release, not the end of reverb. FluidSynth uses the supplied font; procedural synthesis uses known additive oscillators and note envelopes. Both operate on compiler MIDI, so ground-truth labels do not assume original score timing survived quantization unchanged.

## Training and extension
Add an instrument to `music/instruments.py`, then include its name in a new versioned config and generate labeled mixtures. Head shapes and masks derive from that list. Existing checkpoints require their original taxonomy; extending it requires a new checkpoint or explicit migration. Add a renderer through `render_midi`; always return the same event label semantics and record engine/font provenance. Dataset adapters should emit JSONL manifest rows with composition identity, split, audio/events paths and verified license. Custom augmentation must preserve timing or transform labels consistently.

Losses mask instrument pitch ranges and padded times. Velocity regression is supervised only on active notes. Training caches are bounded per worker; `workers=0` keeps RAM predictable. Atomic best/last checkpoints retain model/optimizer/scheduler/scaler/RNG and run metadata. CPU exact-resume tests compare all tensors; GPU nondeterminism can remain. Checkpoint interval is one epoch, so exceptionally large epochs should be kept bounded by online sample count.

## Evaluation
Maximum-cardinality one-to-one matching requires identical MIDI pitch and instrument and 50 ms onset tolerance. Offset tolerance is max(50 ms,20% reference duration). Duplicate predictions count as false positives. Empty reference/prediction returns zero F1 with zero support. Frame metrics are distinct from note metrics. Successful compiler syntax is distinct from musical or acoustic accuracy.
