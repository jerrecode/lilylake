# Engineering review and verification

An independent read-only review reproduced four runtime/data-integrity defects with the actual LilyPond toolchain. Each received a failing regression test, a root-cause fix and a passing follow-up review:

- Empty instrument streams produced no MIDI despite a MIDI block. Empty scores now contain a silent measure.
- Compiler MIDI canonicalized piano and bass-clarinet aliases, silently eliminating training positives under those configurations. Piece rendering now retains explicit source taxonomy; ambiguous names sharing one GM program are rejected.
- Late sustain releases vanished and sustained sound was truncated. Pedal commands now attach as post-events to timed skips; the timeline extends past the final control and audio allocation includes pedal release time.
- Resume accepted changed validation data while retaining earlier best scores. New checkpoints fingerprint both audio/event cohorts and online policy, and reject altered training/validation semantics. Legacy checkpoints receive conservative identity checks.

A minor MIDI program-attribution bug was fixed by capturing source program at note-on. A subsequent split audit found that curriculum levels could move variants of the same generating seed between cohorts. Online seed-family splits now remain stable across levels; manifest loading/merging reject exact and seed-family leakage across train, validation and test. Published static experiment pools use disjoint seed ranges and retain their historical split assignments.

Final local verification: **52 tests passed, zero skipped** with LilyPond 2.24.3; lint, dependency consistency and Python compilation also passed. The independent reviewer approved the original fixes after separately running 48 tests, then approved the seed-family changes after targeted adapter/resume checks. The final two tests additionally cover direct train/test-family leakage. All 24 regenerated critical scores compiled and rendered. The standalone domain-evaluation script reproduced the released held-out and unseen-SoundFont metrics exactly.

This verifies engineering behavior and development experiments. It does not establish professional transcription accuracy, real-recording generalization, accelerator reproducibility, calibrated uncertainty or unimplemented expressive inference. See `status.md` and `experiments.md`.
