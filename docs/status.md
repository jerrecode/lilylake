# Capability status

## Implemented and tested
Validated continuous-time notes, tempo/meter/key/pedal maps; GM taxonomy; independent same-pitch source axes; MIDI read/write; deterministic absolute LilyPond; piano staves, voices, chords, rests, rational/dotted durations and bar-crossing ties; percussion; compiler diagnostics; actual LilyPond MIDI ground truth; procedural and FluidSynth synthesis; seeded musical curricula; split-before-augmentation; coverage reports; gain/noise/EQ/echo randomization; harmonic STFT features; real multi-task neural training; atomic safe checkpoints; exact CPU resume; bounded feature cache; chunked inference; onset/offset decoder; one-to-one note evaluation; per-instrument/frame metrics; CLI and diagnostic piano rolls.

## Experimental evidence
See the committed JSON reports. Tiny-set overfit is a pipeline test on identical training/evaluation compositions, not generalization. Full-range serialization and successful rendering do not prove full-range recognition. CPU experiments provide explicit held-out composition and held-out renderer metrics.

## Not yet implemented or validated
Professional accuracy on real piano/violin mixtures; physical key release vs pedal acoustics; learned pedal/articulation heads; continuous pitch bend and glissando labels; bow direction/speed; learned source separation; expressive notation such as grace notes/slur inference; reliable tempo-change/meter/key inference from unrestricted audio (conservative event-derived heuristics are implemented); enharmonic context optimization; held-out SoundFont generalization using multiple fonts; hard-example mining; external real dataset training. GM aliases cannot distinguish concert grand/upright acoustic subtypes from an audio mixture. The default model taxonomy is grand_piano + violin; other mappings can be configured and require training.

No unsupported field is asserted as certain. Probabilities are uncalibrated model scores. Tempo, meter and key inference use conservative event-derived heuristics with explicit uncertainty and alternatives. Unknown key signatures are omitted; meter and tempo fallbacks are labeled assumptions.
