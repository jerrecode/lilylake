# Research and design decisions (2026-10-04)

Primary sources consulted:
- Hawthorne et al., Onsets and Frames (ISMIR 2018): https://archives.ismir.net/ismir2018/paper/000019.pdf . Joint onset/frame modeling motivates explicit temporal heads. This implementation is an independently implemented compact baseline, not a reproduction of reported accuracy.
- Gardner et al., MT3 (ICLR 2022): https://openreview.net/pdf?id=iMSjopcOn0p and https://github.com/magenta/mt3 . Joint multitrack modeling supports instrument-aware inference; T5X and larger training are less practical for this CPU session.
- MAESTRO: https://magenta.withgoogle.com/datasets/maestro . CC BY-NC-SA 4.0, aligned piano MIDI/audio, real performer data; do not bundle or silently download.
- Slakh: https://www.slakh.com/ . CC BY 4.0 stated by provider, synthetic multitrack stems; preserve attribution and inspect upstream MIDI provenance before reuse.
- LilyPond manual: https://lilypond.org/doc/v2.24/Documentation/notation/creating-midi-files . Actual compiler, not a handwritten validator, decides syntax and MIDI semantics.

Source separation may help but also introduces errors; the first baseline learns directly on mixtures. Bow direction/speed and physical key release under pedal cannot generally be uniquely recovered from a mixture: unknown fields are preferable to fabricated certainty. Staff assignment and meter are not unique; keep performance truth separate from notation assumptions.

- Ellis (2007) and librosa documentation: https://librosa.org/doc/main/api/generated/librosa.beat.beat_track.html . Onset autocorrelation motivates the conservative tempo hypothesis module; LilyLake does not implement or claim the full Ellis dynamic-programming tracker.

Compiler debugging verified that pedal commands are post-events, attached after a timed skip at their event position. See [LilyPond piano notation](https://lilypond.org/doc/v2.24/Documentation/notation/piano). Actual compiler regressions cover late sustain releases instead of relying only on textual output.
