import shutil

import numpy as np
import pytest
import soundfile as sf

from lilylake.data import augment, compose, generate_dataset, split_for
from lilylake.lilypond import serialize
from lilylake.music.schema import Meter, Note, Part, Pedal, Piece, Tempo
from lilylake.rendering import compile_score, render_piece, synthesize


def test_compiler_88_keys_and_polyphony(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    p = Piece(
        [
            Part(
                "p",
                "grand_piano",
                [Note(i, (i - 21) * 0.125, (i - 21) * 0.125 + 0.125) for i in range(21, 109)],
            ),
            Part("v", "violin", [Note(60, 0, 2)]),
        ]
    )
    result = render_piece(p, tmp_path, "procedural", sample_rate=16000)
    assert result["midi"].exists()
    truth = Piece.load(tmp_path / "events.json")
    assert len([n for x in truth.parts if x.instrument == "grand_piano" for n in x.notes]) == 88
    assert result["audio"].exists()
    x, sr = sf.read(result["audio"])
    assert sr == 16000 and np.isfinite(x).all() and np.max(abs(x)) > 0


def test_compiler_rhythm_pedals_percussion(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    p = Piece(
        [
            Part(
                "p",
                "grand_piano",
                [Note(60, 0, 3), Note(64, 1 / 6, 0.5)],
                [Pedal(0, 1), Pedal(3, 0)],
            ),
            Part("d", "drum_kit", [Note(36, 0, 0.2), Note(42, 0.5, 0.6), Note(38, 1, 1.2)]),
        ],
        tempo_map=[Tempo(0, 120), Tempo(2, 90)],
        meter_map=[Meter(0, 4, 4), Meter(2, 3, 4)],
    )
    (tmp_path / "p.ly").write_text(serialize(p))
    assert compile_score(tmp_path / "p.ly", tmp_path / "compiled")["midi"].exists()


def test_invalid_lilypond_reports_diagnostics(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    (tmp_path / "p.ly").write_text("\\score { thisIsInvalid }")
    with pytest.raises(RuntimeError, match="LilyPond"):
        compile_score(tmp_path / "p.ly", tmp_path / "out")
    assert (tmp_path / "out.compiler.log").exists()


def test_seeded_composition_and_split():
    assert compose(42, level=6).to_dict() == compose(42, level=6).to_dict()
    assert compose(43, level=6).to_dict() != compose(42, level=6).to_dict()
    assert split_for("abc") == split_for("abc")
    p = compose(1, level=6)
    assert len(p.parts) == 2


def test_synth_repeatability_and_augmentation():
    p = Piece([Part("p", "grand_piano", [Note(21, 0, 0.2), Note(108, 0.25, 0.5)])])
    a = synthesize(p, 16000, seed=1)
    b = synthesize(p, 16000, seed=1)
    assert np.array_equal(a, b)
    assert a.shape[1] == 2
    x, meta = augment(a, 16000, seed=2)
    y, _ = augment(a, 16000, seed=2)
    assert np.array_equal(x, y) and x.shape == a.shape and np.isfinite(x).all()


def test_generated_pairs_and_manifest(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    manifest = generate_dataset(tmp_path, count=2, seed=10, level=6, sample_rate=16000)
    assert len(manifest) == 2 and manifest[0]["composition_id"] != manifest[1]["composition_id"]
    for row in manifest:
        folder = tmp_path / row["id"]
        for file in ["audio.wav", "score.ly", "score.midi", "events.json", "metadata.json"]:
            assert (folder / file).exists()
        assert Piece.load(folder / "events.json").duration > 0


def test_valid_layout_only_score_does_not_reuse_stale_midi(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    source = tmp_path / "layout.ly"
    source.write_text('\\version "2.24.3"\n\\score { { c\'4 } \\layout {} }')
    (tmp_path / "out.midi").write_bytes(b"stale unrelated MIDI")
    with pytest.raises(RuntimeError, match="no MIDI"):
        compile_score(source, tmp_path / "out")
    assert not (tmp_path / "out.midi").exists()
    assert compile_score(source, tmp_path / "valid", require_midi=False)["pdf"].exists()


def test_all_general_midi_drums_roundtrip_correct_pitch(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    p = Piece(
        [
            Part(
                "drums",
                "drum_kit",
                [Note(p, (p - 35) * 0.125, (p - 35) * 0.125 + 0.1) for p in range(35, 82)],
            )
        ]
    )
    result = render_piece(p, tmp_path)
    truth = Piece.load(result["events"])
    assert sorted(n.pitch for part in truth.parts for n in part.notes) == list(range(35, 82))


def test_hard_example_unisons_preserve_both_sources():
    from lilylake.hard_examples import compose_hard

    p = compose_hard(6000, "unison")
    assert p.to_dict() == compose_hard(6000, "unison").to_dict()
    assert len(p.parts) == 2
    assert all(n.pitch != 60 for part in p.parts for n in part.notes)
    assert {n.pitch for n in p.parts[0].notes} == {n.pitch for n in p.parts[1].notes}
    assert all(a.offset != b.offset for a, b in zip(p.parts[0].notes, p.parts[1].notes))


@pytest.mark.parametrize("parts", [[], [Part("p", "grand_piano"), Part("v", "violin")]])
def test_empty_predictions_compile_to_silent_midi(tmp_path, parts):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    result = render_piece(Piece(parts), tmp_path)
    assert result["midi"].exists()
    assert not any(p.notes for p in Piece.load(result["events"]).parts)
    audio, _ = sf.read(result["audio"])
    assert audio.size and np.max(np.abs(audio)) == 0


@pytest.mark.parametrize("instrument,pitch", [("piano", 60), ("bass_clarinet", 40)])
def test_compiler_labels_preserve_supplied_instrument_taxonomy(tmp_path, instrument, pitch):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    from lilylake.config import Config
    from lilylake.models import targets

    p = Piece([Part("source", instrument, [Note(pitch, 0, 0.5)])])
    result = render_piece(p, tmp_path)
    truth = Piece.load(result["events"])
    assert [p.instrument for p in truth.parts] == [instrument]
    y = targets(truth, 100, Config(instruments=[instrument]))
    assert y["frame"][:, 0, pitch].sum() > 0


def test_sustain_release_after_last_note_is_rendered(tmp_path):
    if not shutil.which("lilypond"):
        pytest.skip("LilyPond not installed")
    p = Piece([Part("p", "grand_piano", [Note(60, 0, 0.25)], [Pedal(0, 1), Pedal(3, 0)])])
    result = render_piece(p, tmp_path)
    truth = Piece.load(result["events"])
    assert any(
        pedal.time >= 2.99 and pedal.value < 0.5 for part in truth.parts for pedal in part.pedals
    )
    audio, sr = sf.read(result["audio"])
    assert len(audio) / sr >= 3.4
    assert np.max(np.abs(audio[int(2 * sr) : int(2.2 * sr)])) > 0


def test_render_rejects_ambiguous_same_program_taxonomy(tmp_path):
    piece = Piece(
        [Part("c", "clarinet", [Note(60, 0, 0.5)]), Part("b", "bass_clarinet", [Note(40, 0, 0.5)])]
    )
    with pytest.raises(ValueError, match="Ambiguous GM program"):
        render_piece(piece, tmp_path)
