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
