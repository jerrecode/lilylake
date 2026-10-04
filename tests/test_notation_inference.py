import pytest

from lilylake.music.infer import infer_notation
from lilylake.music.schema import Note, Part, Piece


def test_tempo_hypotheses_for_regular_quarter_notes():
    p = Piece(
        [
            Part(
                "p",
                "grand_piano",
                [
                    Note(60 + i % 5, i * 0.5, i * 0.5 + 0.3, 100 if i % 4 == 0 else 70)
                    for i in range(24)
                ],
            )
        ]
    )
    report = infer_notation(p)
    assert report["tempo"]["bpm"] == pytest.approx(120, abs=2)
    assert report["tempo"]["alternatives"] and report["tempo"]["confidence"] <= 1
    assert p.parts[0].notes[1].onset == 0.5


def test_unknown_notation_and_user_tempo_are_explicit():
    report = infer_notation(Piece(), tempo=90)
    assert report["tempo"]["source"] == "user" and report["tempo"]["bpm"] == 90
    assert report["key"]["tonic"] is None and report["meter"]["confidence"] == 0
    report = infer_notation(Piece())
    assert report["tempo"]["source"] == "fallback" and report["tempo"]["confidence"] == 0
