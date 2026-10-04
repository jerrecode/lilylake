from fractions import Fraction
import pytest
from lilylake.music.schema import Piece, Part, Note, Tempo, Meter, Pedal
from lilylake.music.notation import seconds_to_beats, beats_to_seconds, quantize, assign_voices
from lilylake.music.midi import write_midi, read_midi
from lilylake.lilypond import serialize, pitch_name


def fixture_piece():
    return Piece(parts=[Part('piano', 'grand_piano', [Note(60, 0, 1, 90), Note(64, 0, .5, 70), Note(60, 1, 1.5, 80)]), Part('violin', 'violin', [Note(60, 0, 1.5, 60)])], tempo_map=[Tempo(0,120),Tempo(1,60)])


def test_invalid_notes_rejected():
    for args in [(128,0,1), (60,1,.5),(60,-1,1),(60,0,float('nan'))]:
        with pytest.raises(ValueError): Note(*args)


def test_tempo_conversion_piecewise_inverse():
    p=fixture_piece()
    assert seconds_to_beats(2,p.tempo_map)==3
    for s in [0,.123,1,2.345]: assert beats_to_seconds(seconds_to_beats(s,p.tempo_map),p.tempo_map)==pytest.approx(s)


def test_all_piano_keys_absolute_pitch():
    assert pitch_name(21)=='a,,,'
    assert pitch_name(60)=="c'"
    assert pitch_name(108)=="c'''''"
    assert len({pitch_name(p) for p in range(21,109)})==88


def test_json_roundtrip(tmp_path):
    p=fixture_piece(); p.save(tmp_path/'p.json')
    assert Piece.load(tmp_path/'p.json').to_dict()==p.to_dict()


def test_quantization_retains_performance_and_independent_sources():
    p=fixture_piece(); q=quantize(p)
    assert len(q.parts)==2
    assert p.parts[0].notes[0].offset==1
    assert q.parts[0].notes[0].duration==Fraction(2)
    voices=assign_voices(q.parts[0].notes)
    assert len(voices)>=2
    for v in voices:
        assert all(a.onset+a.duration<=b.onset for a,b in zip(v,v[1:]))


def test_midi_roundtrip_same_pitch_sources(tmp_path):
    p=fixture_piece(); p.parts[0].pedals=[Pedal(.2,1),Pedal(1.2,0)]
    write_midi(p,tmp_path/'p.mid'); q=read_midi(tmp_path/'p.mid')
    assert sum(len(x.notes) for x in q.parts)==4
    assert {x.instrument for x in q.parts}=={'grand_piano','violin'}
    assert sorted(n.offset for x in q.parts for n in x.notes)==pytest.approx([.5,1,1.5,1.5],abs=.002)
    assert q.parts[0].pedals[-1].value==0


def test_serializer_structures_and_escaped_title():
    p=fixture_piece();p.title='Test "score" \\ escape'
    ly=serialize(p)
    for token in ['\\version','\\score','\\new PianoStaff','\\new Staff','\\midi','\\absolute','\\tempo']: assert token in ly
    assert serialize(p)==ly
