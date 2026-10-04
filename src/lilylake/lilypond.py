"""Deterministic engraving of derived scores; syntax authority is LilyPond itself."""
from fractions import Fraction
import json
from .music.schema import Piece
from .music.notation import quantize, assign_voices, seconds_to_beats
from .music.instruments import INSTRUMENTS, DRUM_NAMES


def pitch_name(pitch):
    names=['c','cis','d','ees','e','f','fis','g','aes','a','bes','b']
    octave=pitch//12-4
    return names[pitch%12]+("'"*octave if octave>=0 else ','*(-octave))


def duration_token(beat):
    beat=Fraction(beat)
    for den in [1,2,4,8,16,32,64,128]:
        for dots in range(3):
            value=Fraction(4,den)*sum((Fraction(1,2**i) for i in range(dots+1)),Fraction(0))
            if value==beat:return str(den)+'.'*dots
    return f'4*{beat.numerator}/{beat.denominator}'


def _quoted(text):
    return '"'+text.replace('\\','\\\\').replace('"','\\"').replace('\n',' ')+'"'


def serialize(piece:Piece,subdivisions=(1,2,3,4,6,8)):
    score=quantize(piece,subdivisions)
    # Meter events reset bar alignment. Ties are split at every bar boundary.
    boundaries=set()
    for i,m in enumerate(piece.meter_map):
        start=Fraction(seconds_to_beats(m.time,piece.tempo_map)).limit_denominator(96)
        stop=Fraction(seconds_to_beats(piece.meter_map[i+1].time,piece.tempo_map)).limit_denominator(96) if i+1<len(piece.meter_map) else score.end
        pos=start;bar=Fraction(4*m.numerator,m.denominator)
        while pos<=stop: boundaries.add(pos);pos+=bar
    for t in piece.tempo_map: boundaries.add(Fraction(seconds_to_beats(t.time,piece.tempo_map)).limit_denominator(96))
    boundaries=sorted(boundaries)
    globals=[]
    for t in piece.tempo_map: globals.append((Fraction(seconds_to_beats(t.time,piece.tempo_map)).limit_denominator(96),f'\\tempo 4 = {round(t.bpm)}'))
    for m in piece.meter_map: globals.append((Fraction(seconds_to_beats(m.time,piece.tempo_map)).limit_denominator(96),f'\\time {m.numerator}/{m.denominator}'))
    for k in piece.key_map: globals.append((Fraction(seconds_to_beats(k.time,piece.tempo_map)).limit_denominator(96),f'\\key {k.tonic} \\{k.mode}'))
    def controls(events):
        tokens=[];cursor=Fraction(0)
        for beat,text in sorted(events,key=lambda x:x[0]):
            if beat>cursor:tokens.append('s'+duration_token(beat-cursor))
            tokens.append(text);cursor=beat
        if score.end>cursor:tokens.append('s'+duration_token(score.end-cursor))
        return '{ '+' '.join(tokens)+' }'
    global_music=controls(globals)
    def voice(notes,drums=False):
        tokens=[];cursor=Fraction(0)
        def segments(start,end):
            cuts=[start]+[b for b in boundaries if start<b<end]+[end]
            return list(zip(cuts,cuts[1:]))
        for n in notes:
            for a,b in segments(cursor,n.onset):
                if b>a:tokens.append('r'+duration_token(b-a))
            names=[DRUM_NAMES.get(p,'sn') if drums else pitch_name(p) for p in n.pitches]
            note=names[0] if len(names)==1 else '<'+' '.join(names)+'>'
            spans=segments(n.onset,n.onset+n.duration)
            for j,(a,b) in enumerate(spans):
                token=note+duration_token(b-a)
                if j==0:
                    dynamic=['pp','p','mp','mf','f','ff'][min(5,n.velocity//22)]
                    token+='\\'+dynamic
                    token+={None:'','staccato':'-.','tenuto':'--','accent':'->','marcato':'-^','legato':''}[n.articulation]
                if j+1<len(spans):token+=' ~'
                tokens.append(token)
            cursor=n.onset+n.duration
        for a,b in segments(cursor,score.end):
            if b>a:tokens.append('r'+duration_token(b-a))
        return ('\\drummode' if drums else '\\absolute')+' { '+' '.join(tokens)+' }'
    staves=[]
    for index,p in enumerate(score.parts):
        inst=INSTRUMENTS[p.instrument];original=piece.parts[index]
        def staff(notes,clef,label):
            voices=assign_voices(notes) or [[]]
            context='DrumStaff' if inst.percussion else 'Staff'
            settings=f'instrumentName = {_quoted(label)}'
            if not inst.percussion:settings+=f' midiInstrument = {_quoted(inst.midi_name)}'
            streams=[global_music]+['\\new '+('DrumVoice' if inst.percussion else 'Voice')+' { '+voice(v,inst.percussion)+' }' for v in voices]
            if original.pedals:
                commands={64:('sustainOff','sustainOn'),66:('sostenutoOff','sostenutoOn'),67:('unaCorda','treCorde')}
                events=[]
                for ped in original.pedals:
                    off,on=commands[ped.controller]
                    if ped.controller==67:off,on='treCorde','unaCorda'
                    events.append((Fraction(seconds_to_beats(ped.time,piece.tempo_map)).limit_denominator(96),'\\'+(on if ped.value>=.5 else off)))
                streams.append(controls(events))
            clef_command='' if inst.percussion else '\\clef '+_quoted(clef)
            return f'\\new {context} \\with {{ {settings} }} << {{ {clef_command} }} '+ ' '.join(streams)+' >>'
        label=p.instrument.replace('_',' ').title()
        if inst.keyboard:
            from .music.notation import ScoreNote
            lower=[];upper=[]
            for n in p.notes:
                for target,pitches in [(lower,tuple(x for x in n.pitches if x<60)),(upper,tuple(x for x in n.pitches if x>=60))]:
                    if pitches:target.append(ScoreNote(pitches,n.onset,n.duration,n.velocity,n.articulation,n.quantization_error_sec))
            staves.append('\\new PianoStaff << '+staff(upper,'treble',label)+staff(lower,'bass','')+' >>')
        else: staves.append(staff(p.notes,inst.clef,label))
    if not staves:staves=[staff([], 'treble','Silence')] if score.parts else ['\\new Staff { r1 }']
    return '\\version "2.24.3"\n\\header { title = '+_quoted(piece.title)+' tagline = ##f }\n\\score {\n  <<\n    '+'\n    '.join(staves)+'\n  >>\n  \\layout { }\n  \\midi { }\n}\n'
