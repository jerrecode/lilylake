"""MIDI interchange with tempo-aware seconds and independent program/channel sources."""

from collections import defaultdict, deque
from pathlib import Path

import mido

from .instruments import INSTRUMENTS, PROGRAM_NAMES
from .notation import seconds_to_beats
from .schema import Meter, Note, Part, Pedal, Piece, Tempo


def write_midi(piece: Piece, path):
    mid = mido.MidiFile(ticks_per_beat=960)
    conductor = mido.MidiTrack()
    mid.tracks.append(conductor)
    events = []
    for t in piece.tempo_map:
        events.append(
            (
                round(seconds_to_beats(t.time, piece.tempo_map) * 960),
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t.bpm)),
            )
        )
    for m in piece.meter_map:
        events.append(
            (
                round(seconds_to_beats(m.time, piece.tempo_map) * 960),
                mido.MetaMessage(
                    "time_signature", numerator=m.numerator, denominator=m.denominator
                ),
            )
        )
    _append(conductor, events)
    channels = iter(c for c in range(16) if c != 9)
    for p in piece.parts:
        inst = INSTRUMENTS[p.instrument]
        try:
            channel = 9 if inst.percussion else next(channels)
        except StopIteration:
            raise ValueError("MIDI supports at most 15 pitched parts plus drums") from None
        track = mido.MidiTrack()
        mid.tracks.append(track)
        events = [
            (0, mido.MetaMessage("track_name", name=p.id)),
            (0, mido.Message("program_change", channel=channel, program=inst.program)),
        ]
        for n in p.notes:
            for sec, typ, velocity in [(n.onset, "note_on", n.velocity), (n.offset, "note_off", 0)]:
                events.append(
                    (
                        round(seconds_to_beats(sec, piece.tempo_map) * 960),
                        mido.Message(typ, channel=channel, note=n.pitch, velocity=velocity),
                    )
                )
        for pedal in p.pedals:
            events.append(
                (
                    round(seconds_to_beats(pedal.time, piece.tempo_map) * 960),
                    mido.Message(
                        "control_change",
                        channel=channel,
                        control=pedal.controller,
                        value=round(pedal.value * 127),
                    ),
                )
            )
        _append(track, events)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(path)


def _append(track, events):
    last = 0

    def priority(msg):
        return 0 if msg.type == "note_off" else 2 if msg.type == "note_on" else 1

    for tick, msg in sorted(events, key=lambda x: (x[0], priority(x[1]))):
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def read_midi(path, instrument_map=None):
    mid = mido.MidiFile(path)
    clock = 0.0
    tempo = 500000
    programs = defaultdict(int)
    active = defaultdict(deque)
    parts = {}
    tempos = [Tempo(0, 120)]
    meters = [Meter()]

    def part(channel):
        instrument = (
            "drum_kit"
            if channel == 9
            else (instrument_map or {}).get(
                programs[channel], PROGRAM_NAMES.get(programs[channel], "unknown")
            )
        )
        key = (channel, instrument)
        if key not in parts:
            parts[key] = Part(f"{instrument}_{channel}", instrument)
        return parts[key]

    for msg in mido.merge_tracks(mid.tracks):
        clock += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            t = Tempo(clock, mido.tempo2bpm(tempo))
            if clock == tempos[-1].time:
                tempos[-1] = t
            else:
                tempos.append(t)
        elif msg.type == "time_signature":
            m = Meter(clock, msg.numerator, msg.denominator)
            if clock == meters[-1].time:
                meters[-1] = m
            else:
                meters.append(m)
        elif msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            p = part(msg.channel)
            active[(msg.channel, msg.note)].append((clock, msg.velocity, p, programs[msg.channel]))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            q = active[(msg.channel, msg.note)]
            if q:
                on, vel, p, program = q.popleft()
                if clock > on:
                    p.notes.append(
                        Note(
                            msg.note,
                            on,
                            clock,
                            vel,
                            expression={"source_midi_program": program}
                            if p.instrument == "unknown"
                            else {},
                        )
                    )
        elif msg.type == "control_change" and msg.control in [64, 66, 67]:
            part(msg.channel).pedals.append(Pedal(clock, msg.value / 127, msg.control))
    for (channel, pitch), q in active.items():
        for on, vel, p, program in q:
            if clock > on:
                p.notes.append(
                    Note(pitch, on, clock, vel, expression={"unterminated_midi_note": True})
                )
    # Merge compiler-generated staves of the same GM instrument; no duplicate pitch collapse.
    merged = {}
    for p in parts.values():
        if p.instrument not in merged:
            merged[p.instrument] = Part(p.instrument, p.instrument)
        merged[p.instrument].notes.extend(p.notes)
        merged[p.instrument].pedals.extend(p.pedals)
    for p in merged.values():
        p.notes.sort(key=lambda n: (n.onset, n.pitch))
        p.pedals.sort(key=lambda v: v.time)
    return Piece(list(merged.values()), tempos, meters, metadata={"source_midi": Path(path).name})
