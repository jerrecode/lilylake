"""General MIDI taxonomy. Names describe render programs, not proven classifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    program: int
    midi_name: str
    low: int = 0
    high: int = 127
    clef: str = "treble"
    keyboard: bool = False
    percussion: bool = False


INSTRUMENTS = {
    "grand_piano": Instrument(0, "acoustic grand", 21, 108, keyboard=True),
    "piano": Instrument(0, "acoustic grand", 21, 108, keyboard=True),
    "upright_piano": Instrument(1, "bright acoustic", 21, 108, keyboard=True),
    "electric_piano": Instrument(4, "electric piano 1", 21, 108, keyboard=True),
    "violin": Instrument(40, "violin", 55, 103),
    "viola": Instrument(41, "viola", 48, 91, "alto"),
    "cello": Instrument(42, "cello", 36, 84, "bass"),
    "double_bass": Instrument(43, "contrabass", 28, 67, "bass"),
    "acoustic_guitar": Instrument(25, "acoustic guitar (steel)", 40, 88),
    "classical_guitar": Instrument(24, "acoustic guitar (nylon)", 40, 88),
    "electric_guitar": Instrument(26, "electric guitar (jazz)", 40, 96),
    "harp": Instrument(46, "orchestral harp", 24, 103),
    "flute": Instrument(73, "flute", 60, 96),
    "piccolo": Instrument(72, "piccolo", 74, 108),
    "clarinet": Instrument(71, "clarinet", 50, 94),
    "bass_clarinet": Instrument(71, "clarinet", 34, 82, "bass"),
    "oboe": Instrument(68, "oboe", 58, 91),
    "bassoon": Instrument(70, "bassoon", 34, 79, "bass"),
    "soprano_sax": Instrument(64, "soprano sax", 56, 88),
    "alto_sax": Instrument(65, "alto sax", 49, 81),
    "tenor_sax": Instrument(66, "tenor sax", 44, 76),
    "baritone_sax": Instrument(67, "baritone sax", 37, 69, "bass"),
    "trumpet": Instrument(56, "trumpet", 54, 94),
    "trombone": Instrument(57, "trombone", 40, 82, "bass"),
    "french_horn": Instrument(60, "french horn", 34, 89),
    "tuba": Instrument(58, "tuba", 28, 65, "bass"),
    "drum_kit": Instrument(0, "", 35, 81, "percussion", percussion=True),
}
PROGRAM_NAMES = {}
for name, item in INSTRUMENTS.items():
    PROGRAM_NAMES.setdefault(item.program, name)
# Prefer a stable program canonical name where GM has no unique upright/bass clarinet identity.
PROGRAM_NAMES[0] = "grand_piano"
PROGRAM_NAMES[71] = "clarinet"
# General MIDI percussion 35..81; standard LilyPond drum names.
DRUM_NAMES = dict(
    enumerate(
        [
            "bda",
            "bd",
            "ss",
            "sna",
            "hc",
            "sne",
            "tomfl",
            "hhc",
            "tomfh",
            "hhp",
            "toml",
            "hho",
            "tomml",
            "tommh",
            "cymca",
            "tomh",
            "cymra",
            "cymch",
            "rb",
            "tamb",
            "cyms",
            "cb",
            "cymcb",
            "vibs",
            "cymrb",
            "boh",
            "bol",
            "cghm",
            "cgho",
            "cglo",
            "timh",
            "timl",
            "agh",
            "agl",
            "cab",
            "mar",
            "whs",
            "whl",
            "guis",
            "guil",
            "cl",
            "wbh",
            "wbl",
            "cuim",
            "cuio",
            "trim",
            "trio",
        ],
        start=35,
    )
)
