\version "2.24.3"
\header { title = "LilyLake transcription" tagline = ##f }
\score {
  <<
    \new PianoStaff << \new Staff \with { instrumentName = "Grand Piano" midiInstrument = "acoustic grand" } << { \clef "treble" } { \tempo 4 = 120 \time 4/4 \key fis \major s4*39/8 } \new Voice { \absolute { r4*13/8 cis'16.\mf ees'8\f r16 fis'16\f r4*5/6 bes'4*1/6\mf r8.. } } \new Voice { \absolute { r4*9/4 f''16\mf r4*4/3 bes''32\f r4*1/24 r8.. } } \new Voice { \absolute { r4*19/8 f'8\mf r4 f'''32\f r8.. } } >>\new Staff \with { instrumentName = "" midiInstrument = "acoustic grand" } << { \clef "bass" } { \tempo 4 = 120 \time 4/4 \key fis \major s4*39/8 } \new Voice { \absolute { fis32\f fis16.\f r4*8/3 fis4*11/24\mf aes16.\f ~ aes32 aes32\mf r4*5/8 } } \new Voice { \absolute { r16. aes16.\f bes16.\mf r4*1/24 b4*1/3\f b4*1/6\f r4*7/3 bes16.\mf b4*11/24\f r4*1/24 } } >> >>
    \new Staff \with { instrumentName = "Violin" midiInstrument = "violin" } << { \clef "treble" } { \tempo 4 = 120 \time 4/4 \key fis \major s4*39/8 } \new Voice { \absolute { r4*1/6 b'4*1/3\f r32 bes'4*5/24\f r4*1/6 aes'4*1/3\f r8 fis''8\f f'4*7/24\mf ees'32\f r4*5/8 b'8\f r32 bes16\f r16. fis'16\f } } \new Voice { \absolute { r8. bes16\f r8 cis'4*1/6\f r4*5/24 fis'4*7/24\f f''4*5/24\f r16 ees''16.\f cis''8\f r16 bes'16\f ~ bes'32 r4*1/24 aes'4*2/3\f r4*1/24 } } >>
  >>
  \layout { }
  \midi { }
}
