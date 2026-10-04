\version "2.24.3"
\header { title = "LilyLake scale seed 12" tagline = ##f }
\score {
  <<
    \new PianoStaff << \new Staff \with { instrumentName = "Grand Piano" midiInstrument = "acoustic grand" } << { \clef "treble" } { \tempo 4 = 150 \time 4/4 \key c \major s4*49/8 } \new Voice { \absolute { r4 c''''4\mf cis'16.\mp r32 ees'8.\f r8. r4*17/8 } } \new Voice { \absolute { r2. f'8\mp fis'16\ff-. r16 r4*17/8 } } >>\new Staff \with { instrumentName = "" midiInstrument = "acoustic grand" } << { \clef "bass" } { \tempo 4 = 150 \time 4/4 \key c \major s4*49/8 } \new Voice { \absolute { a,,4\mf r2. r4*17/8 } } \new Voice { \absolute { fis8.\ff-. r4*13/4 fis16.\f r32 aes8.\p r8.. } } \new Voice { \absolute { r8 aes16\mp r16 bes16\mp r16 b8\mp r2 r4 bes16.\mf r32 b16.\mp r16 } } >> >>
    \new Staff \with { instrumentName = "Violin" midiInstrument = "violin" } << { \clef "treble" } { \tempo 4 = 150 \time 4/4 \key c \major s4*49/8 } \new Voice { \absolute { r16 bes8\mp-. b16.\mf r32 cis'16.\mf r32 ees'8.\mf r8. g8\mp aes16\f-. ~ aes16 bes8\ff b16.\ff r32 cis'8.\ff r32 } } \new Voice { \absolute { r4*9/4 f'8\ff fis'16\mp r4 r4.. ees'16.\p } } >>
  >>
  \layout { }
  \midi { }
}
