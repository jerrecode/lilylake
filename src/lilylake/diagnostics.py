"""Exact standalone SVG piano-roll diagnostics, with no graphics dependency."""

from html import escape
from pathlib import Path


def piano_roll_svg(piece, path, width=1000):
    height = 80 + len(piece.parts) * 180
    duration = max(piece.duration, 1)
    colors = ["#3c83f6", "#ed9241", "#51ab72", "#a75fee", "#e8588c"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        f'<text x="20" y="28" fill="white" font-family="sans-serif" font-size="20">{escape(piece.title)}</text>',
    ]
    for i, part in enumerate(piece.parts):
        top = 65 + i * 180
        color = colors[i % len(colors)]
        svg.append(
            f'<text x="15" y="{top}" fill="{color}" font-family="sans-serif">{escape(part.instrument)}</text>'
        )
        for pitch in range(21, 109, 12):
            y = top + 150 - (pitch - 21) * 1.5
            svg.append(
                f'<line x1="75" y1="{y}" x2="{width - 20}" y2="{y}" stroke="#374151"/><text x="20" y="{y + 4}" fill="#9ca3af" font-size="10">MIDI {pitch}</text>'
            )
        for n in part.notes:
            x = 75 + n.onset / duration * (width - 95)
            w = max(1, (n.offset - n.onset) / duration * (width - 95))
            y = top + 150 - (n.pitch - 21) * 1.5
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="2" fill="{color}"><title>{escape(part.instrument)} pitch {n.pitch}, {n.onset:.3f} to {n.offset:.3f} seconds, velocity {n.velocity}</title></rect>'
            )
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg) + "\n")
