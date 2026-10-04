"""Conservative event-derived tempo, meter and key hypotheses, with alternatives.

Onset autocorrelation follows the broad tempo-estimation idea described by
Ellis (2007), not the full dynamic-programming beat tracker. These are heuristics,
not a trained score model or calibrated probabilities.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, find_peaks


def infer_notation(piece, tempo=None):
    notes = sorted([n for p in piece.parts for n in p.notes], key=lambda n: n.onset)
    report = {
        "tempo": {"bpm": 120.0, "source": "fallback", "confidence": 0.0, "alternatives": []},
        "meter": {"numerator": 4, "denominator": 4, "source": "fallback", "confidence": 0.0},
        "key": {"tonic": None, "mode": None, "confidence": 0.0, "source": "unknown"},
        "confidence_calibrated": False,
    }
    if tempo is not None:
        if not np.isfinite(tempo) or tempo <= 0:
            raise ValueError("Tempo must be finite and positive")
        report["tempo"].update({"bpm": float(tempo), "source": "user", "confidence": 1.0})
    elif len(notes) >= 8 and piece.duration >= 2:
        dt = 0.01
        end = min(piece.duration, 120.0)
        envelope = np.zeros(int(end / dt) + 2)
        for n in notes:
            if n.onset <= end:
                envelope[round(n.onset / dt)] += n.velocity / 127
        envelope = gaussian_filter1d(envelope, 1.5)
        correlation = fftconvolve(envelope, envelope[::-1], mode="full")[len(envelope) - 1 :]
        correlation /= max(correlation[0], 1e-9)
        lo, hi = round(60 / 240 / dt), round(60 / 40 / dt)
        lags = np.arange(lo, min(hi, len(correlation)))
        bpms = 60 / (lags * dt)
        # A weak 120 BPM prior resolves common half/double-time ambiguity; retain alternatives.
        scores = correlation[lags] * np.exp(-0.35 * np.log2(bpms / 120) ** 2)
        peaks, _ = find_peaks(scores)
        candidates = peaks if len(peaks) else np.array([int(np.argmax(scores))])
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        alternatives = [{"bpm": float(bpms[i]), "score": float(scores[i])} for i in ranked[:5]]
        best = ranked[0]
        report["tempo"].update(
            {
                "bpm": float(bpms[best]),
                "source": "event_onset_autocorrelation",
                "confidence": float(np.clip(scores[best], 0, 0.8)),
                "alternatives": alternatives,
                "ambiguity": "half/double-time and subdivision ambiguity",
            }
        )
    if len(notes) >= 12:
        bpm = report["tempo"]["bpm"]
        beats = np.array([n.onset * bpm / 60 for n in notes])
        velocities = np.array([n.velocity for n in notes], dtype=float)
        integer = np.round(beats).astype(int)
        aligned = np.abs(beats - integer) < 0.12
        meter_scores = []
        for size in [3, 4]:
            strengths = np.array(
                [
                    velocities[aligned & (integer % size == phase)].mean()
                    if np.any(aligned & (integer % size == phase))
                    else 0
                    for phase in range(size)
                ]
            )
            mean = strengths.mean()
            contrast = (strengths.max() - mean) / max(mean, 1)
            meter_scores.append((contrast, size, int(np.argmax(strengths))))
        contrast, size, phase = max(meter_scores)
        if contrast > 0.12:
            report["meter"].update(
                {
                    "numerator": size,
                    "source": "event_accent_periodicity",
                    "confidence": float(min(0.5, contrast)),
                    "downbeat_phase_beats": phase,
                    "ambiguity": "accent pattern need not equal notated meter",
                }
            )
    if len(notes) >= 8:
        chroma = np.zeros(12)
        for n in notes:
            chroma[n.pitch % 12] += (n.offset - n.onset) * n.velocity / 127
        if chroma.sum() > 0:
            chroma /= chroma.sum()
            candidates = []
            for tonic in range(12):
                for mode, scale in [
                    ("major", [0, 2, 4, 5, 7, 9, 11]),
                    ("minor", [0, 2, 3, 5, 7, 8, 10]),
                ]:
                    coverage = sum(chroma[(tonic + x) % 12] for x in scale)
                    score = (
                        coverage
                        + 0.15 * chroma[tonic]
                        + (0.03 if notes[-1].pitch % 12 == tonic else 0)
                    )
                    candidates.append((score, coverage, tonic, mode))
            candidates.sort(reverse=True)
            best = candidates[0]
            gap = best[0] - candidates[1][0]
            if best[1] > 0.85 and gap > 0.005:
                tonic = ["c", "cis", "d", "ees", "e", "f", "fis", "g", "aes", "a", "bes", "b"][
                    best[2]
                ]
                report["key"].update(
                    {
                        "tonic": tonic,
                        "mode": best[3],
                        "confidence": float(min(0.6, gap * 5)),
                        "source": "duration_weighted_diatonic_fit",
                        "ambiguity": "relative modes, modulation and enharmonic spelling",
                    }
                )
    return report
