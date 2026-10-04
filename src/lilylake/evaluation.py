"""One-to-one matching, independent instruments, explicit zero-support metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "support": int(tp + fn),
    }


def frame_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError("Frame shapes must match")
    return _metrics(
        np.sum(truth & prediction), np.sum(~truth & prediction), np.sum(truth & ~prediction)
    )


def _match(actual, predicted, onset_tolerance, require_offset):
    if not actual or not predicted:
        return []
    cost = np.full((len(actual), len(predicted)), 1e6)
    for i, a in enumerate(actual):
        for j, b in enumerate(predicted):
            onset_error = abs(a.onset - b.onset)
            offset_ok = abs(a.offset - b.offset) <= max(0.05, 0.2 * (a.offset - a.onset))
            if (
                a.pitch == b.pitch
                and onset_error <= onset_tolerance
                and (offset_ok or not require_offset)
            ):
                cost[i, j] = onset_error + (
                    abs(a.offset - b.offset) * 0.01 if require_offset else 0
                )
    # Large invalid cost maximizes cardinality before minimizing timing distance.
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e6]


def evaluate_notes(truth, prediction, onset_tolerance=0.05):
    instruments = sorted({p.instrument for p in truth.parts + prediction.parts})
    per = {}
    totals = {k: [0, 0, 0] for k in ["onset", "note_with_offset"]}
    on_errors = []
    off_errors = []
    vel_errors = []
    for name in instruments:
        a = [n for p in truth.parts if p.instrument == name for n in p.notes]
        b = [n for p in prediction.parts if p.instrument == name for n in p.notes]
        stats = {}
        for key, offset in [("onset", False), ("note_with_offset", True)]:
            matches = _match(a, b, onset_tolerance, offset)
            tp = len(matches)
            fp = len(b) - tp
            fn = len(a) - tp
            stats[key] = _metrics(tp, fp, fn)
            totals[key] = [x + y for x, y in zip(totals[key], [tp, fp, fn])]
            if not offset:
                for i, j in matches:
                    on_errors.append(abs(a[i].onset - b[j].onset))
                    off_errors.append(abs(a[i].offset - b[j].offset))
                    vel_errors.append(a[i].velocity - b[j].velocity)
        per[name] = stats
    return {
        **{k: _metrics(*v) for k, v in totals.items()},
        "per_instrument": per,
        "onset_mae_ms": float(np.mean(on_errors) * 1000) if on_errors else None,
        "offset_mae_ms": float(np.mean(off_errors) * 1000) if off_errors else None,
        "velocity_mae": float(np.mean(np.abs(vel_errors))) if vel_errors else None,
        "velocity_rmse": float(np.sqrt(np.mean(np.square(vel_errors)))) if vel_errors else None,
        "matched_notes": len(on_errors),
        "missing_notes": totals["onset"][2],
        "spurious_notes": totals["onset"][1],
        "tolerances": {
            "onset_ms": onset_tolerance * 1000,
            "offset_ms_min": 50,
            "offset_duration_fraction": 0.2,
        },
    }
