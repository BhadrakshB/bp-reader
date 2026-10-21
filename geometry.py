# geometry.py
"""Decides WHICH number is SYS, DIA and PUL.

Idea: almost every BP monitor stacks the three readings in a column
(SYS on top, DIA in the middle, PUL at the bottom). We look at every
combination of 3 candidate numbers, reject impossible ones, score the rest
on how "reading-like" they look, and keep the best. Printed labels are only
used to confirm or adjust the order, never as the main method.
"""

import math
from itertools import combinations

import numpy as np

# Plausible ranges for each field (hard limits).
RANGES = {'SYS': (70, 260), 'DIA': (30, 150), 'PUL': (30, 220)}
DEFAULT_ORDER = ['SYS', 'DIA', 'PUL']  # top-to-bottom order on most monitors


def calculate_median_slope(ocr_boxes):
    """Median tilt (dy/dx) of the text lines, from 4-point OCR polygons."""
    slopes = []
    for box in ocr_boxes:
        if len(box) < 2:
            continue
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        if abs(dx) > 5 and abs(dy / dx) < 1.0:
            slopes.append(dy / dx)
    return float(np.median(slopes)) if slopes else 0.0


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _h(b):
    return b[3] - b[1]


def _w(b):
    return b[2] - b[0]


def _dims(c):
    """(height, width) of a candidate; uses the rotated 4-point polygon when
    available so tilted rows don't inflate their axis-aligned box."""
    p = c.get('poly')
    if p and len(p) == 4:
        h = (math.dist(p[0], p[3]) + math.dist(p[1], p[2])) / 2
        w = (math.dist(p[0], p[1]) + math.dist(p[3], p[2])) / 2
        return h, w
    return _h(c['box']), _w(c['box'])


def _to_tilted(cx, cy, c, s):
    """Rotate a point into the device's own frame.
    u = along the text direction, v = 'down' on the device."""
    return cx * c + cy * s, -cx * s + cy * c


def filter_reading_candidates(cands, max_keep=8):
    """Keep only the BIG numbers. The real readings are the largest digits on
    the screen; bezel scales, dates and memory numbers are smaller."""
    if not cands:
        return []
    heights = sorted((_dims(c)[0] for c in cands), reverse=True)
    ref = heights[min(2, len(heights) - 1)]  # 3rd tallest ignores one giant outlier
    kept = [c for c in cands if _dims(c)[0] >= 0.5 * ref]
    kept.sort(key=lambda c: _dims(c)[0] * (0.5 + c['conf']), reverse=True)
    return kept[:max_keep]


def _layout_from_labels(labels, c, s):
    """Use printed labels (if found) to learn the layout.
    Returns (axis, order, conflict). axis 'v' = stacked, 'u' = side by side."""
    if len(labels) < 2:
        return 'v', DEFAULT_ORDER, False
    pos = {f: _to_tilted(*_center(b), c, s) for f, b in labels.items()}
    u_spread = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values())
    v_spread = max(p[1] for p in pos.values()) - min(p[1] for p in pos.values())
    axis = 'v' if v_spread >= u_spread else 'u'
    key = 1 if axis == 'v' else 0
    ordered = sorted(labels, key=lambda f: pos[f][key])
    if len(labels) == 3:
        return axis, ordered, False
    # Only 2 labels: keep the default order, but note if they disagree with it.
    default_rank = [f for f in DEFAULT_ORDER if f in labels]
    return axis, DEFAULT_ORDER, ordered != default_rank


def _score_triple(trip, axis, order, c, s, img_h):
    key = 1 if axis == 'v' else 0
    cross = 1 - key
    pos = [_to_tilted(*_center(t['box']), c, s) for t in trip]
    idx = sorted(range(3), key=lambda i: pos[i][key])
    ordered = [trip[i] for i in idx]
    opos = [pos[i] for i in idx]
    assign = dict(zip(order, ordered))

    # Hard rules
    for field, n in assign.items():
        lo, hi = RANGES[field]
        if not lo <= n['value'] <= hi:
            return None
    if assign['SYS']['value'] <= assign['DIA']['value']:
        return None

    # The three numbers must sit on clearly separate rows (or columns).
    extents = [(_dims(n)[0] if axis == 'v' else _dims(n)[1]) for n in ordered]
    mean_ext = float(np.mean(extents))
    gaps = [opos[i + 1][key] - opos[i][key] for i in range(2)]
    if min(gaps) < 0.6 * mean_ext:
        return None

    heights = [_dims(n)[0] for n in ordered]
    widths = [_dims(n)[1] for n in ordered]
    conf_term = float(np.mean([0.5 * n['conf'] + 0.5 * n['agreement'] for n in ordered]))
    size_term = min(1.0, float(np.mean(heights)) / (0.1 * img_h))
    sim_term = min(heights) / max(heights)
    cross_ref = float(np.mean(widths if axis == 'v' else heights))
    spread = max(p[cross] for p in opos) - min(p[cross] for p in opos)
    align_term = max(0.0, 1 - spread / (2 * cross_ref))
    pp = assign['SYS']['value'] - assign['DIA']['value']
    pp_term = 1.0 if 15 <= pp <= 110 else 0.3

    score = (2.0 * conf_term + 1.0 * size_term + 0.7 * sim_term
             + 0.8 * align_term + 0.5 * pp_term)
    return score, assign


def assign_readings(cands, labels, slope, img_h):
    """Pick the best SYS/DIA/PUL assignment.
    Returns {'readings': {field: candidate}, 'needs_review': bool, 'reasons': [...]}"""
    theta = math.atan(slope)
    c, s = math.cos(theta), math.sin(theta)
    axis, order, conflict = _layout_from_labels(labels, c, s)

    pool = filter_reading_candidates(cands)
    scored = []
    for trip in combinations(pool, 3):
        r = _score_triple(trip, axis, order, c, s, img_h)
        if r:
            scored.append(r)

    if not scored:
        return {'readings': {}, 'needs_review': True,
                'reasons': ['no valid SYS/DIA/PUL combination found']}

    scored.sort(key=lambda r: -r[0])
    best_score, best = scored[0]
    best_vals = {f: n['value'] for f, n in best.items()}

    reasons = []
    for sc, other in scored[1:]:
        if {f: n['value'] for f, n in other.items()} != best_vals:
            if best_score - sc < 0.15:
                reasons.append('another assignment scored almost the same')
            break
    if any(n['agreement'] < 0.5 for n in best.values()):
        reasons.append('readers disagreed on a number')
    if any(n['agreement'] < 0.6 for n in best.values()):
        reasons.append('readings differ between photometric variants')
    if any(n['conf'] < 0.5 for n in best.values()):
        reasons.append('low digit confidence')
    if any(n.get('glare_frac', 0.0) > 0.03 for n in best.values()):
        reasons.append('glare on display')
    if conflict:
        reasons.append('printed label order disagrees with layout')
    pp = best['SYS']['value'] - best['DIA']['value']
    if not 20 <= pp <= 100:
        reasons.append('unusual SYS-DIA gap')

    return {'readings': best, 'needs_review': bool(reasons), 'reasons': reasons}
