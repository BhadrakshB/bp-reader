# test_robustness.py
"""Unit tests for the robustness changes. Runs with plain python or pytest:
    venv/bin/python test_robustness.py
"""

import cv2
import numpy as np

import seven_segment as ss

# Segment endpoints in a 4-wide x 7-tall unit grid (x, y).
_SEG = {
    'a': ((1, 0), (3, 0)), 'b': ((3, 0), (3, 3)), 'c': ((3, 3), (3, 6)),
    'd': ((1, 6), (3, 6)), 'e': ((0, 3), (0, 6)), 'f': ((0, 0), (0, 3)),
    'g': ((1, 3), (3, 3)),
}
_PATTERNS = {d: pat for pat, d in ss.DIGITS.items() if d != 7 or pat == (1, 1, 1, 0, 0, 0, 0)}


def draw_digit(digit, scale=12, thick=11, slant=0.0, ink=30, bg=200):
    """Render a seven-segment digit as a gray image (dark ink on bright bg)."""
    pat = _PATTERNS[digit]
    h, w = 7 * scale, 4 * scale + int(abs(slant) * 7 * scale) + scale
    img = np.full((h, w), bg, np.uint8)
    xoff = int(abs(slant) * 7 * scale) if slant > 0 else 0
    for name, on in zip('abcdefg', pat):
        if not on:
            continue
        (x1, y1), (x2, y2) = _SEG[name]
        p1 = (int(x1 * scale + slant * y1 * scale) + xoff, y1 * scale)
        p2 = (int(x2 * scale + slant * y2 * scale) + xoff, y2 * scale)
        cv2.line(img, p1, p2, int(ink), thick)
    return img


def draw_number(text, **kw):
    parts = [draw_digit(int(ch), **kw) for ch in text]
    gap = np.full((parts[0].shape[0], 8), kw.get('bg', 200), np.uint8)
    out = parts[0]
    for p in parts[1:]:
        out = np.hstack([out, gap, p])
    return out


# --------------------------------------------------------------- dark_mask
def test_dark_mask_dim_image_still_finds_segments():
    img = draw_number('105', ink=40, bg=75)  # very dim display
    m = ss.dark_mask(img)
    assert m[img < 60].mean() > 0.5, 'dim segments missing from mask'


def test_dark_mask_glare_creates_no_false_segments():
    img = draw_number('105', ink=30, bg=200)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - 0.85 * w) ** 2 + (yy - 0.2 * h) ** 2) / (2 * (0.1 * w) ** 2)))
    glared = np.clip(img.astype(np.float32) + 255 * blob, 0, 255).astype(np.uint8)
    m = ss.dark_mask(glared)
    blob_core = blob > 0.5
    assert m[blob_core].mean() < 0.05, 'glare blob produced false segments'
    digit_region = m[:, :int(0.5 * w)]
    assert digit_region.mean() > 0.02, 'real segments lost under glare'


def test_dark_mask_output_shape_and_dtype():
    img = draw_number('88')
    m = ss.dark_mask(img)
    assert m.shape == img.shape and m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 1}


def test_glare_fraction():
    img = np.full((100, 100), 128, np.uint8)
    img[:10, :10] = 255
    assert abs(ss.glare_fraction(img) - 0.01) < 1e-6


# --------------------------------------------------------------- deskew
def test_deskew_recovers_known_angle():
    img = draw_number('138', slant=0.0)
    h, w = img.shape
    pad = np.full((h + 80, w + 80), 200, np.uint8)
    pad[40:40 + h, 40:40 + w] = img
    for angle in (6.0, -9.0):
        M0 = np.eye(3, dtype=np.float64)
        rot = cv2.getRotationMatrix2D((pad.shape[1] / 2, pad.shape[0] / 2), angle, 1.0)
        tilted = cv2.warpAffine(pad, rot, (pad.shape[1], pad.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
        fixed, M1, found_angle = ss._deskew(tilted, M0)
        assert abs(found_angle + angle) < 1.5, f'{angle} -> {found_angle}'
        # centroid of the deskewed mask must map back onto the ink in the
        # original image (compare centroids; the centroid itself can sit in a
        # gap between segments)
        m = ss.dark_mask(fixed)
        ys, xs = np.nonzero(m)
        back = cv2.perspectiveTransform(
            np.float32([[[xs.mean(), ys.mean()]]]), np.linalg.inv(M1))[0, 0]
        mt = ss.dark_mask(tilted)
        yt, xt = np.nonzero(mt)
        dist = np.hypot(back[0] - xt.mean(), back[1] - yt.mean())
        assert dist < 3.0, f'mapped centroid {dist:.1f}px off the ink'


# ----------------------------------------------------------- vote merging
def test_vote_merge_majority_wins():
    rows_per_variant = [
        [{'text': '105', 'conf': 0.8, 'h': 100, 'box': [0, 0, 100, 100]},
         {'text': '66', 'conf': 0.7, 'h': 90, 'box': [0, 150, 90, 240]}],
        [{'text': '105', 'conf': 0.9, 'h': 100, 'box': [1, 1, 101, 101]},
         {'text': '66', 'conf': 0.6, 'h': 90, 'box': [0, 151, 90, 241]}],
        [{'text': '185', 'conf': 0.95, 'h': 100, 'box': [0, 0, 100, 100]},
         {'text': '66', 'conf': 0.8, 'h': 90, 'box': [0, 150, 90, 240]}],
    ]
    merged = ss._vote_merge(rows_per_variant, n_variants=3)
    by_text = sorted(r['text'] for r in merged)
    assert by_text == ['105', '66']
    r105 = next(r for r in merged if r['text'] == '105')
    assert abs(r105['agreement'] - 2 / 3) < 1e-6
    assert abs(r105['conf'] - 0.85) < 1e-6  # mean of the agreeing members


# ------------------------------------------------------------- n-best read
def test_read_digit_nbest_all_digits():
    for d in range(10):
        img = draw_number(str(d) * 3, slant=0.15)
        m = ss.dark_mask(img)
        sh = ss._shear(m, ss._best_shear(m))     # mirror the pipeline: shear the
        cols = sh.sum(axis=0) > 0.05 * sh.shape[0]   # row, split columns, crop
        pieces, c0 = [], None
        for cx in range(len(cols) + 1):
            on = cx < len(cols) and cols[cx]
            if on and c0 is None:
                c0 = cx
            if not on and c0 is not None:
                piece, _ = ss._tight(sh[:, c0:cx])
                if piece.size:
                    pieces.append(piece)
                c0 = None
        assert len(pieces) == 3, f'digit {d}: got {len(pieces)} pieces'
        for piece in pieces:
            nbest = ss._read_digit_nbest(piece)
            assert len(nbest) == 3
            assert nbest[0][0] == d, f'digit {d} read as {nbest[0][0]}'
            assert ss._read_digit(piece)[0] == d   # wrapper agrees


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL {fn.__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    raise SystemExit(1 if failed else 0)
