# seven_segment.py
"""Reads the big LCD seven-segment digits with plain OpenCV (no OCR model).

Steps: find several likely LCD screens -> straighten each one -> mask the dark
segments -> glue segments into digits -> group digits into rows -> remove the
italic slant -> check which of the 7 bars of every digit are lit.  Several
screens and several sensitivity levels are read and compared; disagreement is
reported as a low 'agreement' so the caller can ask for a manual check.
"""

import cv2
import numpy as np

# Feature toggles for ablation. Every robustness change can be switched off here.
CFG = {
    'use_glare_mask': True,     # exclude saturated pixels from background estimate
    'use_inpaint': False,       # inpaint glare before masking (A/B: can hurt when
                                # glare covers a whole digit)
    'mask_method': 'contrast',  # 'contrast' = log relative contrast, 'blackhat'
    'use_tta': True,            # photometric test-time augmentation + voting
    'tta_channels': True,       # TTA: also try the B, G, R channels
    'use_deskew': True,         # in-plane rotation search on the warped LCD
    'warp_long_side': 800,      # the straightened LCD is resized to about this
    'whole_image_fallback': True,  # read the whole photo when no screen qualifies
}

# Lit bars, in order: a=top, b=top-right, c=bottom-right, d=bottom,
# e=bottom-left, f=top-left, g=middle.  (1,1,1,0,0,1,0) is the 4-stroke "7"
# some monitors draw.
DIGITS = {
    (1, 1, 1, 1, 1, 1, 0): 0, (0, 1, 1, 0, 0, 0, 0): 1, (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3, (0, 1, 1, 0, 0, 1, 1): 4, (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6, (1, 1, 1, 0, 0, 0, 0): 7, (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8, (1, 1, 1, 1, 0, 1, 1): 9,
}
WARP_LONG_SIDE = 600   # legacy default; CFG['warp_long_side'] is the live value
MAX_CANDIDATES = 10    # how many possible screens are read and compared


# ----------------------------------------------------------------- masking
def _glare_mask(gray):
    """1 where the pixel is a saturated specular highlight (dilated a little)."""
    g = (gray >= 245).astype(np.uint8) * 255
    return cv2.dilate(g, np.ones((7, 7), np.uint8)) > 0


def glare_fraction(gray):
    return float((gray >= 245).mean())


def contrast_map(gray):
    """Relative log-darkness vs the local glare-free background (0 on glare).
    This is what dark_mask thresholds; it is also used directly for gray-level
    zone scoring of digits."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    glare = _glare_mask(gray) if CFG['use_glare_mask'] else np.zeros(gray.shape, bool)
    sigma = max(8, min(gray.shape) / 8)
    valid = (~glare).astype(np.float32)
    num = cv2.resize(g.astype(np.float32) * valid, None, fx=0.25, fy=0.25,
                     interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(num, (0, 0), sigma / 4)
    den = cv2.GaussianBlur(den, (0, 0), sigma / 4)
    bg = cv2.resize(num / np.maximum(den, 1e-3), (g.shape[1], g.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    d = np.log(bg + 8.0) - np.log(g.astype(np.float32) + 8.0)
    np.clip(d, 0.0, None, out=d)
    d[glare] = 0.0
    return d


def dark_mask(gray, frac=0.35):
    """1 where a pixel is clearly darker than its surroundings (the lit segments).

    Relative (log) contrast against a local background that ignores glare, so
    dim photos and photos with specular highlights both still yield segments."""
    if CFG['use_inpaint'] and glare_fraction(gray) > 0:
        gm = ((gray >= 245).astype(np.uint8) * 255)
        gray = cv2.inpaint(gray, gm, 5, cv2.INPAINT_TELEA)
    if CFG['mask_method'] == 'blackhat':
        return _dark_mask_blackhat(gray, frac)

    d = contrast_map(gray)
    if CFG['use_glare_mask']:
        glare = _glare_mask(gray)
        ref = d[~glare] if glare.any() else d.ravel()
    else:
        ref = d.ravel()
    if ref.size == 0:               # the whole crop is saturated
        ref = d.ravel()
    thr = max(0.03, frac * float(np.percentile(ref, 99.5)))
    m = (d > thr).astype(np.uint8) * 255
    return (cv2.medianBlur(m, 5) > 0).astype(np.uint8)


def _dark_mask_blackhat(gray, frac=0.35):
    """Alternative segment mask: black-hat transform with an elliptical kernel
    sized relative to the crop. Kept as a fallback variant for voting."""
    h, w = gray.shape
    k = max(3, int(0.08 * h)) | 1
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    glare = _glare_mask(gray) if CFG['use_glare_mask'] else np.zeros(gray.shape, bool)
    bh = bh.astype(np.float32)
    bh[glare] = 0.0
    ref = bh[~glare] if glare.any() else bh.ravel()
    if ref.size == 0:               # the whole crop is saturated
        ref = bh.ravel()
    thr = max(8.0, frac * float(np.percentile(ref, 99.5)))
    m = (bh > thr).astype(np.uint8) * 255
    return (cv2.medianBlur(m, 5) > 0).astype(np.uint8)


def _digit_score(gray):
    """Rough count of tall digit-sized blobs inside a candidate screen (for sorting)."""
    m = dark_mask(gray)
    h, w = m.shape
    ih, iw = int(.06 * h), int(.06 * w)
    m[:ih] = 0; m[-ih:] = 0; m[:, :iw] = 0; m[:, -iw:] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return sum(1 for i in range(1, n) if st[i, 3] > 0.04 * h and st[i, 4] > 0.003 * h * w)


# --------------------------------------------------------------- LCD finder
def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl,tr,br,bl


def _grow(quad, pad):
    """Make the quad bigger by `pad` (0.03 = 3 %) on every side."""
    c = quad.mean(axis=0)
    return (c + (quad - c) * (1 + 2 * pad)).astype(np.float32)


def _warp(img, quad):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    k = CFG['warp_long_side'] / max(w, h)
    w, h = max(20, int(w * k)), max(20, int(h * k))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h)), M


def find_lcd_candidates(img, keep=MAX_CANDIDATES, label_boxes=None):
    """Up to `keep` likely screens as [(straightened gray LCD, matrix), ...].
    Contours are searched on several tone curves (raw, CLAHE, gamma-brightened)
    and with auto-Canny thresholds. `label_boxes` (optional, OCR label rects in
    original-image pixels) adds a candidate crop around the printed labels."""
    H, W = img.shape[:2]
    s = 1000 / max(H, W)
    small = cv2.resize(img, None, fx=s, fy=s)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    A = small.shape[0] * small.shape[1]
    quads, seen = [], set()

    def add(q):
        key = tuple((q.mean(axis=0) / 8).astype(int)) + (int(np.linalg.norm(q[0] - q[2]) / 8),)
        if key not in seen:
            seen.add(key)
            quads.append(q)

    lut = np.clip((np.arange(256) / 255.0) ** 0.6 * 255.0, 0, 255).astype(np.uint8)
    tones = [gray, cv2.createCLAHE(2.0, (8, 8)).apply(gray), cv2.LUT(gray, lut)]
    for base in tones:
        for blur in (5, 9):
            g = cv2.GaussianBlur(base, (blur, blur), 0)
            med = float(np.median(g))
            pairs = [(max(5.0, 0.66 * med), max(10.0, 1.33 * med)),
                     (40, 120), (20, 60), (60, 160)]
            for lo, hi in pairs:
                e = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((3, 3), np.uint8))
                cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in cs:
                    if not 0.01 * A < cv2.contourArea(c) < 0.4 * A:
                        continue
                    hull = cv2.convexHull(c)
                    # (a) steady choice: the tilted rectangle that best fits the outline
                    rect = cv2.minAreaRect(hull)
                    rw, rh = rect[1]
                    if rw > 0 and rh > 0 and cv2.contourArea(hull) / (rw * rh) > 0.8 \
                            and 0.4 < rw / rh < 2.5:
                        add(_order_quad(cv2.boxPoints(rect)))
                    # (b) perspective choice: a 4-corner outline
                    ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
                    if len(ap) == 4 and cv2.isContourConvex(ap):
                        add(_order_quad(ap))

    if label_boxes:
        # The numbers sit next to their printed labels; propose a generous crop
        # around the label region as one more candidate.
        x1 = min(b[0] for b in label_boxes); y1 = min(b[1] for b in label_boxes)
        x2 = max(b[2] for b in label_boxes); y2 = max(b[3] for b in label_boxes)
        cx, cy = (x1 + x2) / 2 * s, (y1 + y2) / 2 * s
        hw, hh = (x2 - x1) / 2 * s * 3.0, (y2 - y1) / 2 * s * 1.4
        add(np.float32([[cx - hw, cy - hh], [cx + hw, cy - hh],
                        [cx + hw, cy + hh], [cx - hw, cy + hh]]))

    scored = []
    for q in quads:
        for pad in (0.0, 0.03):
            warped, M = _warp(img, _grow(q / s, pad))
            wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            sc = _digit_score(wg)
            if sc >= 1:
                scored.append((sc, wg, M))
    for gate in (3, 2, 1):      # adaptive gate: prefer strict, relax if empty
        good = [t for t in scored if t[0] >= gate]
        if good:
            break
    good.sort(key=lambda t: -t[0])
    return [(g, M) for _, g, M in good[:keep]]


# ---------------------------------------------------------- digit reading
def _shear(mask, k, flags=cv2.INTER_NEAREST):
    h, w = mask.shape
    M = np.float32([[1, -k, k * h / 2.0], [0, 1, 0]])
    return cv2.warpAffine(mask, M, (w, h), flags=flags)


def _best_shear(mask):
    """The slant that makes the digits' columns most separate."""
    best, bk = -1.0, 0.0
    for k in np.arange(-0.35, 0.351, 0.025):
        col = _shear(mask, k).sum(axis=0).astype(np.float64)
        sc = float((col ** 2).sum())
        if sc > best:
            best, bk = sc, float(k)
    return bk


def _zone_fills(m, d=None):
    """How strongly each of the 7 bars is lit, as a 7-vector in a,b,c,d,e,f,g
    order. With a contrast map `d`, zones are scored by mean contrast and
    normalized by this digit's OWN strongest zone, which keeps ghost segments
    (faint unlit bars) below the truly lit ones."""
    h, w = m.shape
    zones = {
        'a': (.25, .75, 0, .17), 'g': (.3, .7, .41, .59), 'd': (.25, .75, .83, 1),
        'f': (0, .25, .1, .44), 'b': (.75, 1, .1, .44),
        'e': (0, .25, .56, .9), 'c': (.75, 1, .56, .9),
    }
    src = d if d is not None else m
    r = []
    for name in 'abcdefg':
        x1, x2, y1, y2 = zones[name]
        z = src[int(y1 * h):max(int(y2 * h), int(y1 * h) + 1),
                int(x1 * w):max(int(x2 * w), int(x1 * w) + 1)]
        r.append(float(z.mean()))
    if d is None:
        return [min(1.0, v / 0.7) for v in r]
    mx = max(r)
    return [v / mx for v in r] if mx > 1e-6 else [0.0] * 7


def _tight(piece, piece_d=None, margin=0.1):
    """Tight crop around the ink, with a small margin so that the bars at the
    edge (a, d) are not cut in half. Returns (mask_crop, contrast_crop)."""
    ys, xs = np.nonzero(piece)
    mh = int(margin * (ys.max() - ys.min() + 1)) + 1
    y1, y2 = max(0, ys.min() - mh), min(piece.shape[0], ys.max() + 1 + mh)
    x1, x2 = max(0, xs.min() - mh), min(piece.shape[1], xs.max() + 1 + mh)
    return piece[y1:y2, x1:x2], (piece_d[y1:y2, x1:x2] if piece_d is not None else None)


def _read_digit_nbest(m, d=None):
    """m = tight 0/1 crop of ONE digit, slant removed; d = optional contrast map
    of the same crop. Returns the 3 best [(digit, distance), ...], ascending."""
    h, w = m.shape
    if w < 0.28 * h:    # a sliver this narrow can only be a "1"; zone sampling
        return [(1, 0.0), (7, 1.0), (4, 1.0)]   # is unreliable at this width
    r = _zone_fills(m, d)
    dists = sorted((sum(abs(a - b) for a, b in zip(r, pat)), dg)
                   for pat, dg in DIGITS.items())
    best = {}
    for dist, dg in dists:          # dists ascending: first hit is the smallest
        if dg not in best:
            best[dg] = dist
    return sorted(((dg, dist) for dg, dist in best.items()), key=lambda t: t[1])[:3]


def _read_digit(m, d=None):
    """m = tight 0/1 crop of ONE digit, slant removed. Returns (digit, confidence)."""
    nb = _read_digit_nbest(m, d)
    if nb[0][1] == 0.0 and nb[0][0] == 1:
        return 1, 0.6    # narrow-sliver shortcut: a structural guess, not a read
    return nb[0][0], max(0.0, min(1.0, nb[1][1] - nb[0][1]))


def _deskew(gray, M):
    """Remove small in-plane rotation of the warped LCD.
    Searches the angle that makes the digit rows most horizontal (sharpest row
    separation), rotates the image, and folds the rotation into M so that
    inv(M_new) still maps boxes back to original-image coordinates.
    Returns (rotated gray, M_new, angle_applied)."""
    if not CFG['use_deskew']:
        return gray, M, 0.0
    m = dark_mask(gray)
    s = min(1.0, 300.0 / max(m.shape))     # search on a small copy for speed
    ms = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else m
    h, w = ms.shape

    def score(ang):
        R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        rs = cv2.warpAffine(ms, R, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1)
        return float((rs.astype(np.float64) ** 2).sum())

    angs = np.arange(-12.0, 12.01, 1.0)
    best = float(angs[int(np.argmax([score(a) for a in angs]))])
    fine = np.arange(best - 1.0, best + 1.01, 0.25)
    best = float(fine[int(np.argmax([score(a) for a in fine]))])
    if abs(best) < 0.25:
        return gray, M, 0.0
    h0, w0 = gray.shape
    R = cv2.getRotationMatrix2D((w0 / 2.0, h0 / 2.0), best, 1.0)
    fixed = cv2.warpAffine(gray, R, (w0, h0), borderMode=cv2.BORDER_REPLICATE)
    return fixed, np.vstack([R, [0.0, 0.0, 1.0]]) @ M, best


def _rows_from_lcd(gray, M, frac=0.35):
    """Read all digit rows on one straightened LCD picture.
    Each row: {'text','conf','h','box','touch'}, top to bottom, box in full-image pixels."""
    gray, M, _deskew_angle = _deskew(gray, M)
    h, w = gray.shape
    m = dark_mask(gray, frac)
    dmap = contrast_map(gray)
    ins = max(3, int(0.02 * min(w, h)))
    m[:ins] = 0; m[-ins:] = 0; m[:, :ins] = 0; m[:, -ins:] = 0

    # drop frame leftovers and side bar-graphs (they touch the border strip)
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n0):
        bx, by, bw, bh, _a = st0[i]
        touches_edge = bx <= ins or by <= ins or bx + bw >= w - ins or by + bh >= h - ins
        too_big = bw > 0.6 * w or bh > 0.7 * h   # a frame ring, never a digit
        if touches_edge or too_big:
            m[lab0 == i] = 0

    # glue the bars of one digit together
    kv, kh = max(3, int(0.03 * h)) | 1, max(3, int(0.012 * h)) | 1
    glued = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kh, kv)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(glued)
    glyphs = []
    for i in range(1, n):
        gx, gy, gw, gh, _a = st[i]
        if gh < 0.07 * h or gw > 1.1 * gh or gw < 0.10 * gh:
            continue
        glyphs.append({'x': gx, 'y': gy, 'w': gw, 'h': gh, 'id': i})

    # group glyphs into rows
    glyphs.sort(key=lambda g: g['y'] + g['h'] / 2)
    rows = []
    for g in glyphs:
        cy = g['y'] + g['h'] / 2
        for r in rows:
            if abs(cy - r['cy']) < 0.4 * max(g['h'], r['h']):
                r['g'].append(g)
                r['cy'] = float(np.mean([q['y'] + q['h'] / 2 for q in r['g']]))
                r['h'] = max(r['h'], g['h'])
                break
        else:
            rows.append({'g': [g], 'cy': cy, 'h': g['h']})

    Minv = np.linalg.inv(M)
    out = []
    for r in rows:
        gl = sorted((g for g in r['g'] if g['h'] >= 0.6 * r['h']), key=lambda g: g['x'])
        if not gl:
            continue
        # digits of one number sit close together; a far-away glyph (bar graph,
        # icon) is not part of it. Keep the widest cluster of neighbours.
        clusters, cur = [], [gl[0]]
        for a, b in zip(gl, gl[1:]):
            if b['x'] - (a['x'] + a['w']) > 0.9 * r['h']:
                clusters.append(cur); cur = []
            cur.append(b)
        clusters.append(cur)
        gl = max(clusters, key=lambda c: sum(g['w'] for g in c))

        y1 = min(g['y'] for g in gl); y2 = max(g['y'] + g['h'] for g in gl)
        x1 = min(g['x'] for g in gl); x2 = max(g['x'] + g['w'] for g in gl)
        pad = max(4, int(0.06 * (y2 - y1)))
        ry1, ry2, rx1, rx2 = max(0, y1 - pad), y2 + pad, max(0, x1 - 3 * pad), x2 + 3 * pad
        sub = np.zeros_like(m[ry1:ry2, rx1:rx2])
        for g in gl:
            sub |= ((lab == g['id'])[ry1:ry2, rx1:rx2] & (m[ry1:ry2, rx1:rx2] > 0)).astype(np.uint8)
        k = _best_shear(sub)
        sh = _shear(sub, k)
        shd = _shear(dmap[ry1:ry2, rx1:rx2] * sub, k, flags=cv2.INTER_LINEAR)

        cols = sh.sum(axis=0) > 0.03 * sh.shape[0]   # one noise pixel cannot
        digits, nbests, c0 = [], [], None            # bridge two digits
        for cx in range(len(cols) + 1):
            on = cx < len(cols) and cols[cx]
            if on and c0 is None:
                c0 = cx
            if not on and c0 is not None:
                piece = sh[:, c0:cx]
                ys, xs = np.nonzero(piece)
                if len(ys) and (ys.max() - ys.min()) > 0.5 * (y2 - y1):
                    pm, pd = _tight(piece, shd[:, c0:cx])
                    nb = _read_digit_nbest(pm, pd)
                    nbests.append(nb)
                    digits.append(_read_digit(pm, pd))
                c0 = None
        if not digits:
            continue
        text = ''.join(str(d) for d, _ in digits)
        conf = float(np.mean([c for _, c in digits]))
        if len(text) >= 2 and set(text) == {'1'}:
            # bar-graph ticks and bezel marks group into all-ones rows; a real
            # reading is never 11/111 mmHg, so don't let such rows look certain
            conf *= 0.5
        # alternative readings: flip the least confident digits to their
        # second-best pattern (geometry hard rules pick between them later)
        alternatives = []
        if len(digits) > 1:
            order = sorted(range(len(digits)), key=lambda i: digits[i][1])
            for i in order[:2]:
                if digits[i][1] < 0.6 and len(nbests[i]) > 1:
                    alt = list(text)
                    alt[i] = str(nbests[i][1][0])
                    alt = ''.join(alt)
                    if alt != text and alt not in alternatives:
                        alternatives.append(alt)
        corners = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
        back = cv2.perspectiveTransform(corners, Minv).reshape(4, 2)
        box = [int(back[:, 0].min()), int(back[:, 1].min()),
               int(back[:, 0].max()), int(back[:, 1].max())]
        near_edge = y1 < 0.03 * h or y2 > 0.97 * h or x1 < 0.03 * w or x2 > 0.97 * w
        out.append({'text': text, 'conf': conf, 'h': y2 - y1, 'box': box,
                    'touch': near_edge, 'alternatives': alternatives,
                    'poly': back.astype(int).tolist(), 'deskew': _deskew_angle})
    return out


def _plausible(row):
    return row['text'].isdigit() and 2 <= len(row['text']) <= 3 and 30 <= int(row['text']) <= 260


def _quality(rows):
    """How believable a screen reading is: tall, confident, plausible number rows.
    Rows touching the edge of the crop may have lost digits, so they count less."""
    good = sorted((r for r in rows if _plausible(r)), key=lambda r: -r['h'])[:3]
    q = sum(r['conf'] * r['h'] * (0.3 if r.get('touch') else 1.0) for r in good)
    return q * (1.0 + 0.5 * (len(good) == 3))


def _row_center_y(r):
    return (r['box'][1] + r['box'][3]) / 2.0


def _agrees(row, other_rows):
    cy, half = _row_center_y(row), 0.5 * (row['box'][3] - row['box'][1])
    return any(abs(_row_center_y(o) - cy) < half and o['text'] == row['text'] for o in other_rows)


def _read_one_orientation(img, label_boxes=None):
    """Returns (rows, quality) for the most believable screen. Every row gets
    'agreement' = share of other readings (other sensitivity levels and other
    screen candidates) that produced the same text at the same place."""
    scored = []
    for gray, M in find_lcd_candidates(img, label_boxes=label_boxes):
        rows = _rows_from_lcd(gray, M)
        scored.append((_quality(rows), gray, M, rows))
    scored = [t for t in scored if t[0] > 0]
    if not scored and CFG['whole_image_fallback']:
        # No screen qualified: try the whole photo (downscaled) as a last resort.
        s = min(1.0, CFG['warp_long_side'] / max(img.shape[:2]))
        wg = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        M = np.float64([[s, 0, 0], [0, s, 0], [0, 0, 1]])   # original -> scaled
        rows = _rows_from_lcd(wg, M)
        if _quality(rows) > 0:
            scored = [(_quality(rows), wg, M, rows)]
    if not scored:
        return [], 0.0
    scored.sort(key=lambda t: -t[0])
    best_q, gray, M, rows = scored[0]

    others = [_rows_from_lcd(gray, M, f) for f in (0.28, 0.42)]           # same screen
    others += [t[3] for t in scored[1:4] if t[0] >= 0.5 * best_q]          # other screens
    for r in rows:
        r['agreement'] = (1 + sum(_agrees(r, o) for o in others)) / (1 + len(others))
    return rows, best_q


def _unrotate_pt(x, y, rot, H, W):
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return y, H - 1 - x
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return W - 1 - y, x
    return W - 1 - x, H - 1 - y


def _unrotate_box(box, rot, H, W):
    """Map a box from a rotated copy of the image back to the original picture.
    H, W = height and width of the ORIGINAL image."""
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if rot == cv2.ROTATE_90_CLOCKWISE:
        back = [(y, H - 1 - x) for x, y in pts]
    elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        back = [(W - 1 - y, x) for x, y in pts]
    else:  # 180
        back = [(W - 1 - x, H - 1 - y) for x, y in pts]
    xs, ys = [p[0] for p in back], [p[1] for p in back]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ------------------------------------------------- photometric TTA + voting
def photometric_variants(img):
    """Copies of the photo with different tone curves, so that a digit that is
    invisible under one enhancement often shows up under another."""
    yield img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    yield cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    lut = np.clip((np.arange(256) / 255.0) ** 0.6 * 255.0, 0, 255).astype(np.uint8)
    yield cv2.LUT(img, lut)
    if CFG['tta_channels']:
        for ch in cv2.split(img):
            yield cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)


def _x_overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0


def _vote_merge(rows_per_variant, n_variants):
    """Merge the rows read from all photometric variants.
    Rows at the same place vote; the majority text wins and 'agreement' is the
    share of variants that read that text there."""
    clusters = []  # {'cy', 'h', 'members': [(variant_idx, row), ...]}
    for vi, rows in enumerate(rows_per_variant):
        for r in rows:
            cy, hh = _row_center_y(r), 0.5 * (r['box'][3] - r['box'][1])
            for cl in clusters:
                if abs(cl['cy'] - cy) < max(hh, 0.5 * cl['h']) \
                        and _x_overlap(cl['box'], r['box']):
                    cl['members'].append((vi, r))
                    cl['cy'] = float(np.mean([_row_center_y(m[1]) for m in cl['members']]))
                    cl['h'] = max(cl['h'], r['box'][3] - r['box'][1])
                    cl['box'] = [min(cl['box'][0], r['box'][0]), min(cl['box'][1], r['box'][1]),
                                 max(cl['box'][2], r['box'][2]), max(cl['box'][3], r['box'][3])]
                    break
            else:
                clusters.append({'cy': cy, 'h': r['box'][3] - r['box'][1],
                                 'box': list(r['box']), 'members': [(vi, r)]})

    out = []
    for cl in clusters:
        plausible = [(vi, r) for vi, r in cl['members'] if _plausible(r)]
        if not plausible:
            continue
        counts = {}
        for vi, r in plausible:                      # one vote per variant
            counts.setdefault(r['text'], set()).add(vi)
        text, voters = max(counts.items(), key=lambda kv: len(kv[1]))
        members = [r for vi, r in plausible if r['text'] == text]
        best = max(members, key=lambda r: r['conf'] * r['h'])
        row = dict(best)
        row['text'] = text
        row['conf'] = float(np.mean([r['conf'] for r in members]))
        row['agreement_local'] = best.get('agreement', 0.0)
        row['agreement'] = len(voters) / n_variants
        out.append(row)
    out.sort(key=_row_center_y)
    return out


def _apply_glare_cap(rows, img):
    """Variants often share the same glare-induced error, so a row sitting on a
    specular highlight may not claim high agreement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gm = _glare_mask(gray)
    for r in rows:
        x1, y1, x2, y2 = (int(v) for v in r['box'])
        sub = gm[max(0, y1):min(gm.shape[0], y2), max(0, x1):min(gm.shape[1], x2)]
        r['glare_frac'] = float(sub.mean()) if sub.size else 0.0
        if r['glare_frac'] > 0.03:
            r['agreement'] = min(r['agreement'], 0.5)
            r['glare'] = True
    return rows


def _read_with_tta(img, label_boxes=None):
    """Read the photo under several tone curves and vote. Returns
    (rows, quality, n_variants_run). Stops early once 2 variants fully agree
    on 3 plausible rows."""
    rows_per_variant = []
    merged, quality = [], 0.0
    for n_run, variant in enumerate(photometric_variants(img), start=1):
        rows, _q = _read_one_orientation(variant, label_boxes=label_boxes)
        rows_per_variant.append(rows)
        merged = _vote_merge(rows_per_variant, n_run)
        quality = _quality(merged)
        if n_run >= 2:
            solid = [r for r in merged if _plausible(r) and r['agreement'] >= 0.999]
            if len(solid) >= 3:
                break
    return merged, quality, len(rows_per_variant)


def read_lcd(img, label_boxes=None):
    """Returns (rows, found): the digit rows of the most believable screen.
    Tries the photo upright first; if that does not give 3 plausible numbers it also
    tries sideways / upside-down (some photos are stored rotated).
    `label_boxes` (optional) are OCR label rects in original-image pixels, used
    to propose one extra LCD candidate around the printed labels."""
    H, W = img.shape[:2]
    if CFG['use_tta']:
        def read(i):
            r, q, _n = _read_with_tta(i, label_boxes=label_boxes)
            return r, q
    else:
        def read(i):
            return _read_one_orientation(i, label_boxes=label_boxes)
    rows, q = read(img)
    if sum(_plausible(r) for r in rows) >= 3:
        return _apply_glare_cap(rows, img), True
    best_rows, best_q = rows, q
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        r_rows, r_q = read(cv2.rotate(img, rot))[:2]
        if r_q > best_q and sum(_plausible(r) for r in r_rows) >= 3:
            for r in r_rows:
                r['box'] = _unrotate_box(r['box'], rot, H, W)
                if 'poly' in r:
                    r['poly'] = [list(_unrotate_pt(px, py, rot, H, W))
                                 for px, py in r['poly']]
            best_rows, best_q = r_rows, r_q
    return _apply_glare_cap(best_rows, img), best_q > 0
