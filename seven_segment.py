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

# Lit bars, in order: a=top, b=top-right, c=bottom-right, d=bottom,
# e=bottom-left, f=top-left, g=middle.  (1,1,1,0,0,1,0) is the 4-stroke "7"
# some monitors draw.
DIGITS = {
    (1, 1, 1, 1, 1, 1, 0): 0, (0, 1, 1, 0, 0, 0, 0): 1, (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3, (0, 1, 1, 0, 0, 1, 1): 4, (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6, (1, 1, 1, 0, 0, 0, 0): 7, (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8, (1, 1, 1, 1, 0, 1, 1): 9,
}
WARP_LONG_SIDE = 600   # the straightened LCD is resized to about this many pixels
MAX_CANDIDATES = 10    # how many possible screens are read and compared


# ----------------------------------------------------------------- masking
def dark_mask(gray, frac=0.35):
    """1 where a pixel is clearly darker than its surroundings (the lit segments)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    # Background brightness = a very wide blur. Doing it on a 1/4-size copy gives
    # the same result about 20x faster.
    sigma = max(8, min(gray.shape) / 8)
    small = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    bg = cv2.resize(small, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = bg.astype(np.int16) - g.astype(np.int16)
    thr = max(10, frac * np.percentile(d, 99.5))
    m = (d > thr).astype(np.uint8) * 255
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
    k = WARP_LONG_SIDE / max(w, h)
    w, h = max(20, int(w * k)), max(20, int(h * k))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h)), M


def find_lcd_candidates(img, keep=MAX_CANDIDATES):
    """Up to `keep` likely screens as [(straightened gray LCD, matrix), ...]."""
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

    for blur in (5, 9):
        g = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(gray), (blur, blur), 0)
        for lo, hi in ((40, 120), (20, 60), (60, 160)):
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

    scored = []
    for q in quads:
        for pad in (0.0, 0.03):
            warped, M = _warp(img, _grow(q / s, pad))
            wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            sc = _digit_score(wg)
            if sc >= 3:
                scored.append((sc, wg, M))
    scored.sort(key=lambda t: -t[0])
    return [(g, M) for _, g, M in scored[:keep]]


# ---------------------------------------------------------- digit reading
def _shear(mask, k):
    h, w = mask.shape
    M = np.float32([[1, -k, k * h / 2.0], [0, 1, 0]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)


def _best_shear(mask):
    """The slant that makes the digits' columns most separate."""
    best, bk = -1.0, 0.0
    for k in np.arange(-0.35, 0.351, 0.025):
        col = _shear(mask, k).sum(axis=0).astype(np.float64)
        sc = float((col ** 2).sum())
        if sc > best:
            best, bk = sc, float(k)
    return bk


def _read_digit(m):
    """m = tight 0/1 crop of ONE digit, slant removed. Returns (digit, confidence)."""
    h, w = m.shape
    if w < 0.42 * h:
        return 1, 1.0
    zones = {
        'a': (.15, .85, 0, .17), 'g': (.15, .85, .41, .59), 'd': (.15, .85, .83, 1),
        'f': (0, .3, .08, .46), 'b': (.7, 1, .08, .46),
        'e': (0, .3, .54, .92), 'c': (.7, 1, .54, .92),
    }
    r = []
    for name in 'abcdefg':
        x1, x2, y1, y2 = zones[name]
        z = m[int(y1 * h):max(int(y2 * h), int(y1 * h) + 1),
              int(x1 * w):max(int(x2 * w), int(x1 * w) + 1)]
        r.append(min(1.0, float(z.mean()) / 0.7))
    dists = sorted((sum(abs(a - b) for a, b in zip(r, pat)), d) for pat, d in DIGITS.items())
    return dists[0][1], max(0.0, min(1.0, dists[1][0] - dists[0][0]))


def _rows_from_lcd(gray, M, frac=0.35):
    """Read all digit rows on one straightened LCD picture.
    Each row: {'text','conf','h','box','touch'}, top to bottom, box in full-image pixels."""
    h, w = gray.shape
    m = dark_mask(gray, frac)
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
        if gh < 0.07 * h or gw > 1.1 * gh or gw < 0.16 * gh:
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
        pad = 6
        ry1, ry2, rx1, rx2 = max(0, y1 - pad), y2 + pad, max(0, x1 - 3 * pad), x2 + 3 * pad
        sub = np.zeros_like(m[ry1:ry2, rx1:rx2])
        for g in gl:
            sub |= ((lab == g['id'])[ry1:ry2, rx1:rx2] & (m[ry1:ry2, rx1:rx2] > 0)).astype(np.uint8)
        sh = _shear(sub, _best_shear(sub))

        cols = sh.sum(axis=0) > 0
        digits, c0 = [], None
        for cx in range(len(cols) + 1):
            on = cx < len(cols) and cols[cx]
            if on and c0 is None:
                c0 = cx
            if not on and c0 is not None:
                piece = sh[:, c0:cx]
                ys, xs = np.nonzero(piece)
                if len(ys) and (ys.max() - ys.min()) > 0.5 * (y2 - y1):
                    digits.append(_read_digit(piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
                c0 = None
        if not digits:
            continue
        text = ''.join(str(d) for d, _ in digits)
        conf = float(np.mean([c for _, c in digits]))
        corners = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
        back = cv2.perspectiveTransform(corners, Minv).reshape(4, 2)
        box = [int(back[:, 0].min()), int(back[:, 1].min()),
               int(back[:, 0].max()), int(back[:, 1].max())]
        near_edge = y1 < 0.03 * h or y2 > 0.97 * h or x1 < 0.03 * w or x2 > 0.97 * w
        out.append({'text': text, 'conf': conf, 'h': y2 - y1, 'box': box, 'touch': near_edge})
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


def _read_one_orientation(img):
    """Returns (rows, quality) for the most believable screen. Every row gets
    'agreement' = share of other readings (other sensitivity levels and other
    screen candidates) that produced the same text at the same place."""
    scored = []
    for gray, M in find_lcd_candidates(img):
        rows = _rows_from_lcd(gray, M)
        scored.append((_quality(rows), gray, M, rows))
    scored = [t for t in scored if t[0] > 0]
    if not scored:
        return [], 0.0
    scored.sort(key=lambda t: -t[0])
    best_q, gray, M, rows = scored[0]

    others = [_rows_from_lcd(gray, M, f) for f in (0.28, 0.42)]           # same screen
    others += [t[3] for t in scored[1:4] if t[0] >= 0.5 * best_q]          # other screens
    for r in rows:
        r['agreement'] = (1 + sum(_agrees(r, o) for o in others)) / (1 + len(others))
    return rows, best_q


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


def read_lcd(img):
    """Returns (rows, found): the digit rows of the most believable screen.
    Tries the photo upright first; if that does not give 3 plausible numbers it also
    tries sideways / upside-down (some photos are stored rotated)."""
    H, W = img.shape[:2]
    rows, q = _read_one_orientation(img)
    if sum(_plausible(r) for r in rows) >= 3:
        return rows, True
    best_rows, best_q = rows, q
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        r_rows, r_q = _read_one_orientation(cv2.rotate(img, rot))
        if r_q > best_q and sum(_plausible(r) for r in r_rows) >= 3:
            for r in r_rows:
                r['box'] = _unrotate_box(r['box'], rot, H, W)
            best_rows, best_q = r_rows, r_q
    return best_rows, best_q > 0
