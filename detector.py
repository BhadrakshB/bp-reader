# detector.py
"""Finds the three big readings plus (optionally) the printed labels.

Main reader : seven_segment.py  - plain OpenCV, made for LCD digits.
Helper      : PaddleOCR         - finds the printed words SYS / DIA / PUL, the tilt
                                  of the device, and cross-checks the numbers.
PaddleOCR is optional: if it is missing, the seven-segment reader still works.
"""

import os
import re

import cv2
import numpy as np

os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
try:
    from paddleocr import PaddleOCR
except ImportError:  # still works, just without labels / cross-check
    PaddleOCR = None

import seven_segment

CFG = {
    'use_alternatives': True,  # expose per-row second-best readings as candidates
    'use_dedup': True,         # merge duplicate candidates (same value, same place)
}

# Patterns run on LETTERS ONLY, so "P/min" becomes "PMIN".
LABEL_PATTERNS = {
    'SYS': re.compile(r'^(SYS|SYSTOLIC)'),
    'DIA': re.compile(r'^(DIA|DIASTOLIC|DYS)'),
    'PUL': re.compile(r'^(PUL|PULSE|PMIN|BPM)'),
}
CONFUSABLE = {'O': '0', 'I': '1', 'L': '1', '|': '1', '!': '1', 'B': '8', 'S': '5', 'Z': '2'}
NUMBER_TOKEN = re.compile(r'^[0-9OILBSZ|!]{2,3}$')


class BPDetector:
    def __init__(self):
        self.ocr = None
        self.last_slope = None   # fallback device tilt from the LCD reader
        if PaddleOCR is None:
            print('[WARN] PaddleOCR not installed: no labels, no cross-check.')
            return
        try:
            self.ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                 use_textline_orientation=True, lang='en', enable_mkldnn=False)
        except (ValueError, TypeError):
            self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)

    # ------------------------------------------------------------ PaddleOCR
    def _ocr(self, img, min_score=0.3):
        """[(4-point polygon, text, score), ...]"""
        out = []
        if self.ocr is None:
            return out
        if hasattr(self.ocr, 'predict'):
            for res in self.ocr.predict(img):
                data = res if isinstance(res, dict) else getattr(res, 'res', res)
                polys = data.get('dt_polys', data.get('rec_polys', []))
                for poly, text, score in zip(polys, data.get('rec_texts', []), data.get('rec_scores', [])):
                    if score >= min_score:
                        out.append((poly.tolist() if hasattr(poly, 'tolist') else list(poly),
                                    str(text), float(score)))
        else:
            res = self.ocr.ocr(img, cls=True)
            if res and res[0]:
                for line in res[0]:
                    if line[1][1] >= min_score:
                        out.append((line[0], line[1][0], float(line[1][1])))
        return out

    @staticmethod
    def _numbers_from_text(text, rect):
        """One OCR line -> list of (value, box). Handles 'SYS 118' style lines."""
        tokens = [t.strip(".,:;-_'\"") for t in text.upper().split()]
        tokens = [t for t in tokens if t]
        total = sum(len(t) for t in tokens) or 1
        x1, y1, x2, y2 = rect
        found, cum = [], 0
        for tok in tokens:
            start = x1 + (x2 - x1) * cum / total
            cum += len(tok)
            end = x1 + (x2 - x1) * cum / total
            if NUMBER_TOKEN.match(tok) and re.search(r'\d', tok):
                value = int(''.join(CONFUSABLE.get(ch, ch) for ch in tok))
                if 30 <= value <= 260:
                    found.append((value, [int(start), y1, int(end), y2]))
        return found

    @staticmethod
    def _overlap(a, b):
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        small = max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
        return (ix2 - ix1) * (iy2 - iy1) / small

    @staticmethod
    def _to_rect(pts):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    @staticmethod
    def _dedup(cands):
        """Merge candidates with the same value at the same place (e.g. the LCD
        reader and the PaddleOCR safety net both found the 78)."""
        kept = []
        for c in sorted(cands, key=lambda c: -(c['conf'] + c['agreement'])):
            for k in kept:
                if k['value'] == c['value'] and BPDetector._overlap(k['box'], c['box']) > 0.3:
                    k['agreement'] = max(k['agreement'], c['agreement'])
                    break
            else:
                kept.append(c)
        return kept

    # ----------------------------------------------------------------- main
    def detect_all(self, image):
        """`image` is a BGR array. Returns (labels, candidates, polygons).
        candidate = {'value','box','conf','agreement'}"""
        # 1) Printed words + other numbers through PaddleOCR (optional helper)
        labels, polys, paddle_cands = {}, [], []
        for pts, text, score in self._ocr(image):
            rect = self._to_rect(pts)
            if score >= 0.4:
                polys.append(pts)
            letters = re.sub(r'[^A-Za-z]', '', text).upper()
            for field, pat in LABEL_PATTERNS.items():
                if pat.search(letters) and field not in labels:
                    labels[field] = rect
            for value, box in self._numbers_from_text(text, rect):
                paddle_cands.append({'value': value, 'box': box, 'conf': score})

        # 2) The real reader: seven-segment digits
        rows, _ = seven_segment.read_lcd(image, label_boxes=list(labels.values()))
        cands = []
        n_alts = 0
        for r in rows:
            if not seven_segment._plausible(r):
                continue
            cand = {'value': int(r['text']), 'box': r['box'],
                    'conf': r['conf'], 'agreement': r['agreement']}
            if 'poly' in r:
                cand['poly'] = r['poly']
            cand['glare_frac'] = r.get('glare_frac', 0.0)
            for p in paddle_cands:                      # cross-check with PaddleOCR
                if self._overlap(cand['box'], p['box']) > 0.5:
                    if p['value'] == cand['value']:
                        cand['agreement'] = 1.0
                    elif p['conf'] > 0.8:
                        cand['agreement'] = min(cand['agreement'], 0.3)
                    break
            cands.append(cand)
            # second-best readings of the same row, penalized and capped, so the
            # geometry hard rules can pick them when the primary read is wrong
            if CFG['use_alternatives']:
                for alt in r.get('alternatives', [])[:2]:
                    if n_alts >= 4 or not alt.isdigit():
                        continue
                    value = int(alt)
                    if 30 <= value <= 260:
                        cands.append({'value': value, 'box': r['box'],
                                      'conf': r['conf'] * 0.7,
                                      'agreement': r['agreement'] * 0.5,
                                      'alternative': True})
                        n_alts += 1

        if CFG['use_dedup']:
            cands = self._dedup(cands)

        # Fallback device tilt: median top-edge slope of the LCD rows, for when
        # PaddleOCR produced no polygons.
        slopes = []
        for r in rows:
            p = r.get('poly')
            if p and seven_segment._plausible(r):
                dx = p[1][0] - p[0][0]
                if abs(dx) > 5:
                    slopes.append((p[1][1] - p[0][1]) / dx)
        self.last_slope = float(np.median(slopes)) if slopes else None

        # 3) Safety net: if the LCD reader found fewer than 3 numbers, add PaddleOCR's
        if len(cands) < 3:
            for p in paddle_cands:
                if not any(self._overlap(c['box'], p['box']) > 0.5 for c in cands):
                    cands.append({'value': p['value'], 'box': p['box'],
                                  'conf': p['conf'], 'agreement': 0.7})
        return labels, cands, polys
