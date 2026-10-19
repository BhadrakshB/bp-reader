# augment.py
"""Create a stressed copy of the labeled image set.

Variants per source image (labels are copied unchanged):
  rot_p4/rot_m4/rot_p8/rot_m8/rot_p12/rot_m12  in-plane rotation, border replicate
  gamma16/gamma22                              gamma darkening
  glare_rand/glare_lcd                         Gaussian saturated highlight
  blur                                         slight Gaussian blur
  combo1/combo2/combo3                         rotation + darker + glare combined

Usage:
    venv/bin/python augment.py --folder images --truth truth.csv --out images_aug
Writes <out>/ with the originals plus all variants, and <out>/../truth_aug.csv
(or --truth-out) with one row per written image.
"""

import argparse
import csv
import os

import cv2
import numpy as np

SEED = 42


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def darken(img, gamma):
    lut = np.clip((np.arange(256) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def glare(img, cx, cy, radius, peak=255.0):
    """Additive Gaussian highlight, saturating near its center."""
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    g = np.exp(-(((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2)
                 / (2.0 * radius * radius)))
    out = img.astype(np.float32) + g[:, :, None] * peak
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img):
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _screen_center(img):
    """Best guess for where the digits are: center of the best LCD candidate."""
    try:
        import seven_segment
        cands = seven_segment.find_lcd_candidates(img, keep=1)
        if cands:
            gray, M = cands[0]
            Minv = np.linalg.inv(M)
            h, w = gray.shape
            c = cv2.perspectiveTransform(
                np.float32([[[w / 2.0, h / 2.0]]]), Minv)[0, 0]
            return float(c[0]), float(c[1])
    except Exception:
        pass
    h, w = img.shape[:2]
    return w / 2.0, h / 2.0


def variants_for(img, rng):
    """[(suffix, image), ...] for one source image."""
    out = []
    for deg in (4, -4, 8, -8, 12, -12):
        out.append((f'rot_{"p" if deg > 0 else "m"}{abs(deg)}', rotate(img, deg)))
    out.append(('gamma16', darken(img, 1.6)))
    out.append(('gamma22', darken(img, 2.2)))
    h, w = img.shape[:2]
    r = 0.22 * min(h, w)
    out.append(('glare_rand', glare(img, rng.uniform(0.2, 0.8) * w,
                                    rng.uniform(0.2, 0.8) * h, r)))
    cx, cy = _screen_center(img)
    out.append(('glare_lcd', glare(img, cx, cy, r)))
    out.append(('blur', blur(img)))
    out.append(('combo1', glare(darken(rotate(img, 8), 1.6), cx, cy, r)))
    out.append(('combo2', glare(darken(rotate(img, -12), 2.2),
                                rng.uniform(0.2, 0.8) * w,
                                rng.uniform(0.2, 0.8) * h, r)))
    out.append(('combo3', blur(darken(rotate(img, 4), 1.6))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--truth-out', default=None)
    args = ap.parse_args()

    truth = {}
    with open(args.truth, newline='') as f:
        for row in csv.DictReader(f):
            truth[row['filename']] = (row['sys'], row['dia'], row['pul'])

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (sys_v, dia_v, pul_v) in sorted(truth.items()):
        src = os.path.join(args.folder, name)
        img = cv2.imread(src)
        if img is None:
            print(f'[WARN] cannot read {src}, skipped')
            continue
        stem, ext = os.path.splitext(name)
        copies = [(name, img)] + [(f'{stem}__{sfx}{ext}', v)
                                  for sfx, v in variants_for(img, rng)]
        for out_name, out_img in copies:
            cv2.imwrite(os.path.join(args.out, out_name), out_img)
            rows.append((out_name, sys_v, dia_v, pul_v))
        print(f'{name}: {len(copies)} images written')

    truth_out = args.truth_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), 'truth_aug.csv')
    with open(truth_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'sys', 'dia', 'pul'])
        w.writerows(rows)
    print(f'\n{len(rows)} images in {args.out}, labels in {truth_out}')


if __name__ == '__main__':
    main()
