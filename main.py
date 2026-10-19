# main.py
"""Usage (needs seven_segment.py, detector.py, geometry.py next to it):
    python main.py images/photo.jpg                       # one photo
    python main.py --folder images --truth truth.csv      # measure accuracy
    python main.py --folder images --truth truth.csv --report --json out.json

truth.csv looks like:
    filename,sys,dia,pul
    photo1.jpg,118,76,72
"""

import argparse
import csv
import json
import os
import time

import cv2

import seven_segment
from detector import BPDetector
from geometry import assign_readings, calculate_median_slope


def _stage_report(image, image_path, result, report_dir, failed):
    """Per-image stage diagnostics. Always writes <stem>.txt; on failure also
    saves the best warped LCD crop and its dark_mask as PNGs."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(report_dir, exist_ok=True)

    cands = seven_segment.find_lcd_candidates(image)
    best_q, best_rows, best_gray = 0.0, [], None
    for gray, M in cands:
        rows = seven_segment._rows_from_lcd(gray, M)
        q = seven_segment._quality(rows)
        if q > best_q:
            best_q, best_rows, best_gray = q, rows, gray

    n_plausible = sum(seven_segment._plausible(r) for r in best_rows)
    sat = float((best_gray >= 245).mean()) if best_gray is not None else 0.0

    lines = [
        f'lcd candidates: {len(cands)}',
        f'best candidate: rows={len(best_rows)} plausible={n_plausible} '
        f'quality={best_q:.1f}',
        f'saturated fraction in LCD crop: {sat:.4f}',
        f"rows: {[(r['text'], round(r['conf'], 2)) for r in best_rows]}",
        f"readings: { {f: c['value'] for f, c in result['readings'].items()} }",
        f"needs_review: {result['needs_review']}",
        f"reasons: {result['reasons']}",
    ]
    print('  [REPORT] ' + ' | '.join(lines[:3]))
    with open(os.path.join(report_dir, stem + '.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    if failed and best_gray is not None:
        cv2.imwrite(os.path.join(report_dir, stem + '_warped.png'), best_gray)
        cv2.imwrite(os.path.join(report_dir, stem + '_mask.png'),
                    seven_segment.dark_mask(best_gray) * 255)


def run_pipeline(image_path, detector, debug_dir='debug', report=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not load image at {image_path}')

    labels, cands, polys = detector.detect_all(image)
    slope = calculate_median_slope(polys)
    print(f'[INFO] image size: {image.shape[1]}x{image.shape[0]}  labels found: {list(labels)}  tilt: {slope:.3f}')
    print('[INFO] numbers found:', [(c['value'], round(c['conf'], 2), round(c['agreement'], 2)) for c in cands])
    result = assign_readings(cands, labels, slope, image.shape[0])
    result['labels_found'] = list(labels.keys())

    if report:
        _stage_report(image, image_path, result,
                      os.path.join(debug_dir or 'debug', 'report'),
                      failed=not result['readings'] or result['needs_review'])

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for c in cands:
            x1, y1, x2, y2 = c['box']
            cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), 1)
        for f, box in labels.items():
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (255, 100, 0), 2)
        for f, c in result['readings'].items():
            x1, y1, x2, y2 = c['box']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, f"{f}: {c['value']}", (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        out = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(out, image)
    return result


def summarize(result):
    vals = {f: c['value'] for f, c in result['readings'].items()}
    flag = 'CHECK MANUALLY' if result['needs_review'] else 'OK'
    return vals, flag


def evaluate(folder, truth_csv, detector, report=False):
    truth = {}
    with open(truth_csv, newline='') as f:
        for row in csv.DictReader(f):
            truth[row['filename']] = (int(row['sys']), int(row['dia']), int(row['pul']))

    total = correct = flagged = accepted = accepted_correct = 0
    t0 = time.time()
    for name, expected in truth.items():
        res = run_pipeline(os.path.join(folder, name), detector, report=report)
        vals, flag = summarize(res)
        got = (vals.get('SYS'), vals.get('DIA'), vals.get('PUL'))
        total += 1
        ok = got == expected
        correct += ok
        if res['needs_review']:
            flagged += 1
        else:
            accepted += 1
            accepted_correct += ok
        print(f'{name}: got {got} expected {expected} -> '
              f"{'match' if ok else 'WRONG'} [{flag}]")

    elapsed = time.time() - t0
    metrics = {
        'total': total,
        'correct': correct,
        'exact_match_pct': round(100 * correct / max(1, total), 1),
        'flagged': flagged,
        'flagged_pct': round(100 * flagged / max(1, total), 1),
        'accepted': accepted,
        'accepted_correct': accepted_correct,
        'correct_among_accepted_pct': round(100 * accepted_correct / accepted, 1) if accepted else None,
        'wrong_and_not_flagged': accepted - accepted_correct,
        'mean_seconds_per_image': round(elapsed / max(1, total), 2),
    }

    print('\n=== RESULTS ===')
    print(f"All 3 values exactly right: {correct}/{total} ({metrics['exact_match_pct']}%)")
    print(f'Flagged for manual check:   {flagged}/{total}')
    if accepted:
        print(f"Correct among auto-accepted: {accepted_correct}/{accepted} "
              f"({metrics['correct_among_accepted_pct']}%)")
        print(f"Wrong AND not flagged (dangerous): {metrics['wrong_and_not_flagged']}")
    print(f"Mean seconds per image: {metrics['mean_seconds_per_image']}")
    return metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image', nargs='?')
    ap.add_argument('--folder')
    ap.add_argument('--truth')
    ap.add_argument('--report', action='store_true',
                    help='write per-image stage reports to debug/report/')
    ap.add_argument('--json', help='write eval metrics to this JSON file')
    args = ap.parse_args()

    det = BPDetector()  # load the OCR model once
    if args.folder and args.truth:
        metrics = evaluate(args.folder, args.truth, det, report=args.report)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'metrics written to {args.json}')
    elif args.image:
        res = run_pipeline(args.image, det, report=args.report)
        vals, flag = summarize(res)
        print('\n--- FINAL BP READINGS ---')
        print(vals, f'[{flag}]')
        for r in res['reasons']:
            print(' -', r)
        print('Debug picture saved in debug/')
    else:
        ap.print_help()
