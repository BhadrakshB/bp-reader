# main.py
"""Usage (needs seven_segment.py, detector.py, geometry.py next to it):
    python main.py images/photo.jpg                       # one photo
    python main.py --folder images --truth truth.csv      # measure accuracy

truth.csv looks like:
    filename,sys,dia,pul
    photo1.jpg,118,76,72
"""

import argparse
import csv
import os

import cv2

from detector import BPDetector
from geometry import assign_readings, calculate_median_slope


def run_pipeline(image_path, detector, debug_dir='debug'):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not load image at {image_path}')

    labels, cands, polys = detector.detect_all(image)
    slope = calculate_median_slope(polys)
    print(f'[INFO] image size: {image.shape[1]}x{image.shape[0]}  labels found: {list(labels)}  tilt: {slope:.3f}')
    print('[INFO] numbers found:', [(c['value'], round(c['conf'], 2), round(c['agreement'], 2)) for c in cands])
    result = assign_readings(cands, labels, slope, image.shape[0])
    result['labels_found'] = list(labels.keys())

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


def evaluate(folder, truth_csv, detector):
    truth = {}
    with open(truth_csv, newline='') as f:
        for row in csv.DictReader(f):
            truth[row['filename']] = (int(row['sys']), int(row['dia']), int(row['pul']))

    total = correct = flagged = accepted = accepted_correct = 0
    for name, expected in truth.items():
        res = run_pipeline(os.path.join(folder, name), detector)
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

    print('\n=== RESULTS ===')
    print(f'All 3 values exactly right: {correct}/{total} ({100 * correct / max(1, total):.0f}%)')
    print(f'Flagged for manual check:   {flagged}/{total}')
    if accepted:
        print(f'Correct among auto-accepted: {accepted_correct}/{accepted} '
              f'({100 * accepted_correct / accepted:.0f}%)')
        print(f'Wrong AND not flagged (dangerous): {accepted - accepted_correct}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image', nargs='?')
    ap.add_argument('--folder')
    ap.add_argument('--truth')
    args = ap.parse_args()

    det = BPDetector()  # load the OCR model once
    if args.folder and args.truth:
        evaluate(args.folder, args.truth, det)
    elif args.image:
        res = run_pipeline(args.image, det)
        vals, flag = summarize(res)
        print('\n--- FINAL BP READINGS ---')
        print(vals, f'[{flag}]')
        for r in res['reasons']:
            print(' -', r)
        print('Debug picture saved in debug/')
    else:
        ap.print_help()
