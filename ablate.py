# ablate.py
"""Ablation runner: evaluate() once per CFG toggle switched off (plus all-on),
writing one JSON per run. Usage:
    venv/bin/python ablate.py --folder images_aug --truth truth_aug.csv --out ablation
"""

import argparse
import json
import os

import seven_segment as ss
import detector as det_mod
from detector import BPDetector
from main import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', default='ablation')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det = BPDetector()  # load the OCR model once
    runs = [('all_on', {})]
    for key in ss.CFG:
        runs.append((f'ss_{key}_off', {('ss', key): False}))
    for key in det_mod.CFG:
        runs.append((f'det_{key}_off', {('det', key): False}))
    runs.append(('ss_blackhat', {('ss', 'mask_method'): 'blackhat'}))

    for name, overrides in runs:
        saved = {}
        for (mod, key), val in overrides.items():
            cfg = ss.CFG if mod == 'ss' else det_mod.CFG
            saved[(mod, key)] = cfg[key]
            cfg[key] = val
        metrics = evaluate(args.folder, args.truth, det)
        metrics['run'] = name
        with open(os.path.join(args.out, name + '.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        for (mod, key), val in saved.items():
            (ss.CFG if mod == 'ss' else det_mod.CFG)[key] = val
        print(f'>>> {name}: {metrics}\n')


if __name__ == '__main__':
    main()
