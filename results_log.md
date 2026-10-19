# Results log

Metric definitions: exact-match % = all 3 of SYS/DIA/PUL exactly right;
flagged % = needs_review; wrong-and-not-flagged = accepted silently with a
wrong value (the metric that must never go up).

## Baseline (pre-change, commit 03ec3de)

**Blocked:** no truth.csv exists yet (user will provide it). The table below
tracks the 4 known-failure images instead until then.

Baseline on the 4 known-failure images (all returned `{}`, i.e. flagged):

| image | true | baseline result |
|---|---|---|
| 20210729_07_43_50 (NUTEC)  | 105/66/88 | `{}` — zero number candidates |
| 20210730_09_55_30 (Beurer) | 102/61/78 | `{}` — triples died on hard rules |
| 20210731_12_46_15 (Omron)  | 136/87/78 | `{}` — row fragmentation under tilt |
| bhadraksh-1 (TATA)         | 142/64/80 | `{}` — zero number candidates |

## Per-phase progression on the 4 known failures (read_lcd plausible rows)

| phase | NUTEC 105/66/88 | Beurer 102/61/78 | Omron 136/87/78 | TATA 142/64/80 |
|---|---|---|---|---|
| baseline | — | 81/78(dup)/30 garbage | 86/68/61 garbage | — |
| P1 dark_mask | 105 (0.86) found | 30 only | 86/68/61 | — |
| P2 TTA+voting | 105, 66 | 81/81/30 | 135/87/78 +junk | 61 only |
| P3 deskew | 105, 66 | 81/70/30 | 135/87/78 +junk | 61 only |
| P4 finder | 105, 66 | 81/30/36 | 135/87/78 | 79/61 |
| P5 decoding | 105, 66, 112 | 81/81/30 | 135(+alt **136**), 87 | 42/161 |
| P6 geometry (end-to-end) | **105/66**/112 flagged | `{}` flagged | **136/87**/111 flagged | `{}` flagged |

End-to-end after Phase 6 (full pipeline incl. PaddleOCR): 2 of 4 images now
produce 2-of-3 correct values; all 4 remain flagged for review;
wrong-and-not-flagged = 0 (baseline also 0 because everything was flagged).

## Mask method A/B (Phase 1, 4 known failures)

- `contrast` (log relative contrast, glare-excluded background): recovers NUTEC
  105 row; no confident garbage.
- `blackhat`: misses NUTEC entirely, produces confident wrong reads (111, 211
  at conf 1.0) on Omron. Kept as fallback variant only, default = contrast.
- `use_inpaint`: no measurable difference on these 4 (no large glare); default
  off pending augmented-set A/B.

## warp_long_side A/B (Phase 4, 4 known failures)

600: ~5.5–9 s/img (read_lcd only). 800: ~7–11 s/img, no accuracy gain on this
sample (one regression: 105→115). Default 800 per spec; ablate on aggregate.

## Stress-set check (augmented variants of the 4 known failures, read_lcd level)

60 variants (14 per image: 6 rotations, 2 gamma, 2 glare, blur, 3 combos).
Metric: all 3 true values present among plausible rows (full recall).

- Full recall: 0/4 in every augmentation category — these 4 images are the
  hardest of the set; their variants stay hard. Omron variants usually recover
  87 and 78 (and 135/136 SYS); NUTEC variants usually recover 105 and 66;
  Beurer is consistently wrong (81/30); TATA consistently fails.
- Inpaint A/B on the 8 glare variants: 0/8 vs 0/8 full recall — no measurable
  effect. Default stays OFF.
- Found and fixed via this check: artifact slivers (tick marks, icons) grouped
  into all-ones rows like '111' at conf 1.0 via the narrow-'1' shortcut —
  a triple like (111, 44, 50) could have passed the geometry hard rules
  unflagged. Now: narrow-'1' shortcut returns conf 0.6, all-ones rows get
  conf * 0.5.

## Old-vs-new regression check (12 random clean-ish images, read_lcd only)

- New reader finds strictly more rows (old often found none); e.g. Omron
  20210809_16_13_29: old found 1 of 3 true values, new finds 2 (100, 67).
- New also emits junk rows on blank/mid-inflation screens ('111', '44') —
  these carry low conf/agreement after the all-ones fix and cannot form a
  valid unflagged triple.

## Baseline / augmented-set metrics

PENDING truth.csv. Commands to run once it exists:

    venv/bin/python main.py --folder images --truth truth.csv --report --json baseline.json
    venv/bin/python augment.py --folder images --truth truth.csv --out images_aug
    venv/bin/python main.py --folder images_aug --truth truth_aug.csv --report --json baseline_aug.json

(For the pre-change baseline numbers, run the same from a worktree at 03ec3de.)
