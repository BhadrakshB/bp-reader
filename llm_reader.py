# llm_reader.py
"""Reads SYS / DIA / PUL from a BP monitor photo with Gemini and returns JSON.

    pip install google-genai opencv-python numpy
    export GEMINI_API_KEY=...
    python llm_reader.py photo.jpg
    python llm_reader.py photo.jpg --crosscheck   # also run seven_segment.py
"""

import argparse
import json
import os
import re
import sys

import cv2

MODEL = os.environ.get('BP_MODEL', 'gemini-2.5-flash')  # e.g. gemini-2.5-pro for harder photos
MAX_SIDE = 1568

RANGES = {'systolic': (70, 260), 'diastolic': (30, 150), 'pulse': (30, 220)}

PROMPT = """You are reading the display of a blood pressure monitor in a photo.

Return ONLY a JSON object, no prose, no markdown fences, with exactly these keys:
{
  "systolic": <integer or null>,
  "diastolic": <integer or null>,
  "pulse": <integer or null>,
  "confidence": "high" | "medium" | "low",
  "notes": "<short reason if anything is unclear, else empty string>"
}

Rules:
- Systolic is the larger pressure (usually top, labelled SYS), diastolic the smaller (DIA),
  pulse is heart rate (PUL, PULSE, or a heart icon, often in /min or bpm).
- Use the printed labels or icons to decide which number is which; do not rely on position alone.
- Ignore the date, time, memory slot number, battery icon, and bezel scale markings.
- Read seven-segment digits carefully: do not confuse 1/7, 3/8, 5/6, 0/8.
- If a digit is hidden by glare, blurred, or cut off, use null for that field and lower the confidence.
- Never guess a value that you cannot see."""


# --------------------------------------------------------------- image prep
def load_image_bytes(path, max_side=MAX_SIDE):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Could not load image at {path}')
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return buf.tobytes()


# ------------------------------------------------------------------ parsing
def parse_json(text):
    """Extract the first JSON object from the model reply, tolerating fences/prose."""
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise ValueError(f'No JSON object in model reply: {text[:200]!r}')
        return json.loads(m.group(0))


def to_int(v):
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- validation
def validate(data):
    """Returns (clean_readings, needs_review, reasons)."""
    reasons = []
    out = {k: to_int(data.get(k)) for k in RANGES}

    for field, (lo, hi) in RANGES.items():
        v = out[field]
        if v is None:
            reasons.append(f'{field} missing')
        elif not lo <= v <= hi:
            reasons.append(f'{field} {v} outside plausible range {lo}-{hi}')
            out[field] = None

    sys_, dia = out['systolic'], out['diastolic']
    if sys_ is not None and dia is not None:
        if sys_ <= dia:
            reasons.append('systolic must be greater than diastolic')
        elif not 20 <= sys_ - dia <= 100:
            reasons.append('unusual systolic-diastolic gap')

    if data.get('confidence') in ('low', 'medium'):
        reasons.append(f"model confidence {data.get('confidence')}")
    if data.get('notes'):
        reasons.append(f"model note: {data['notes']}")

    return out, bool(reasons), reasons


# ------------------------------------------------------------------ LLM call
def make_client():
    from google import genai
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('Set GEMINI_API_KEY in your environment.')
    return genai.Client(api_key=api_key)


def read_with_llm(path, client=None, model=MODEL, retries=2):
    from google.genai import types
    client = client or make_client()
    image_bytes = load_image_bytes(path)
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                    response_mime_type='application/json',  # forces JSON output
                ),
            )
            return parse_json(resp.text)
        except (ValueError, json.JSONDecodeError) as e:   # bad/empty JSON -> retry
            last_err = e
    raise RuntimeError(f'Model did not return valid JSON: {last_err}')


# --------------------------------------------------- optional cross-check
def crosscheck_seven_segment(path, llm_vals):
    """Run the old OpenCV reader and compare. Returns a list of review reasons."""
    try:
        import seven_segment
        from geometry import assign_readings
    except ImportError:
        return ['cross-check skipped: seven_segment.py / geometry.py not importable']

    img = cv2.imread(path)
    rows, _ = seven_segment.read_lcd(img)
    cands = [{'value': int(r['text']), 'box': r['box'],
              'conf': r['conf'], 'agreement': r['agreement']}
             for r in rows if seven_segment._plausible(r)]
    res = assign_readings(cands, {}, 0.0, img.shape[0])
    if not res['readings']:
        return ['cross-check: seven-segment reader found no valid triple']

    cv_vals = {'systolic': res['readings']['SYS']['value'],
               'diastolic': res['readings']['DIA']['value'],
               'pulse': res['readings']['PUL']['value']}
    diffs = [f'{k}: LLM={llm_vals[k]} vs OpenCV={cv_vals[k]}'
             for k in cv_vals if llm_vals.get(k) is not None and llm_vals[k] != cv_vals[k]]
    return [f'readers disagree ({"; ".join(diffs)})'] if diffs else []


# --------------------------------------------------------------- public API
def read_bp(path, crosscheck=False):
    """Main entry point. Returns a JSON-serialisable dict."""
    raw = read_with_llm(path)
    readings, needs_review, reasons = validate(raw)
    if crosscheck:
        extra = crosscheck_seven_segment(path, readings)
        reasons += extra
        needs_review = needs_review or bool(extra)
    return {
        'file': os.path.basename(path),
        'systolic': readings['systolic'],
        'diastolic': readings['diastolic'],
        'pulse': readings['pulse'],
        'needs_review': needs_review,
        'reasons': reasons,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--crosscheck', action='store_true')
    args = ap.parse_args()
    try:
        print(json.dumps(read_bp(args.image, args.crosscheck), indent=2))
    except Exception as e:
        print(json.dumps({'error': str(e)}, indent=2))
        sys.exit(1)