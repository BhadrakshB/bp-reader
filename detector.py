import os
import re
import cv2
import numpy as np

# Disable PaddleX MKLDNN acceleration to prevent CPU PIR graph compilation crashes
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'

from paddleocr import PaddleOCR

LABEL_PATTERNS = {
    'SYS': re.compile(r'^(SYS|SYSTOLIC)', re.IGNORECASE),
    'DIA': re.compile(r'^(DIA|DIASTOLIC|DYS)', re.IGNORECASE),
    'PUL': re.compile(r'^(PUL|PULSE|P/MIN|BPM)', re.IGNORECASE),
}

# Words printed on the BP monitor casing that must never be converted into numbers
BLACKLIST_WORDS = {
    'SYS',
    'DIA',
    'PUL',
    'STOP',
    'START',
    'MEM',
    'ELKO',
    'MMHG',
    'MIN',
    'EL510',
}


class FreeBPDetector:

    def __init__(self):
        try:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang='en',
                enable_mkldnn=False,
            )
        except (ValueError, TypeError):
            self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)

    def _run_ocr_unified(self, img):
        """Supports both PaddleOCR 3.x (.predict) and 2.x (.ocr) formats."""
        parsed_lines = []
        if hasattr(self.ocr, 'predict'):
            results = self.ocr.predict(img)
            for res in results:
                data = res if isinstance(res, dict) else getattr(res, 'res', res)
                polys = data.get('dt_polys', data.get('rec_polys', []))
                texts = data.get('rec_texts', [])
                scores = data.get('rec_scores', [])
                for poly, text, score in zip(polys, texts, scores):
                    four_pt = (
                        poly.tolist() if hasattr(poly, 'tolist') else list(poly)
                    )
                    parsed_lines.append((four_pt, str(text), float(score)))
        else:
            results = self.ocr.ocr(img, cls=True)
            if results and results[0]:
                for line in results[0]:
                    parsed_lines.append((line[0], line[1][0], float(line[1][1])))
        return parsed_lines

    def preprocess_lcd_MultiScale(self, image):
        """Uses Local Adaptive Thresholding + light Morphological Closing so dark backgrounds
        don't ruin LCD screen contrast, without fusing top numbers into the screen border.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=10,
        )

        # Compact (3, 5) kernel bridges 7-segment gaps without fusing top digits into the bezel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        fused = cv2.bitwise_not(closed)
        return cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)

    def detect_all(self, image_path):
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise FileNotFoundError(f'Cannot read image at {image_path}')

        orig_h, orig_w = raw_img.shape[:2]

        # Pass 1: Raw image for printed labels (SYS, DIA, PUL)
        label_lines = self._run_ocr_unified(raw_img)

        detected_labels = {}
        all_four_point_boxes = []

        for four_pt, text, conf in label_lines:
            if conf < 0.4:
                continue
            all_four_point_boxes.append(four_pt)
            rect_box = self._to_rect(four_pt)
            clean_txt = re.sub(r'[^A-Za-z]', '', text).upper()

            for field, pattern in LABEL_PATTERNS.items():
                if pattern.search(clean_txt) and field not in detected_labels:
                    detected_labels[field] = rect_box

        # Pass 2: Preprocessed & Multi-Scale passes for giant 7-segment numbers
        lcd_img = self.preprocess_lcd_MultiScale(raw_img)
        cv2.imwrite('fused_lcd_debug.jpg', lcd_img)

        num_lines = []
        for scale in [1.0, 0.5, 0.35]:
            for src_img in [raw_img, lcd_img]:
                scaled_img = (
                    cv2.resize(src_img, (int(orig_w * scale), int(orig_h * scale)))
                    if scale != 1.0
                    else src_img
                )
                lines = self._run_ocr_unified(scaled_img)
                for four_pt, text, conf in lines:
                    rescaled_pt = [[pt[0] / scale, pt[1] / scale] for pt in four_pt]
                    num_lines.append((rescaled_pt, text, conf))

        # Extract valid numbers
        raw_number_candidates = []
        for four_pt, text, conf in num_lines:
            if conf < 0.30:
                continue

            raw_clean = re.sub(r'[^A-Za-z0-9|!]', '', text).upper()

            # 1. Block casing words first (so "SYS" or "STOP" never turn into numbers)
            if any(bw in raw_clean for bw in BLACKLIST_WORDS):
                continue

            # 2. Convert 7-segment vertical strokes ('I', 'L', '|', '!') into '1' BEFORE counting digits
            cleaned = (
                raw_clean.replace('I', '1')
                .replace('L', '1')
                .replace('|', '1')
                .replace('!', '1')
                .replace('O', '0')
                .replace('B', '8')
            )

            # 3. Verify that at least 65% of the cleaned string is valid digits
            actual_digits = sum(c.isdigit() for c in cleaned)
            if len(cleaned) == 0 or (actual_digits / len(cleaned)) < 0.65:
                continue

            digits_only = re.sub(r'\D', '', cleaned)

            if digits_only.isdigit():
                rect_box = self._to_rect(four_pt)
                bx1, by1, bx2, by2 = rect_box
                box_h = by2 - by1
                box_w = bx2 - bx1

                # Ignore tiny bezel numbers (e.g., 1, 2, 3 WHO scale or bottom clock)
                if detected_labels:
                    avg_label_h = np.mean(
                        [(b[3] - b[1]) for b in detected_labels.values()]
                    )
                    if box_h < (avg_label_h * 1.15):
                        continue

                # Split vertically merged stacked numbers if height > 1.5 * width
                if box_h > (box_w * 1.5) and len(digits_only) >= 4:
                    mid_y = int((by1 + by2) / 2)
                    top_v = int(digits_only[: len(digits_only) // 2])
                    bot_v = int(digits_only[len(digits_only) // 2 :])
                    if 35 <= top_v <= 260:
                        raw_number_candidates.append(
                            {'value': top_v, 'box': [bx1, by1, bx2, mid_y]}
                        )
                    if 35 <= bot_v <= 260:
                        raw_number_candidates.append(
                            {'value': bot_v, 'box': [bx1, mid_y, bx2, by2]}
                        )
                else:
                    val = int(digits_only)
                    if 35 <= val <= 260:
                        raw_number_candidates.append({'value': val, 'box': rect_box})

        # Deduplicate overlapping boxes (prioritizing larger boxes first)
        detected_numbers = []
        seen_boxes = []
        raw_number_candidates.sort(
            key=lambda n: (n['box'][3] - n['box'][1]), reverse=True
        )
        for n in raw_number_candidates:
            if not self._is_duplicate_box(n['box'], seen_boxes):
                seen_boxes.append(n['box'])
                detected_numbers.append(n)

        return detected_labels, detected_numbers, all_four_point_boxes

    @staticmethod
    def _to_rect(four_pt_box):
        xs = [pt[0] for pt in four_pt_box]
        ys = [pt[1] for pt in four_pt_box]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    @staticmethod
    def _is_duplicate_box(new_box, existing_boxes, iou_thresh=0.4):
        nx1, ny1, nx2, ny2 = new_box
        n_area = max(1, (nx2 - nx1) * (ny2 - ny1))
        for ex1, ey1, ex2, ey2 in existing_boxes:
            ix1, iy1 = max(nx1, ex1), max(ny1, ey1)
            ix2, iy2 = min(nx2, ex2), min(ny2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                e_area = max(1, (ex2 - ex1) * (ey2 - ey1))
                if (inter / min(n_area, e_area)) > iou_thresh:
                    return True
        return False
