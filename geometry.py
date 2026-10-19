import math
import numpy as np


def calculate_median_slope(ocr_boxes):
    """Calculates the median tilt slope (m) from 4-point OCR bounding boxes."""
    slopes = []
    for box in ocr_boxes:
        pt1, pt2 = box[0], box[1]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        if abs(dx) > 5:
            slope = dy / dx
            if abs(slope) < 1.0:
                slopes.append(slope)

    return float(np.median(slopes)) if slopes else 0.0


def match_labels_to_numbers(labels, numbers, slope=0.0):
    """Projects a line (Ax + By + C = 0) with slope `m` both left and right from each
    label's center and matches the closest valid number on that row.
    """
    results = {}
    used_numbers = set()

    A = slope
    B = -1.0
    denom = math.sqrt(A**2 + B**2)

    for field in ['SYS', 'DIA', 'PUL']:
        if field not in labels:
            continue

        l_box = labels[field]
        x_L = (l_box[0] + l_box[2]) / 2.0
        y_L = (l_box[1] + l_box[3]) / 2.0
        C = y_L - (A * x_L)

        best_idx = None
        best_score = float('inf')

        for idx, num in enumerate(numbers):
            if idx in used_numbers:
                continue

            val = num['value']

            # Physiological range checks
            if field == 'SYS' and not (85 <= val <= 250):
                continue
            if field == 'DIA' and not (40 <= val <= 150):
                continue
            if field == 'PUL' and not (40 <= val <= 190):
                continue

            nx1, ny1, nx2, ny2 = num['box']
            x_N = (nx1 + nx2) / 2.0
            y_N = (ny1 + ny2) / 2.0
            num_height = ny2 - ny1

            # Perpendicular distance from number center to the label's projected line
            perp_dist = abs(A * x_N + B * y_N + C) / denom
            projected_y_at_number = (A * x_N) + C
            passes_through_box = ny1 <= projected_y_at_number <= ny2
            span_penalty = 0.0 if passes_through_box else (perp_dist * 1.5)

            horiz_dist = math.hypot(x_N - x_L, y_N - y_L)
            score = perp_dist + span_penalty + (horiz_dist * 0.1)

            # Strict 0.65x height tolerance so a label never steals a number from another row
            if perp_dist <= (num_height * 0.65) and score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_numbers.add(best_idx)
            results[field] = {
                'value': numbers[best_idx]['value'],
                'num_box': numbers[best_idx]['box'],
                'label_box': l_box,
                'line_params': (A, B, C),
            }

    return results


def fallback_expanding_box_matcher(labels, numbers, image_shape):
    """Fallback mechanism:
    1. Builds a seed bounding box around detected labels (SYS, DIA, PUL).
    2. Expands incrementally (Left -> Right -> Bottom -> Top) until >= 3 numbers are captured.
    3. Filters extras via font-height & physiological rules, and maps in spatial order.
    """
    if not numbers or len(numbers) < 3:
        return {}, None

    h, w = image_shape[:2]

    # Step 1: Create initial seed box from detected labels
    if labels:
        lx1 = min(box[0] for box in labels.values())
        ly1 = min(box[1] for box in labels.values())
        lx2 = max(box[2] for box in labels.values())
        ly2 = max(box[3] for box in labels.values())

        box_h = max(20, ly2 - ly1)
        if len(labels) == 1:
            if 'SYS' in labels:
                ly2 = min(h, ly2 + int(box_h * 3.5))
            elif 'DIA' in labels:
                ly1 = max(0, ly1 - int(box_h * 1.8))
                ly2 = min(h, ly2 + int(box_h * 1.8))
            elif 'PUL' in labels:
                ly1 = max(0, ly1 - int(box_h * 3.5))
        elif len(labels) == 2:
            if 'PUL' not in labels:
                ly2 = min(h, ly2 + int(box_h * 0.8))
            if 'SYS' not in labels:
                ly1 = max(0, ly1 - int(box_h * 0.8))
    else:
        lx1, ly1, lx2, ly2 = (
            int(w * 0.25),
            int(h * 0.25),
            int(w * 0.5),
            int(h * 0.75),
        )

    exp_box = [lx1, ly1, lx2, ly2]

    def get_captured_numbers(box):
        bx1, by1, bx2, by2 = box
        captured = []
        for idx, num in enumerate(numbers):
            nx1, ny1, nx2, ny2 = num['box']
            cx, cy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                captured.append((idx, num))
        return captured

    # Step 2: Expand incrementally (Left -> Right -> Bottom -> Top)
    step_x = max(10, int(w * 0.025))
    step_y = max(4, int(h * 0.008))

    captured = get_captured_numbers(exp_box)
    max_iters = 50

    for _ in range(max_iters):
        if len(captured) >= 3:
            break

        # 1. Expand Left
        exp_box[0] = max(0, exp_box[0] - step_x)
        captured = get_captured_numbers(exp_box)
        if len(captured) >= 3:
            break

        # 2. Expand Right
        exp_box[2] = min(w, exp_box[2] + step_x)
        captured = get_captured_numbers(exp_box)
        if len(captured) >= 3:
            break

        # 3. Expand Bottom
        exp_box[3] = min(h, exp_box[3] + step_y)
        captured = get_captured_numbers(exp_box)
        if len(captured) >= 3:
            break

        # 4. Expand Top
        exp_box[1] = max(0, exp_box[1] - step_y)
        captured = get_captured_numbers(exp_box)
        if len(captured) >= 3:
            break

    if len(captured) < 3:
        return {}, exp_box

    # Step 3: If > 3 numbers were captured, keep the 3 tallest numbers
    if len(captured) > 3:
        captured.sort(
            key=lambda item: (item[1]['box'][3] - item[1]['box'][1]),
            reverse=True,
        )
        captured = captured[:3]

    chosen_nums = [item[1] for item in captured]

    # Step 4: Determine ordering axis (Vertical vs Horizontal layout)
    xs = [(n['box'][0] + n['box'][2]) / 2.0 for n in chosen_nums]
    ys = [(n['box'][1] + n['box'][3]) / 2.0 for n in chosen_nums]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)

    if y_spread >= x_spread:
        chosen_nums.sort(key=lambda n: (n['box'][1] + n['box'][3]) / 2.0)
    else:
        chosen_nums.sort(key=lambda n: (n['box'][0] + n['box'][2]) / 2.0)

    expected_order = ['SYS', 'DIA', 'PUL']
    if len(labels) == 3:
        if y_spread >= x_spread:
            expected_order = sorted(
                labels.keys(), key=lambda k: (labels[k][1] + labels[k][3]) / 2.0
            )
        else:
            expected_order = sorted(
                labels.keys(), key=lambda k: (labels[k][0] + labels[k][2]) / 2.0
            )

    # Step 5: Map and verify medical sanity (SYS > DIA)
    mapped = {}
    for field, num_obj in zip(expected_order, chosen_nums):
        mapped[field] = {
            'value': num_obj['value'],
            'num_box': num_obj['box'],
            'label_box': labels.get(field),
            'line_params': None,
        }

    if (
        'SYS' in mapped
        and 'DIA' in mapped
        and mapped['SYS']['value'] < mapped['DIA']['value']
    ):
        mapped['SYS']['value'], mapped['DIA']['value'] = (
            mapped['DIA']['value'],
            mapped['SYS']['value'],
        )
        mapped['SYS']['num_box'], mapped['DIA']['num_box'] = (
            mapped['DIA']['num_box'],
            mapped['SYS']['num_box'],
        )

    return mapped, exp_box
