import cv2
from detector import FreeBPDetector
from geometry import (
    calculate_median_slope,
    fallback_expanding_box_matcher,
    match_labels_to_numbers,
)


def is_mapping_valid(mapped):
    """Checks that all 3 metrics are present and physiologically valid (SYS > DIA)."""
    if len(mapped) < 3:
        return False
    sys_val = mapped['SYS']['value']
    dia_val = mapped['DIA']['value']
    pul_val = mapped['PUL']['value']
    return (
        (sys_val > dia_val)
        and (80 <= sys_val <= 250)
        and (35 <= dia_val <= 150)
        and (35 <= pul_val <= 200)
    )


def run_pipeline(image_path, debug_output=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f'Could not load image at {image_path}. Check the file path!'
        )

    detector = FreeBPDetector()

    # Step 1: Detect labels, numbers, and raw OCR boxes
    labels, grouped_numbers, raw_ocr_boxes = detector.detect_all(image_path)
    slope = calculate_median_slope(raw_ocr_boxes)

    print(f'[INFO] Detected Labels: {list(labels.keys())}')
    print(
        f'[INFO] Estimated Device Tilt Slope (m): {slope:.4f}'
        f' (~{slope * 57.29:.1f} degrees)'
    )
    print(f'[INFO] Numbers Found: {[n["value"] for n in grouped_numbers]}')

    # Step 2: Primary Method -> Match labels to numbers using y = mx + c line projection
    mapped_metrics = match_labels_to_numbers(labels, grouped_numbers, slope=slope)
    expanded_debug_box = None

    # Step 3: Fallback Method -> Expanding Bounding Box around Labels
    if not is_mapping_valid(mapped_metrics):
        print(
            '[WARN] Primary ray-projection incomplete. Triggering Expanding'
            ' Bounding Box Fallback...'
        )
        fallback_metrics, expanded_debug_box = fallback_expanding_box_matcher(
            labels, grouped_numbers, image.shape
        )
        if fallback_metrics:
            for field, data in fallback_metrics.items():
                if field not in mapped_metrics or not is_mapping_valid(
                    mapped_metrics
                ):
                    mapped_metrics[field] = data

    # Step 4: Draw visual debugging overlay
    if debug_output:
        h, w, _ = image.shape

        # Draw all candidate numbers in thin Grey boxes
        for num in grouped_numbers:
            bx1, by1, bx2, by2 = num['box']
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (180, 180, 180), 1)

        # If the Expanding Box Fallback ran, draw the final Expanded Box in Purple
        if expanded_debug_box:
            ex1, ey1, ex2, ey2 = expanded_debug_box
            cv2.rectangle(image, (ex1, ey1), (ex2, ey2), (255, 0, 255), 2)
            cv2.putText(
                image,
                'Expanded Search Box',
                (ex1, max(ey1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        # Draw matched labels, full-width rays, and final numbers
        for field, data in mapped_metrics.items():
            nx1, ny1, nx2, ny2 = data['num_box']
            cv2.rectangle(image, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{field}: {data['value']}",
                (nx1, max(ny1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if data['label_box']:
                lx1, ly1, lx2, ly2 = data['label_box']
                cv2.rectangle(image, (lx1, ly1), (lx2, ly2), (255, 100, 0), 2)

            if data['line_params']:
                A, B, C = data['line_params']
                x_start, x_end = 10, w - 10
                y_start = int(A * x_start + C)
                y_end = int(A * x_end + C)
                cv2.line(
                    image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2
                )

        cv2.imwrite('debug_result.jpg', image)
        print('[INFO] Saved visual debug overlay to debug_result.jpg')

    final_output = {
        field: data['value'] for field, data in mapped_metrics.items()
    }
    return final_output


if __name__ == '__main__':
    readings = run_pipeline('images/20210818_20_59_05_468_000_ncpLhobRvRMSJp0cq6hCAghjbff1_F_3264_2448.jpg')
    print('\n--- FINAL BP READINGS ---')
    print(readings)
