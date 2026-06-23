"""
Visual element detector that works independently of Vision API bbox detection.
Detects diagrams, tables, charts, rulers, etc. using pure image processing.
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def detect_visual_elements_in_region(
    image_bytes: bytes,
    question_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """
    Detect visual elements (diagrams, tables, charts, rulers) within a question region.

    This function analyzes the question region independently to find visual content,
    not relying on Vision API's has_image flag.

    Args:
        image_bytes: Full exam paper image bytes
        question_bbox: Normalized question region [x1, y1, x2, y2]

    Returns:
        List of normalized bboxes for detected visual elements
    """
    try:
        # Load image
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size

        # Convert question bbox to pixels
        x1, y1, x2, y2 = question_bbox
        px1 = max(0, int(x1 * width))
        py1 = max(0, int(y1 * height))
        px2 = min(width, int(x2 * width))
        py2 = min(height, int(y2 * height))

        if px2 <= px1 or py2 <= py1:
            return []

        # Convert to OpenCV format
        img_array = np.array(pil_image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Extract question region
        region = img_cv[py1:py2, px1:px2].copy()

        if region.size == 0 or region.shape[0] < 20 or region.shape[1] < 20:
            return []

        region_h, region_w = region.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Detect and remove colored pen marks (red/blue)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red pen (student/teacher marks)
        red_mask1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 60, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Blue pen
        blue_mask = cv2.inRange(hsv, np.array([100, 60, 50]), np.array([130, 255, 255]))

        # Combine and dilate to remove completely
        pen_mask = cv2.bitwise_or(red_mask, blue_mask)
        pen_mask = cv2.dilate(pen_mask, np.ones((7, 7), np.uint8), iterations=2)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=8
        )

        # Remove pen marks from binary
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(pen_mask))

        # Detect dense printed regions (likely diagrams/tables)
        # Use morphological operations to find connected regions
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Filter contours to find visual elements
        visual_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if too small (likely text or noise)
            min_area = region_w * region_h * 0.02  # At least 2% of question area
            if area < min_area or area < 500:
                continue

            # Skip if too large (likely the whole question)
            max_area = region_w * region_h * 0.85
            if area > max_area:
                continue

            # Skip if dimensions are too small
            if w < 60 or h < 30:
                continue

            # Check aspect ratio - skip extremely thin regions (likely lines of text)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 15 or aspect_ratio < 0.1:
                continue

            # Check density - visual elements should have moderate to high density
            roi = binary[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            white_pixels = cv2.countNonZero(roi)
            density = white_pixels / (w * h)

            # Visual elements typically have 15-70% density
            # Pure text lines have lower density
            # Solid filled shapes have very high density
            if density < 0.10 or density > 0.80:
                continue

            # Passed all filters - likely a visual element
            visual_candidates.append((x, y, w, h))

        if not visual_candidates:
            return []

        # Merge overlapping/nearby candidates
        merged = _merge_nearby_boxes(visual_candidates, region_w, region_h)

        # Convert back to full image normalized coordinates
        result_bboxes = []
        for (x, y, w, h) in merged:
            abs_x1 = px1 + x
            abs_y1 = py1 + y
            abs_x2 = px1 + x + w
            abs_y2 = py1 + y + h

            # Add padding to include annotations
            padding_x = max(20, int(width * 0.02))
            padding_y = max(20, int(height * 0.02))

            abs_x1 = max(0, abs_x1 - padding_x)
            abs_y1 = max(0, abs_y1 - padding_y)
            abs_x2 = min(width, abs_x2 + padding_x)
            abs_y2 = min(height, abs_y2 + padding_y)

            # Normalize
            norm_x1 = abs_x1 / width
            norm_y1 = abs_y1 / height
            norm_x2 = abs_x2 / width
            norm_y2 = abs_y2 / height

            result_bboxes.append((norm_x1, norm_y1, norm_x2, norm_y2))

        return result_bboxes

    except Exception:
        return []


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    region_w: int,
    region_h: int
) -> list[tuple[int, int, int, int]]:
    """Merge boxes that are close to each other."""
    if not boxes:
        return []

    # Convert to [x1, y1, x2, y2] format
    boxes_xyxy = [(x, y, x+w, y+h) for x, y, w, h in boxes]

    # Sort by y coordinate
    boxes_xyxy.sort(key=lambda b: b[1])

    merged = []
    current = list(boxes_xyxy[0])

    for box in boxes_xyxy[1:]:
        x1, y1, x2, y2 = current
        bx1, by1, bx2, by2 = box

        # Check if boxes are close enough to merge
        vertical_gap = by1 - y2
        horizontal_overlap = min(x2, bx2) - max(x1, bx1)

        # Merge if vertically close and horizontally overlapping
        if vertical_gap < 30 and horizontal_overlap > -50:
            current[0] = min(x1, bx1)
            current[1] = min(y1, by1)
            current[2] = max(x2, bx2)
            current[3] = max(y2, by2)
        else:
            merged.append(tuple(current))
            current = list(box)

    merged.append(tuple(current))

    # Convert back to [x, y, w, h] format
    return [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in merged]
