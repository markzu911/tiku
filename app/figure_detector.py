"""
Smart figure detection and cropping service for exam questions.
Uses OpenCV to detect visual elements (diagrams, tables, charts, rulers, etc.)
and crops them with proper padding to include annotations.
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def detect_and_expand_visual_bbox(
    image_bytes: bytes,
    initial_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Intelligently expand a visual element bbox to include surrounding annotations.

    This function:
    1. Extracts the region around the initial bbox with generous padding
    2. Removes red/blue pen marks (student answers, corrections)
    3. Detects all printed content using adaptive thresholding
    4. Finds the actual content boundaries
    5. Returns expanded bbox with appropriate padding

    Args:
        image_bytes: Original full image bytes
        initial_bbox: Normalized bbox [x1, y1, x2, y2] from Vision API

    Returns:
        Expanded normalized bbox [x1, y1, x2, y2]
    """
    try:
        # Load image
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size

        # Convert PIL to OpenCV format
        img_array = np.array(pil_image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = initial_bbox
        px1 = int(x1 * width)
        py1 = int(y1 * height)
        px2 = int(x2 * width)
        py2 = int(y2 * height)

        # Validate initial bbox
        if px2 <= px1 or py2 <= py1:
            return initial_bbox

        # Extract initial region with generous padding for analysis
        # Use large padding to ensure we capture all nearby annotations
        analysis_padding_x = max(60, int(width * 0.08))  # At least 60px or 8%
        analysis_padding_y = max(60, int(height * 0.08))

        analysis_x1 = max(0, px1 - analysis_padding_x)
        analysis_y1 = max(0, py1 - analysis_padding_y)
        analysis_x2 = min(width, px2 + analysis_padding_x)
        analysis_y2 = min(height, py2 + analysis_padding_y)

        region = img_cv[analysis_y1:analysis_y2, analysis_x1:analysis_x2]

        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return _apply_fallback_padding(initial_bbox, 0.05)

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold to detect all content including light text
        # Using Gaussian adaptive threshold for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=21, C=12
        )

        # Remove red/blue pen marks (student answers and corrections)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red pen mask (two ranges for red in HSV because red wraps around)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Blue pen mask
        blue_mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))

        # Combine pen masks
        pen_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Dilate pen mask to ensure complete removal
        pen_kernel = np.ones((7, 7), np.uint8)
        pen_mask = cv2.dilate(pen_mask, pen_kernel, iterations=2)

        # Remove pen marks from binary image
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(pen_mask))

        # Morphological operations to connect nearby elements (annotations to figures)
        # Horizontal connection (for axis labels, dimension marks)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        # Vertical connection (for legends, captions)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        # General closing
        kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Close gaps to connect related elements
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_square)

        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return _apply_fallback_padding(initial_bbox, 0.05)

        # Filter contours by area - remove very small noise
        region_area = region.shape[0] * region.shape[1]
        min_area = max(50, region_area * 0.0001)  # At least 0.01% of region area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid_contours:
            return _apply_fallback_padding(initial_bbox, 0.05)

        # Get bounding rectangle that encompasses all valid contours
        all_points = np.vstack([c.reshape(-1, 2) for c in valid_contours])
        content_x, content_y, content_w, content_h = cv2.boundingRect(all_points)

        # Convert back to full image coordinates
        detected_px1 = analysis_x1 + content_x
        detected_py1 = analysis_y1 + content_y
        detected_px2 = analysis_x1 + content_x + content_w
        detected_py2 = analysis_y1 + content_y + content_h

        # Add final safety padding (smaller since we already detected actual content)
        final_padding_x = max(12, int(width * 0.012))  # 1.2% or 12px
        final_padding_y = max(12, int(height * 0.012))

        final_px1 = max(0, detected_px1 - final_padding_x)
        final_py1 = max(0, detected_py1 - final_padding_y)
        final_px2 = min(width, detected_px2 + final_padding_x)
        final_py2 = min(height, detected_py2 + final_padding_y)

        # Convert back to normalized coordinates
        final_x1 = final_px1 / width
        final_y1 = final_py1 / height
        final_x2 = final_px2 / width
        final_y2 = final_py2 / height

        # Sanity check: ensure expanded bbox is reasonable
        expanded_width = final_x2 - final_x1
        expanded_height = final_y2 - final_y1

        # If expansion is too large (more than 50% of image), use fallback
        if expanded_width > 0.6 or expanded_height > 0.6:
            return _apply_fallback_padding(initial_bbox, 0.05)

        return (final_x1, final_y1, final_x2, final_y2)

    except Exception:
        # If anything fails, return initial bbox with generous padding
        return _apply_fallback_padding(initial_bbox, 0.05)


def _apply_fallback_padding(
    bbox: tuple[float, float, float, float],
    padding_percent: float
) -> tuple[float, float, float, float]:
    """Apply symmetric padding to a bbox as fallback."""
    x1, y1, x2, y2 = bbox
    final_x1 = max(0.0, x1 - padding_percent)
    final_y1 = max(0.0, y1 - padding_percent)
    final_x2 = min(1.0, x2 + padding_percent)
    final_y2 = min(1.0, y2 + padding_percent)
    return (final_x1, final_y1, final_x2, final_y2)
