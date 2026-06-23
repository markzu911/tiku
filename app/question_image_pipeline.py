"""Coordinate-traceable question visual extraction for worksheet images."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


QUESTION_NUMBER_PATTERN = re.compile(r"^\s*(?:\d{1,2}[.、．]|[一二三四五六七八九十]+[、.])")

# A question number begins at a line start and includes its delimiter. Sequence
# validation below discards decimal values and duplicated sub-question labels.
QUESTION_NUMBER_PATTERN = re.compile(
    r"^\s*(?:\d{1,2}\s*(?:[.\u3002\u3001\)]|\uff09)|\(\s*\d{1,2}\s*\)|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*[\u3001\u3002.])"
)
QUESTION_LABEL_PATTERN = re.compile(
    r"^\s*(?:\(\s*(\d{1,2})\s*\)|(\d{1,2})\s*(?:[.\u3002\u3001\)]|\uff09)|([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*[\u3001\u3002.])"
)


@dataclass(frozen=True)
class ImageDetectionConfig:
    """Tunable thresholds for coordinate-based visual extraction."""

    ocr_min_confidence: float = 0.35
    scan_border_ratio: float = 0.015
    scan_white_threshold: int = 220
    scan_min_white_ratio: float = 0.82
    scan_max_border_stddev: float = 35.0
    paper_min_area_ratio: float = 0.20
    paper_corner_extent_ratio: float = 0.40
    paper_min_opposite_edge_ratio: float = 0.35
    text_padding: int = 6
    question_padding: int = 24
    min_candidate_area_ratio: float = 0.03
    min_candidate_side: int = 40
    max_aspect_ratio: float = 10.0
    min_candidate_score: float = 0.55
    min_structure_score: float = 0.15
    merge_gap: int = 26
    anchor_column_gap_ratio: float = 0.18
    anchor_left_lane_ratio: float = 0.18
    anchor_max_number_gap: int = 3
    crop_padding: int = 4
    visual_context_top_padding: int = 24
    visual_context_bottom_padding: int = 12
    visual_context_left_padding: int = 12
    visual_context_right_padding: int = 0
    merge_overlap_ratio: float = 0.35
    merge_min_horizontal_overlap_ratio: float = 0.60
    structural_min_line_length_ratio: float = 0.55
    structural_hough_threshold: int = 20


DEFAULT_CONFIG = ImageDetectionConfig()


@dataclass(frozen=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def padded(self, padding: int, width: int, height: int) -> "PixelBox":
        return PixelBox(
            max(0, self.x1 - padding),
            max(0, self.y1 - padding),
            min(width, self.x2 + padding),
            min(height, self.y2 + padding),
        )


@dataclass(frozen=True)
class OcrLine:
    text: str
    box: PixelBox
    confidence: float


@dataclass(frozen=True)
class QuestionRegion:
    question_no: str
    box: PixelBox
    column: int


@dataclass(frozen=True)
class Candidate:
    box: PixelBox
    score: float
    kind: str
    reason: str = ""


@dataclass
class PaperContext:
    original: np.ndarray
    working: np.ndarray
    original_bytes: bytes
    original_mime_type: str
    paper_quad: np.ndarray | None
    inverse_transform: np.ndarray
    is_scanned: bool

    @property
    def working_size(self) -> tuple[int, int]:
        return self.working.shape[1], self.working.shape[0]

    @property
    def original_size(self) -> tuple[int, int]:
        return self.original.shape[1], self.original.shape[0]


@dataclass
class VisualDetectionResult:
    question_regions: list[QuestionRegion] = field(default_factory=list)
    visuals_by_question: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    rejected_by_question: dict[int, list[Candidate]] = field(default_factory=dict)
    ocr_available: bool = False


def prepare_paper_context(
    image_bytes: bytes, config: ImageDetectionConfig = DEFAULT_CONFIG
) -> PaperContext:
    """Prepare an OCR working image while retaining the original image for final crops."""
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    original = np.array(image)
    original_bytes = _encode_png(original)
    height, width = original.shape[:2]
    identity = np.eye(3, dtype=np.float32)

    if _looks_scanned(original, config):
        return PaperContext(original, original, original_bytes, "image/png", None, identity, True)

    quad = _find_paper_quad(original, config)
    if quad is None:
        return PaperContext(original, original, original_bytes, "image/png", None, identity, False)

    destination, target_size = _destination_quad(quad)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    working = cv2.warpPerspective(
        original,
        transform,
        target_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return PaperContext(
        original,
        working,
        original_bytes,
        "image/png",
        quad,
        np.linalg.inv(transform),
        False,
    )


def working_image_bytes(context: PaperContext) -> bytes:
    return _encode_png(context.working)


def run_ocr(image_bytes: bytes) -> list[OcrLine]:
    """Return OCR lines with pixel boxes; return an empty list when OCR is unavailable."""
    try:
        result, _ = _ocr_engine()(image_bytes)
    except Exception:
        return []

    lines: list[OcrLine] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]).strip(), item[2]
        if not text or not isinstance(points, (list, tuple)) or len(points) < 4:
            continue
        try:
            coordinates = np.array(points, dtype=float)
            box = PixelBox(
                int(np.floor(coordinates[:, 0].min())),
                int(np.floor(coordinates[:, 1].min())),
                int(np.ceil(coordinates[:, 0].max())),
                int(np.ceil(coordinates[:, 1].max())),
            )
            lines.append(OcrLine(text, box, float(confidence)))
        except (TypeError, ValueError):
            continue
    return lines


def detect_question_regions(
    ocr_lines: list[OcrLine],
    width: int,
    height: int,
    config: ImageDetectionConfig = DEFAULT_CONFIG,
) -> list[QuestionRegion]:
    anchors = [
        line
        for line in ocr_lines
        if line.confidence >= config.ocr_min_confidence and QUESTION_NUMBER_PATTERN.match(line.text)
    ]
    if not anchors:
        return []

    columns = _anchor_columns(anchors, width, config)
    regions: list[QuestionRegion] = []

    for column_index, column_anchors in enumerate(columns):
        ordered = _clean_question_anchors(column_anchors, width, config)
        if not ordered:
            continue
        x1, x2 = (0, width // 2) if len(columns) == 2 and column_index == 0 else (width // 2, width)
        if len(columns) == 1:
            x1, x2 = 0, width
        for index, anchor in enumerate(ordered):
            next_y = ordered[index + 1].box.y1 if index + 1 < len(ordered) else height
            top = max(0, anchor.box.y1 - config.question_padding)
            bottom = min(height, max(top + config.min_candidate_side, next_y + config.question_padding))
            regions.append(QuestionRegion(_question_label(anchor.text), PixelBox(x1, top, x2, bottom), column_index))

    return regions


def detect_question_visuals(
    context: PaperContext,
    ocr_lines: list[OcrLine],
    question_count: int,
    config: ImageDetectionConfig = DEFAULT_CONFIG,
) -> VisualDetectionResult:
    width, height = context.working_size
    regions = detect_question_regions(ocr_lines, width, height, config)
    result = VisualDetectionResult(question_regions=regions, ocr_available=bool(ocr_lines))
    if not regions:
        return result

    for index, region in enumerate(regions[:question_count]):
        region_lines = [line for line in ocr_lines if _overlap_ratio(line.box, region.box) >= 0.5]
        candidates, rejected = detect_image_candidates(context.working, region.box, region_lines, config)
        result.rejected_by_question[index] = rejected
        visuals: list[dict[str, Any]] = []
        for candidate in candidates[:3]:
            original_box = map_box_to_original_image(candidate.box, context, config)
            if original_box is None:
                continue
            visuals.append(
                {
                    "kind": candidate.kind,
                    "bbox": _normalize_box(original_box, *context.original_size),
                    "source": "auto_detect",
                    "confidence": round(candidate.score, 3),
                    "working_box": _box_dict(candidate.box),
                    "crop_box": _box_dict(original_box),
                }
            )
        result.visuals_by_question[index] = visuals
    return result


def detect_image_candidates(
    image: np.ndarray,
    question_box: PixelBox,
    ocr_lines: list[OcrLine],
    config: ImageDetectionConfig = DEFAULT_CONFIG,
) -> tuple[list[Candidate], list[Candidate]]:
    region = image[question_box.y1:question_box.y2, question_box.x1:question_box.x2]
    if region.size == 0:
        return [], []
    region_height, region_width = region.shape[:2]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    _mask_pen_marks(binary, region)
    text_mask = np.zeros_like(binary)
    for line in ocr_lines:
        local = PixelBox(
            line.box.x1 - question_box.x1,
            line.box.y1 - question_box.y1,
            line.box.x2 - question_box.x1,
            line.box.y2 - question_box.y1,
        ).padded(config.text_padding, region_width, region_height)
        text_mask[local.y1:local.y2, local.x1:local.x2] = 255
    binary[text_mask > 0] = 0

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (22, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 22)))
    structural = cv2.bitwise_or(horizontal, vertical)
    connected = cv2.bitwise_or(binary, structural)
    connected = cv2.morphologyEx(
        connected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted: list[Candidate] = []
    rejected: list[Candidate] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box = PixelBox(question_box.x1 + x, question_box.y1 + y, question_box.x1 + x + box_width, question_box.y1 + y + box_height)
        candidate = _score_candidate(
            box,
            question_box,
            text_mask[y:y + box_height, x:x + box_width],
            connected[y:y + box_height, x:x + box_width],
            config,
        )
        if candidate.score >= config.min_candidate_score:
            accepted.append(candidate)
        elif candidate.box.area >= question_box.area * 0.01:
            rejected.append(candidate)

    merged = _merge_candidates(accepted, config)
    verified = [
        candidate
        for candidate in (judge_candidate_with_ai(candidate, None) for candidate in merged)
        if candidate is not None
    ]
    return [_include_visual_context(candidate, question_box, config) for candidate in verified], rejected


def map_box_to_original_image(
    box: PixelBox,
    context: PaperContext,
    config: ImageDetectionConfig = DEFAULT_CONFIG,
) -> PixelBox | None:
    points = np.array(
        [[[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]], dtype=np.float32
    )
    mapped = cv2.perspectiveTransform(points, context.inverse_transform)[0]
    width, height = context.original_size
    left = max(0, int(np.floor(mapped[:, 0].min())) - config.crop_padding)
    top = max(0, int(np.floor(mapped[:, 1].min())) - config.crop_padding)
    right = min(width, int(np.ceil(mapped[:, 0].max())) + config.crop_padding)
    bottom = min(height, int(np.ceil(mapped[:, 1].max())) + config.crop_padding)
    return PixelBox(left, top, right, bottom) if right > left and bottom > top else None


def save_debug_image(context: PaperContext, result: VisualDetectionResult, directory: Path) -> str | None:
    if os.getenv("PAPER_IMAGE_DEBUG", "false").lower() not in {"1", "true", "yes"}:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(context.working.copy())
    draw = ImageDraw.Draw(image)
    width, height = context.working_size
    draw.rectangle((0, 0, width - 1, height - 1), outline="red", width=3)
    for index, region in enumerate(result.question_regions):
        draw.rectangle((region.box.x1, region.box.y1, region.box.x2, region.box.y2), outline="blue", width=2)
        draw.text((region.box.x1 + 4, region.box.y1 + 4), region.question_no or str(index + 1), fill="blue")
    for index, candidates in result.rejected_by_question.items():
        for candidate in candidates:
            draw.rectangle((candidate.box.x1, candidate.box.y1, candidate.box.x2, candidate.box.y2), outline="gold", width=2)
            if candidate.box.width >= 60 and candidate.box.height >= 30:
                label = _debug_question_label(result, index)
                draw.text(
                    (candidate.box.x1 + 3, candidate.box.y1 + 3),
                    f"{label}:{candidate.reason}",
                    fill="gold",
                )
    for index, visuals in result.visuals_by_question.items():
        for visual in visuals:
            box = visual.get("working_box", {})
            draw.rectangle((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]), outline="green", width=3)
            draw.text(
                (box["x"] + 4, box["y"] + 4),
                f"{_debug_question_label(result, index)}:{visual['confidence']}",
                fill="green",
            )
    filename = f"paper-detect-{uuid4().hex}.png"
    image.save(directory / filename, format="PNG")
    return filename


def _debug_question_label(result: VisualDetectionResult, index: int) -> str:
    if 0 <= index < len(result.question_regions):
        return result.question_regions[index].question_no or str(index + 1)
    return str(index + 1)


def _looks_scanned(image: np.ndarray, config: ImageDetectionConfig) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    border = max(4, int(min(width, height) * config.scan_border_ratio))
    edge_pixels = np.concatenate((gray[:border, :].ravel(), gray[-border:, :].ravel(), gray[:, :border].ravel(), gray[:, -border:].ravel()))
    return (
        float(np.mean(edge_pixels > config.scan_white_threshold)) >= config.scan_min_white_ratio
        and float(np.std(edge_pixels)) <= config.scan_max_border_stddev
    )


def _find_paper_quad(image: np.ndarray, config: ImageDetectionConfig) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < image_area * config.paper_min_area_ratio:
            continue
        approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approximation) != 4:
            continue
        quad = _order_quad(approximation.reshape(4, 2).astype(np.float32))
        if _valid_paper_quad(quad, image.shape[1], image.shape[0], config):
            return quad
    return None


def _order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _valid_paper_quad(
    quad: np.ndarray,
    image_width: int,
    image_height: int,
    config: ImageDetectionConfig,
) -> bool:
    tl, tr, br, bl = quad
    normalized = quad / np.array([image_width, image_height], dtype=np.float32)
    if not (
        normalized[0, 0] <= config.paper_corner_extent_ratio and normalized[0, 1] <= config.paper_corner_extent_ratio
        and normalized[1, 0] >= 1 - config.paper_corner_extent_ratio and normalized[1, 1] <= config.paper_corner_extent_ratio
        and normalized[2, 0] >= 1 - config.paper_corner_extent_ratio and normalized[2, 1] >= 1 - config.paper_corner_extent_ratio
        and normalized[3, 0] <= config.paper_corner_extent_ratio and normalized[3, 1] >= 1 - config.paper_corner_extent_ratio
    ):
        return False
    top, right, bottom, left = np.linalg.norm(tr - tl), np.linalg.norm(br - tr), np.linalg.norm(bl - br), np.linalg.norm(tl - bl)
    if min(top, right, bottom, left) < math_hypot(image_width, image_height) * 0.05:
        return False
    return (
        min(top, bottom) / max(top, bottom) >= config.paper_min_opposite_edge_ratio
        and min(left, right) / max(left, right) >= config.paper_min_opposite_edge_ratio
    )


def _destination_quad(quad: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    tl, tr, br, bl = quad
    width = max(1, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = max(1, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32), (width, height)


def _mask_pen_marks(binary: np.ndarray, region: np.ndarray) -> None:
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 60, 45]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 60, 45]), np.array([180, 255, 255])),
    )
    blue = cv2.inRange(hsv, np.array([100, 60, 45]), np.array([130, 255, 255]))
    pen = cv2.dilate(cv2.bitwise_or(red, blue), np.ones((5, 5), np.uint8), iterations=1)
    binary[pen > 0] = 0


def _score_candidate(
    box: PixelBox,
    question_box: PixelBox,
    text_mask: np.ndarray,
    content: np.ndarray,
    config: ImageDetectionConfig,
) -> Candidate:
    if box.width < config.min_candidate_side or box.height < config.min_candidate_side:
        return Candidate(box, 0.0, "image", "too_small")
    aspect = box.width / box.height
    if aspect > config.max_aspect_ratio or aspect < 1 / config.max_aspect_ratio:
        return Candidate(box, 0.0, "image", "line_like")
    if box.area < question_box.area * config.min_candidate_area_ratio:
        return Candidate(box, 0.0, "image", "area_too_small")
    text_overlap = float(np.count_nonzero(text_mask)) / max(1, box.area)
    if text_overlap > 0.5:
        return Candidate(box, 0.0, "image", "text_overlap")
    density = float(np.count_nonzero(content)) / max(1, box.area)
    if density < 0.005:
        return Candidate(box, 0.0, "image", "low_complexity")
    structure_score, kind = _structural_line_score(content, config)
    if structure_score < config.min_structure_score:
        return Candidate(box, 0.0, kind, "not_a_structured_visual")
    score = min(
        0.95,
        0.35
        + min(0.22, box.area / max(1, question_box.area))
        + min(0.18, density)
        + structure_score
        + (0.08 if text_overlap < 0.1 else 0),
    )
    return Candidate(box, score, kind)


def _merge_candidates(candidates: list[Candidate], config: ImageDetectionConfig) -> list[Candidate]:
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.box.area, reverse=True):
        matching_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _overlap_ratio(candidate.box, existing.box) > config.merge_overlap_ratio
                or _nearby_visual_fragments(candidate.box, existing.box, config)
            ),
            None,
        )
        if matching_index is not None:
            existing = merged[matching_index]
            union = PixelBox(
                min(candidate.box.x1, existing.box.x1),
                min(candidate.box.y1, existing.box.y1),
                max(candidate.box.x2, existing.box.x2),
                max(candidate.box.y2, existing.box.y2),
            )
            merged[matching_index] = Candidate(
                union,
                max(candidate.score, existing.score),
                "table" if "table" in {candidate.kind, existing.kind} else "diagram",
            )
            continue
        merged.append(candidate)
    return sorted(merged, key=lambda item: (item.box.y1, item.box.x1))


def _nearby_visual_fragments(
    left: PixelBox, right: PixelBox, config: ImageDetectionConfig
) -> bool:
    horizontal_overlap = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    horizontal_ratio = horizontal_overlap / max(1, min(left.width, right.width))
    vertical_gap = max(left.y1, right.y1) - min(left.y2, right.y2)
    return (
        horizontal_ratio >= config.merge_min_horizontal_overlap_ratio
        and -config.merge_gap <= vertical_gap <= 0
    )


def _include_visual_context(
    candidate: Candidate, question_box: PixelBox, config: ImageDetectionConfig
) -> Candidate:
    """Retain dimension labels and boundary strokes just outside a detected figure."""
    expanded = PixelBox(
        max(question_box.x1, candidate.box.x1 - config.visual_context_left_padding),
        max(question_box.y1, candidate.box.y1 - config.visual_context_top_padding),
        min(question_box.x2, candidate.box.x2 + config.visual_context_right_padding),
        min(question_box.y2, candidate.box.y2 + config.visual_context_bottom_padding),
    )
    return Candidate(expanded, candidate.score, candidate.kind, candidate.reason)


def judge_candidate_with_ai(
    candidate: Candidate, candidate_image: np.ndarray | None
) -> Candidate | None:
    """Optional verification hook; rules are the non-network fallback for now.

    The pipeline intentionally does not submit the full worksheet to an AI model.
    A future verifier may inspect only ``candidate_image`` and return ``None`` for
    a rejected visual while preserving the same coordinate contract.
    """
    del candidate_image
    return candidate


def _overlap_ratio(left: PixelBox, right: PixelBox) -> float:
    overlap_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    overlap_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return overlap_width * overlap_height / max(1, min(left.area, right.area))


def _normalize_box(box: PixelBox, width: int, height: int) -> list[float]:
    return [round(box.x1 / width, 6), round(box.y1 / height, 6), round(box.x2 / width, 6), round(box.y2 / height, 6)]


def _box_dict(box: PixelBox) -> dict[str, int]:
    return {"x": box.x1, "y": box.y1, "width": box.width, "height": box.height}


def _question_label(text: str) -> str:
    match = re.match(r"\s*(\(?\d{1,2}\)?|[一二三四五六七八九十]+)", text)
    return match.group(1) if match else ""


def _anchor_columns(anchors: list[OcrLine], width: int) -> list[list[OcrLine]]:
    ordered = sorted(anchors, key=lambda line: line.box.x1)
    clusters: list[list[OcrLine]] = []
    for anchor in ordered:
        if not clusters or anchor.box.x1 - np.median([line.box.x1 for line in clusters[-1]]) > width * 0.18:
            clusters.append([anchor])
        else:
            clusters[-1].append(anchor)
    significant = [cluster for cluster in clusters if len(cluster) >= 3]
    # A worksheet is either single-column or two-column. More clusters are OCR
    # noise from answer values, not a multi-column layout.
    return significant if len(significant) == 2 else [anchors]


def _question_label(text: str) -> str:
    match = QUESTION_LABEL_PATTERN.match(text)
    if not match:
        return ""
    return next((group for group in match.groups() if group), "")


def _question_number_value(text: str) -> int | None:
    label = _question_label(text)
    if label.isdigit():
        return int(label)
    chinese_values = {
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
    }
    if label in chinese_values:
        return chinese_values[label]
    if len(label) == 2 and label.startswith("\u5341") and label[1] in chinese_values:
        return 10 + chinese_values[label[1]]
    return None


def _clean_question_anchors(
    anchors: list[OcrLine], page_width: int, config: ImageDetectionConfig
) -> list[OcrLine]:
    """Keep a plausible question-number sequence and discard answer blanks/subitems."""
    left_edge = min(anchor.box.x1 for anchor in anchors)
    left_lane_limit = left_edge + page_width * config.anchor_left_lane_ratio
    cleaned: list[OcrLine] = []
    previous_value: int | None = None
    for anchor in sorted(anchors, key=lambda line: (line.box.y1, line.box.x1)):
        if anchor.box.x1 > left_lane_limit:
            continue
        value = _question_number_value(anchor.text)
        if value is None:
            continue
        if previous_value is None:
            cleaned.append(anchor)
            previous_value = value
            continue
        if previous_value < value <= previous_value + config.anchor_max_number_gap:
            cleaned.append(anchor)
            previous_value = value
    return cleaned


def _anchor_columns(
    anchors: list[OcrLine], width: int, config: ImageDetectionConfig
) -> list[list[OcrLine]]:
    ordered = sorted(anchors, key=lambda line: line.box.x1)
    clusters: list[list[OcrLine]] = []
    for anchor in ordered:
        if not clusters or anchor.box.x1 - np.median([line.box.x1 for line in clusters[-1]]) > width * config.anchor_column_gap_ratio:
            clusters.append([anchor])
        else:
            clusters[-1].append(anchor)
    significant = [cluster for cluster in clusters if len(cluster) >= 3]
    return significant if len(significant) == 2 else [anchors]


def _structural_line_score(
    content: np.ndarray, config: ImageDetectionConfig
) -> tuple[float, str]:
    height, width = content.shape[:2]
    minimum_length = max(36, int(max(width, height) * config.structural_min_line_length_ratio))
    lines = cv2.HoughLinesP(
        content,
        1,
        np.pi / 180,
        threshold=config.structural_hough_threshold,
        minLineLength=minimum_length,
        maxLineGap=8,
    )
    if lines is None:
        return 0.0, "diagram"
    horizontal = 0
    vertical = 0
    diagonal = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle <= 12 or angle >= 168:
            horizontal += 1
        elif 78 <= angle <= 102:
            vertical += 1
        else:
            diagonal += 1
    line_count = horizontal + vertical + diagonal
    if horizontal >= 3 and vertical >= 3:
        return min(0.25, 0.12 + line_count * 0.02), "table"
    if line_count >= 4 and (horizontal + vertical >= 3 or diagonal >= 3):
        return min(0.22, 0.08 + line_count * 0.02), "diagram"
    return 0.0, "diagram"


def _encode_png(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def math_hypot(width: int, height: int) -> float:
    return float(np.hypot(width, height))


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()
