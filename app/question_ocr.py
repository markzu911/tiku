import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR


QUESTION_PADDING = 16
TOP_LEVEL_NUMBER = re.compile(r"^\s*(\d{1,2})\s*[\.．、]")
SUBQUESTION_NUMBER = re.compile(r"^\s*[（(]\s*(\d{1,2})\s*[）)]")
SECTION_TITLE_PATTERN = re.compile(
    r"^\s*[一二三四五六七八九十]+\s*[、．\.]\s*(填空|选择|判断|计算|应用|解决|口算|列式|简答|操作|画图|作图|连线|看图|实践|思考|阅读|写作)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrLine:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2


@dataclass(frozen=True)
class QuestionStart:
    question_no: str
    kind: str
    line: OcrLine


@lru_cache(maxsize=1)
def _ocr_engine() -> RapidOCR:
    return RapidOCR()


def extract_question_regions(image_bytes: bytes, debug_path: Path) -> list[dict]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    lines = _ocr_lines(image_bytes)
    starts = _question_starts(lines, width)
    if not starts:
        _save_debug_image(image, [], debug_path)
        return []

    regions = _build_question_regions(starts, lines, width, height)
    _save_debug_image(image, regions, debug_path)
    return regions


def _ocr_lines(image_bytes: bytes) -> list[OcrLine]:
    result, _ = _ocr_engine()(image_bytes)
    lines = []
    for row in result or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        points, text = row[0], str(row[1]).strip()
        if not text or not isinstance(points, (list, tuple)):
            continue
        coordinates = [coordinate for point in points for coordinate in point]
        if len(coordinates) < 8:
            continue
        lines.append(
            OcrLine(
                text=text,
                x1=int(min(coordinates[::2])),
                y1=int(min(coordinates[1::2])),
                x2=int(max(coordinates[::2])),
                y2=int(max(coordinates[1::2])),
            )
        )
    return sorted(lines, key=lambda line: (line.y1, line.x1))


def _question_starts(lines: list[OcrLine], page_width: int) -> list[QuestionStart]:
    candidates = []
    for line in lines:
        if SECTION_TITLE_PATTERN.match(line.text):
            continue
        if match := TOP_LEVEL_NUMBER.match(line.text):
            candidates.append(QuestionStart(match.group(1), "top", line))
        elif match := SUBQUESTION_NUMBER.match(line.text):
            candidates.append(QuestionStart(f"({match.group(1)})", "sub", line))

    top_level = [
        candidate
        for candidate in candidates
        if candidate.kind == "top" and candidate.line.x1 <= page_width * 0.25
    ]
    selected = top_level if len(top_level) >= 2 else candidates
    return _deduplicate_starts(selected)


def _deduplicate_starts(starts: list[QuestionStart]) -> list[QuestionStart]:
    deduplicated = []
    for start in sorted(starts, key=lambda item: (item.line.y1, item.line.x1)):
        if deduplicated and abs(start.line.y1 - deduplicated[-1].line.y1) < 14:
            continue
        deduplicated.append(start)
    return deduplicated


def _build_question_regions(
    starts: list[QuestionStart],
    lines: list[OcrLine],
    page_width: int,
    page_height: int,
) -> list[dict]:
    has_left = any(start.line.center_x < page_width * 0.42 for start in starts)
    has_right = any(start.line.center_x > page_width * 0.58 for start in starts)
    columns = [(0, page_width)]
    if has_left and has_right:
        middle = page_width // 2
        columns = [(0, middle), (middle, page_width)]

    regions = []
    for column_left, column_right in columns:
        column_width = column_right - column_left
        column_starts = [
            start
            for start in starts
            if column_left <= start.line.center_x < column_right
            and start.line.x1 <= column_left + column_width * 0.28
        ]
        column_starts.sort(key=lambda item: item.line.y1)
        for index, start in enumerate(column_starts):
            next_y = column_starts[index + 1].line.y1 if index + 1 < len(column_starts) else page_height
            if next_y - start.line.y1 < 24:
                continue

            x = max(0, column_left - QUESTION_PADDING)
            y = max(0, start.line.y1 - QUESTION_PADDING)
            right = min(page_width, column_right + QUESTION_PADDING)
            bottom = min(page_height, next_y + QUESTION_PADDING)
            box = {"x": x, "y": y, "width": right - x, "height": bottom - y}
            question_text = _text_in_box(lines, column_left, column_right, start.line.y1, next_y)
            regions.append(
                {
                    "question_no": start.question_no,
                    "question_text": question_text,
                    "question_box": box,
                }
            )
    return regions


def _text_in_box(
    lines: list[OcrLine],
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> str:
    selected = [
        line
        for line in lines
        if left <= line.center_x < right and top <= line.center_y < bottom
    ]
    return "\n".join(line.text for line in sorted(selected, key=lambda line: (line.y1, line.x1)))


def _save_debug_image(image: Image.Image, regions: list[dict], debug_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    for region in regions:
        box = region["question_box"]
        x, y = box["x"], box["y"]
        draw.rectangle((x, y, x + box["width"], y + box["height"]), outline="#2563eb", width=4)
        draw.text((x + 4, max(0, y - 16)), region["question_no"], fill="#2563eb")
    debug.save(debug_path, format="PNG")
