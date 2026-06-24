import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR


QUESTION_PADDING = 16
TOP_LEVEL_NUMBER = re.compile(r"^\s*(\d{1,2})\s*[\.．、]")
CALCULATION_TITLE_PATTERN = re.compile(r"(?:计算|口算|列竖式|脱式|算一算|写出得数|得数)")
GRID_TITLE_PATTERN = re.compile(r"(?:填|比较|○|写出得数|得数)")
CALCULATION_EXPRESSION_PATTERN = re.compile(r"^[\d\s*+\-±×xX÷/＝=()（）.]+$")
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

    return _deduplicate_starts(candidates)


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
            question_end_y = _question_end_y(lines, start.line.y1, next_y)
            if question_end_y - start.line.y1 < 24:
                continue

            calculation_regions = _calculation_regions(
                start,
                lines,
                column_left,
                column_right,
                question_end_y,
                question_end_y == page_height,
            )
            if calculation_regions:
                regions.extend(calculation_regions)
                continue

            grid_regions = _form_grid_regions(
                start,
                lines,
                column_left,
                column_right,
                question_end_y,
                question_end_y == page_height,
            )
            if grid_regions:
                regions.extend(grid_regions)
                continue

            x = max(0, column_left - QUESTION_PADDING)
            y = max(0, start.line.y1 - QUESTION_PADDING)
            right = min(page_width, column_right + QUESTION_PADDING)
            bottom = page_height if question_end_y == page_height else max(start.line.y1 + 24, question_end_y - QUESTION_PADDING)
            box = {"x": x, "y": y, "width": right - x, "height": bottom - y}
            question_text = _text_in_box(lines, column_left, column_right, start.line.y1, question_end_y)
            regions.append(
                {
                    "question_no": start.question_no,
                    "question_text": question_text,
                    "question_box": box,
                }
            )
    return regions


def _question_end_y(lines: list[OcrLine], start_y: int, next_y: int) -> int:
    section_y = min(
        (
            line.y1
            for line in lines
            if start_y < line.y1 < next_y and SECTION_TITLE_PATTERN.match(line.text)
        ),
        default=next_y,
    )
    return section_y


def _calculation_regions(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    reaches_page_bottom: bool,
) -> list[dict]:
    if not CALCULATION_TITLE_PATTERN.search(start.line.text):
        return []

    expression_lines = [
        line
        for line in lines
        if column_left <= line.center_x < column_right
        and start.line.y1 < line.y1 < question_end_y
        and _is_calculation_expression(line.text)
    ]
    rows = [row for row in _cluster_rows(expression_lines) if len(row) >= 2]
    expressions = [line for row in rows for line in row]
    if len(expressions) < 2:
        return []

    columns = _cluster_columns(expressions, column_right - column_left)
    if len(columns) < 2:
        return []

    return _regions_from_grid(
        start,
        lines,
        column_left,
        column_right,
        question_end_y,
        reaches_page_bottom,
        rows,
        columns,
        fill_missing_cells=False,
    )


def _form_grid_regions(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    reaches_page_bottom: bool,
) -> list[dict]:
    if not GRID_TITLE_PATTERN.search(start.line.text):
        return []

    content_lines = [
        line
        for line in lines
        if column_left <= line.center_x < column_right
        and start.line.y1 + 28 < line.y1 < question_end_y
        and not SECTION_TITLE_PATTERN.match(line.text)
    ]
    rows = [row for row in _cluster_rows(content_lines, tolerance=30) if len(row) >= 2]
    if len(rows) < 2:
        return []

    base_row = max(rows, key=lambda row: len(_cluster_columns(row, column_right - column_left, minimum_gap_ratio=0.18)))
    columns = _cluster_columns(base_row, column_right - column_left, minimum_gap_ratio=0.18)
    first_row_columns = _cluster_columns(rows[0], column_right - column_left, minimum_gap_ratio=0.18)
    first_row_y = min(line.y1 for line in rows[0])
    second_row_y = min(line.y1 for line in rows[1])
    if len(columns) < 2 or len(first_row_columns) != len(columns) or second_row_y - first_row_y > 90:
        return []

    return _regions_from_grid(
        start,
        lines,
        column_left,
        column_right,
        question_end_y,
        reaches_page_bottom,
        rows,
        columns,
        fill_missing_cells=True,
    )


def _regions_from_grid(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    reaches_page_bottom: bool,
    rows: list[list[OcrLine]],
    columns: list[list[OcrLine]],
    fill_missing_cells: bool,
) -> list[dict]:

    regions = []
    question_index = 1
    row_starts = [min(line.y1 for line in row) for row in rows]
    column_centers = [sum(line.center_x for line in column) // len(column) for column in columns]
    column_edges = [column_left]
    column_edges.extend((left + right) // 2 for left, right in zip(column_centers, column_centers[1:]))
    column_edges.append(column_right)

    for row_index, row in enumerate(rows):
        row_end_y = row_starts[row_index + 1] if row_index + 1 < len(rows) else question_end_y
        row_columns = (
            range(len(column_centers))
            if fill_missing_cells
            else [
                min(range(len(column_centers)), key=lambda index: abs(expression.center_x - column_centers[index]))
                for expression in sorted(row, key=lambda line: line.x1)
            ]
        )
        for column_index in row_columns:
            left = column_edges[column_index]
            right = column_edges[column_index + 1]
            top = max(start.line.y1, row_starts[row_index] - QUESTION_PADDING)
            bottom = row_end_y if reaches_page_bottom and row_index + 1 == len(rows) else max(top + 24, row_end_y - QUESTION_PADDING)
            box = {"x": left, "y": top, "width": right - left, "height": bottom - top}
            regions.append(
                {
                    "question_no": f"{start.question_no}.{question_index}",
                    "question_text": _text_in_box(lines, left, right, row_starts[row_index], row_end_y),
                    "question_box": box,
                }
            )
            question_index += 1
    return regions


def _is_calculation_expression(text: str) -> bool:
    normalized = text.strip()
    return (
        not normalized.startswith(("=", "＝"))
        and
        (normalized[0].isdigit() or normalized.startswith(("*", "(", "（")))
        and
        bool(CALCULATION_EXPRESSION_PATTERN.fullmatch(normalized))
        and sum(character.isdigit() for character in normalized) >= 2
        and any(symbol in normalized for symbol in "+-*±×xX÷/")
    )


def _cluster_rows(lines: list[OcrLine], tolerance: int = 24) -> list[list[OcrLine]]:
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.y1, item.x1)):
        if rows and line.y1 - min(item.y1 for item in rows[-1]) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


def _cluster_columns(lines: list[OcrLine], column_width: int, minimum_gap_ratio: float = 0.12) -> list[list[OcrLine]]:
    columns: list[list[OcrLine]] = []
    minimum_gap = max(80, int(column_width * minimum_gap_ratio))
    for line in sorted(lines, key=lambda item: item.center_x):
        if columns and line.center_x - sum(item.center_x for item in columns[-1]) // len(columns[-1]) <= minimum_gap:
            columns[-1].append(line)
        else:
            columns.append([line])
    return columns


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
