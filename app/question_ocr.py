import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR


QUESTION_PADDING = 16
QUESTION_BOTTOM_GAP = 8
TOP_LEVEL_NUMBER = re.compile(r"^\s*(\d{1,2})\s*(?:[．、]|[.](?!\d))")
BARE_TOP_LEVEL_NUMBER = re.compile(r"^\s*(\d{1,2})\s+(?=\S)")
OPTION_LINE_PATTERN = re.compile(r"^\s*[A-D]\s*[\.．、)）]")
CHOICE_OPTION_PATTERN = re.compile(r"^\s*[A-D]\s*[\.．、)）]")
ANSWER_BLANK_PATTERN = re.compile(r"[（(]\s*[）)]")
QUESTION_START_HINT = re.compile(
    r"[（(]\s*[）)]|(?:下面|下列|选项|正确|错误|表示|算式|图形|钟面|时刻|方法|路线|最大|最小|共有|多少|哪|什么)"
)
CALCULATION_TITLE_PATTERN = re.compile(r"(?:计算|口算|列竖式|脱式|算一算|写出得数|得数)")
GRID_TITLE_PATTERN = re.compile(r"(?:填|比较|○|写出得数|得数)")
CALCULATION_EXPRESSION_PATTERN = re.compile(r"^[\d\s*+\-±×xX÷/＝=()（）.≈≒]+$")
# 比较类题目标题关键词（比大小、比一比、在○里填 >/<）
_COMPARISON_TITLE_KEYWORD = re.compile(r"(?:比大小|比一比|比较大小|比较|在[○〇]里填)")
# 匹配单个算式，用于拆分被 OCR 合并的行  e.g. "2.4-0.2= 409÷7≈ 655÷5="
SINGLE_EXPRESSION = re.compile(r"[\d.]+[\s]*[+\-×xX÷/±][\s]*[\d.]+[\s]*[≈≒＝=]")
# 匹配含 = 或 ≈ 的算式行，用于纯文本拆分兜底
EXPRESSION_LINE = re.compile(r"[\d.]+[\s]*[+\-×÷±][\s]*[\d.]+[\s]*[≈≒＝=]")
SECTION_TITLE_PATTERN = re.compile(
    r"^\s*[一二三四五六七八九十]+\s*[、．\.]\s*(填空|选择|判断|计算|应用|解决|口算|列式|简答|操作|画图|作图|连线|看图|实践|思考|阅读|写作)",
    re.IGNORECASE,
)
ARABIC_SECTION_TITLE_PATTERN = re.compile(
    r"^\s*\d{1,2}\s*(?:[．、]|[.](?!\d))\s*"
    r"(?:直接写出得数|写出得数|细心计算|计算|口算|用竖式计算|列竖式计算|脱式计算|递等式计算)"
    r"(?:[。．.：:；;、\s]|$)"
)
SUB_QUESTION_PATTERN = re.compile(r"^\s*[(（]\s*(\d{1,2})\s*[)）]")
CIRCLED_NUMBER = re.compile(r"[①-⑳]")
SUB_NUMBER_WITH_PAREN = re.compile(r"^\s*(\d{1,2})\s*[)）]")


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
        regions = [_fallback_page_region(lines, width, height)]
        _save_debug_image(image, regions, debug_path)
        return regions

    regions = _build_question_regions(starts, lines, width, height)
    if not regions:
        regions = [_fallback_page_region(lines, width, height)]
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
    return _split_merged_expression_lines(sorted(lines, key=lambda line: (line.y1, line.x1)))


def _split_merged_expression_lines(lines: list[OcrLine]) -> list[OcrLine]:
    """拆分被 OCR 合并的同行多算式，并保留题号前缀"""
    result: list[OcrLine] = []
    for line in lines:
        matches = list(SINGLE_EXPRESSION.finditer(line.text))
        if len(matches) < 2:
            result.append(line)
            continue

        total_width = line.x2 - line.x1
        text_len = max(len(line.text), 1)

        # 保留第一个算式之前的题号前缀（如 "1. "），避免题号丢失
        prefix = line.text[:matches[0].start()].strip()
        if prefix and TOP_LEVEL_NUMBER.match(prefix):
            prefix_end_x = line.x1 + int(total_width * (matches[0].start() / text_len))
            result.append(OcrLine(
                text=prefix,
                x1=line.x1, y1=line.y1,
                x2=prefix_end_x, y2=line.y2,
            ))

        # 拆分每个算式为独立 OcrLine
        for match in matches:
            ratio_start = match.start() / text_len
            ratio_end = match.end() / text_len
            result.append(OcrLine(
                text=match.group(),
                x1=line.x1 + int(total_width * ratio_start),
                y1=line.y1,
                x2=line.x1 + int(total_width * ratio_end),
                y2=line.y2,
            ))
    return result


def _question_starts(lines: list[OcrLine], page_width: int) -> list[QuestionStart]:
    candidates = []
    for line in lines:
        if _skip_question_start_line(line.text):
            continue
        if match := TOP_LEVEL_NUMBER.match(line.text):
            candidates.append(QuestionStart(match.group(1), "top", line))
            continue
        if match := BARE_TOP_LEVEL_NUMBER.match(line.text):
            if _looks_like_question_start(line.text):
                candidates.append(QuestionStart(match.group(1), "top", line))

    if candidates:
        return _add_missing_leading_start(_deduplicate_starts(candidates), lines, page_width)

    fallback_candidates = []
    for line in lines:
        if _skip_question_start_line(line.text):
            continue
        if match := SUB_QUESTION_PATTERN.match(line.text):
            fallback_candidates.append(QuestionStart(match.group(1), "sub", line))
            continue
        if match := SUB_NUMBER_WITH_PAREN.match(line.text):
            fallback_candidates.append(QuestionStart(match.group(1), "sub", line))
            continue
        if match := CIRCLED_NUMBER.match(line.text.strip()):
            fallback_candidates.append(QuestionStart(str(_circled_number_value(match.group(0))), "sub", line))

    return _deduplicate_starts(_filter_sequence_starts(fallback_candidates))


def _deduplicate_starts(starts: list[QuestionStart]) -> list[QuestionStart]:
    deduplicated = []
    for start in sorted(starts, key=lambda item: (item.line.y1, item.line.x1)):
        if deduplicated and abs(start.line.y1 - deduplicated[-1].line.y1) < 14:
            continue
        deduplicated.append(start)
    return deduplicated


def _add_missing_leading_start(
    starts: list[QuestionStart],
    lines: list[OcrLine],
    page_width: int,
) -> list[QuestionStart]:
    if not starts:
        return starts

    starts = sorted(starts, key=lambda item: (item.line.y1, item.line.x1))
    first = starts[0]
    first_number = _safe_int(first.question_no)
    if first.kind != "top" or first_number <= 1:
        return starts

    prior_lines = [
        line
        for line in lines
        if line.y1 < first.line.y1 - 12
        and not _is_section_title_line(line.text.strip())
        and line.text.strip()
    ]
    if not _looks_like_missing_leading_question(prior_lines):
        return starts

    anchor_candidates = [line for line in prior_lines if not CHOICE_OPTION_PATTERN.match(line.text.strip())]
    if not anchor_candidates:
        return starts
    anchor = min(anchor_candidates, key=lambda item: (item.y1, item.x1))
    synthetic = QuestionStart(str(first_number - 1), "top", anchor)
    return _deduplicate_starts([synthetic, *starts])


def _looks_like_missing_leading_question(lines: list[OcrLine]) -> bool:
    if not lines:
        return False
    option_count = sum(1 for line in lines if CHOICE_OPTION_PATTERN.match(line.text.strip()))
    text = "\n".join(line.text for line in lines)
    has_prompt = ANSWER_BLANK_PATTERN.search(text) is not None or any(
        keyword in text for keyword in ("表示", "是", "多少", "哪", "什么", "正确")
    )
    return option_count >= 2 and has_prompt


def _skip_question_start_line(text: str) -> bool:
    stripped = text.strip()
    return not stripped or SECTION_TITLE_PATTERN.match(stripped) is not None or OPTION_LINE_PATTERN.match(stripped) is not None


def _looks_like_question_start(text: str) -> bool:
    stripped = text.strip()
    if _is_calculation_expression(TOP_LEVEL_NUMBER.sub("", stripped, count=1)):
        return False
    if _is_calculation_expression(BARE_TOP_LEVEL_NUMBER.sub("", stripped, count=1)):
        return False
    return QUESTION_START_HINT.search(stripped) is not None


def _filter_sequence_starts(starts: list[QuestionStart]) -> list[QuestionStart]:
    if not starts:
        return starts

    numbers = [_safe_int(start.question_no) for start in starts]
    if 1 not in numbers:
        return starts

    filtered: list[QuestionStart] = []
    expected = 1
    for start, number in zip(starts, numbers):
        if number == expected:
            filtered.append(start)
            expected += 1
        elif number > expected and number <= expected + 2:
            filtered.append(start)
            expected = number + 1
        elif number == 0:
            filtered.append(start)
    return filtered or starts


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _circled_number_value(value: str) -> int:
    codepoint = ord(value)
    if ord("①") <= codepoint <= ord("⑳"):
        return codepoint - ord("①") + 1
    return 0


def _fallback_page_region(lines: list[OcrLine], page_width: int, page_height: int) -> dict:
    if lines:
        left = max(0, min(line.x1 for line in lines) - QUESTION_PADDING)
        top = max(0, min(line.y1 for line in lines) - QUESTION_PADDING)
        right = min(page_width, max(line.x2 for line in lines) + QUESTION_PADDING)
        bottom = min(page_height, max(line.y2 for line in lines) + QUESTION_PADDING)
        text = _strip_section_titles(_text_in_box(lines, left, right, top, bottom))
    else:
        left, top, right, bottom = 0, 0, page_width, page_height
        text = ""

    return {
        "question_no": "fallback",
        "question_text": text,
        "question_box": {"x": left, "y": top, "width": right - left, "height": bottom - top},
    }


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
    covered_y_ranges: list[tuple[int, int]] = []
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
            # 跳过已被上方网格/计算/表达式区域覆盖的 start，避免重复切分
            if any(
                covered_min <= start.line.y1 <= covered_max
                for covered_min, covered_max in covered_y_ranges
            ):
                continue
            next_y = column_starts[index + 1].line.y1 if index + 1 < len(column_starts) else page_height
            question_end_y = _question_end_y(lines, start.line.y1, next_y)
            question_end_y = _trim_to_internal_next_start(
                start,
                lines,
                column_left,
                column_right,
                question_end_y,
            )
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
                _track_covered_y_ranges(covered_y_ranges, calculation_regions)
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
                _track_covered_y_ranges(covered_y_ranges, grid_regions)
                continue

            # 最终兜底：尝试按算式内容拆分
            raw_text = _text_in_box(lines, column_left, column_right, start.line.y1, question_end_y)
            text_regions = _split_region_by_expressions(
                start, raw_text, lines, column_left, column_right,
                question_end_y, question_end_y == page_height,
            )
            if text_regions:
                regions.extend(text_regions)
                _track_covered_y_ranges(covered_y_ranges, text_regions)
                continue

            x = max(0, column_left - QUESTION_PADDING)
            y = max(0, start.line.y1 - QUESTION_PADDING)
            right = min(page_width, column_right + QUESTION_PADDING)
            bottom = _question_crop_bottom(
                start,
                lines,
                column_left,
                column_right,
                question_end_y,
                page_height,
                reaches_page_bottom=question_end_y == page_height,
            )
            box = {"x": x, "y": y, "width": right - x, "height": bottom - y}
            question_text = _strip_section_titles(raw_text)
            regions.append(
                {
                    "question_no": start.question_no,
                    "question_text": question_text,
                    "question_stem": _section_instruction_for_region(start, lines, column_left, column_right, question_end_y),
                    "question_box": box,
                }
            )
    # 过滤所有区域文本中的标题行
    for region in regions:
        region["question_text"] = _strip_section_titles(region["question_text"])
    return regions


def _trim_to_internal_next_start(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
) -> int:
    candidates = []
    for line in lines:
        if not (column_left <= line.center_x < column_right):
            continue
        if line.y1 <= start.line.y1 + 18 or line.y1 >= question_end_y:
            continue
        if _skip_question_start_line(line.text):
            continue
        if start.kind == "top" and _is_top_start_line(line.text):
            candidates.append(line.y1)
        elif start.kind == "sub" and _is_sub_start_line(line.text):
            candidates.append(line.y1)
    if not candidates:
        return question_end_y
    return max(start.line.y1 + 24, min(candidates))


def _is_top_start_line(text: str) -> bool:
    if TOP_LEVEL_NUMBER.match(text):
        return True
    if BARE_TOP_LEVEL_NUMBER.match(text):
        return _looks_like_question_start(text)
    return False


def _is_sub_start_line(text: str) -> bool:
    stripped = text.strip()
    return (
        SUB_QUESTION_PATTERN.match(stripped) is not None
        or SUB_NUMBER_WITH_PAREN.match(stripped) is not None
        or CIRCLED_NUMBER.match(stripped) is not None
    )


def _question_crop_bottom(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    page_height: int,
    reaches_page_bottom: bool,
) -> int:
    choice_bottom = _choice_question_bottom(start, lines, column_left, column_right, question_end_y)
    if choice_bottom is not None:
        return min(page_height, max(start.line.y1 + 24, choice_bottom + QUESTION_BOTTOM_GAP))
    if reaches_page_bottom:
        return page_height
    return max(start.line.y1 + 24, question_end_y - QUESTION_BOTTOM_GAP)


def _track_covered_y_ranges(
    covered: list[tuple[int, int]],
    new_regions: list[dict],
) -> None:
    """记录新建区域占用的 Y 范围，用于跳过后续 start 避免重复切分"""
    if not new_regions:
        return
    min_y = min(r["question_box"]["y"] for r in new_regions)
    max_y = max(
        r["question_box"]["y"] + r["question_box"]["height"]
        for r in new_regions
    )
    covered.append((min_y, max_y))


def _choice_question_bottom(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
) -> int | None:
    option_lines = [
        line
        for line in lines
        if column_left <= line.center_x < column_right
        and start.line.y1 < line.y1 < question_end_y
        and CHOICE_OPTION_PATTERN.match(line.text.strip())
    ]
    if len(option_lines) < 2:
        return None
    return max(line.y2 for line in option_lines)


def _split_region_by_expressions(
    start: QuestionStart,
    raw_text: str,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    reaches_page_bottom: bool,
) -> list[dict]:
    """纯文本兜底：检测含 =/≈ 的算式并拆分为独立题目"""
    matches = list(EXPRESSION_LINE.finditer(raw_text))
    if len(matches) < 2:
        return []

    # 确认是算式列表（不以小题标号开头）
    for line in raw_text.splitlines():
        if SUB_QUESTION_PATTERN.match(line.strip()):
            return []

    regions: list[dict] = []
    for index, match in enumerate(matches):
        end_pos = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        expr_text = raw_text[match.start():end_pos].strip().rstrip(";；")
        if not expr_text:
            continue

        # 估算 Y 位置（按文本比例）
        text_lines = raw_text.splitlines()
        line_idx = raw_text[:match.start()].count("\n")
        total_lines = max(len(text_lines), 1)
        y_ratio = line_idx / total_lines
        region_height = question_end_y - start.line.y1
        est_y = start.line.y1 + int(region_height * y_ratio)
        est_h = max(24, int(region_height / total_lines))

        box = {
            "x": max(0, column_left - QUESTION_PADDING),
            "y": max(0, est_y - QUESTION_PADDING),
            "width": min(99999, column_right + QUESTION_PADDING) - max(0, column_left - QUESTION_PADDING),
            "height": est_h,
        }
        regions.append({
            "question_no": f"{start.question_no}.{index + 1}",
            "question_text": _strip_section_titles(expr_text),
            "question_stem": _section_instruction_for_region(start, lines, column_left, column_right, question_end_y),
            "question_box": box,
        })
    return regions


def _strip_section_titles(text: str) -> str:
    """移除如 '四、按要求完成下面各题。（共17分）' 的标题行"""
    return "\n".join(
        line for line in text.splitlines()
        if not _is_section_title_line(line.strip())
    ).strip()


def _section_instruction_for_region(
    start: QuestionStart,
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
) -> str:
    instruction_lines = [
        line
        for line in lines
        if column_left <= line.center_x < column_right
        and start.line.y1 <= line.y1 < question_end_y
        and _is_section_title_line(line.text.strip())
    ]
    if not instruction_lines and _is_section_title_line(start.line.text.strip()):
        instruction_lines = [start.line]
    if not instruction_lines:
        return ""

    text = " ".join(line.text.strip() for line in sorted(instruction_lines, key=lambda item: (item.y1, item.x1)))
    text = re.sub(r"^\s*\d{1,2}\s*(?:[．、]|[.](?!\d))\s*", "", text).strip()
    text = re.sub(r"^\s*[一二三四五六七八九十]+\s*[、．.]\s*", "", text).strip()
    text = re.sub(r"[（(]\s*(?:共)?\d+\s*分\s*[）)]", "", text).strip()
    return text.rstrip("。．.：:；;、 ") + "。"


def _is_section_title_line(text: str) -> bool:
    return (
        SECTION_TITLE_PATTERN.match(text) is not None
        or ARABIC_SECTION_TITLE_PATTERN.match(text) is not None
        or _is_comparison_title_line(text)
    )


def _is_comparison_title_line(text: str) -> bool:
    """检测如'3、比大小（在○里填上 >、< 或 =）'的比较标题行"""
    stripped = text.strip()
    if not _COMPARISON_TITLE_KEYWORD.search(stripped):
        return False
    # 确保以题号或中文序号开头，避免误判普通内容行
    if TOP_LEVEL_NUMBER.match(stripped):
        return True
    if re.match(r"^\s*[一二三四五六七八九十]+\s*[、．\.]", stripped):
        return True
    return False


def _question_end_y(lines: list[OcrLine], start_y: int, next_y: int) -> int:
    section_y = min(
        (
            line.y1
            for line in lines
            if start_y < line.y1 < next_y and _is_section_title_line(line.text)
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
    """检测算式区域并按位置拆分为独立题目，不依赖标题关键词"""
    # 收集区域内所有算式行（含与题号同行 Y 的算式）
    expression_lines = [
        line
        for line in lines
        if column_left <= line.center_x < column_right
        and start.line.y1 <= line.y1 < question_end_y
        and _is_calculation_expression(line.text)
        and not TOP_LEVEL_NUMBER.match(line.text)
    ]
    if len(expression_lines) < 2:
        return []

    rows = _cluster_rows(expression_lines)
    if not rows:
        return []

    # 有小题标号 (1)(2) 的不拆分
    for line in lines:
        if column_left <= line.center_x < column_right and start.line.y1 <= line.y1 < question_end_y:
            if SUB_QUESTION_PATTERN.match(line.text) or CIRCLED_NUMBER.search(line.text):
                return []

    all_exprs = [line for row in rows for line in row]
    columns = _cluster_columns(all_exprs, column_right - column_left)
    multi_item_rows = [row for row in rows if len(row) >= 2]

    # 情况 1：多行×多列网格
    if len(columns) >= 2 and multi_item_rows:
        return _regions_from_grid(
            start, lines, column_left, column_right,
            question_end_y, reaches_page_bottom,
            rows, columns, fill_missing_cells=False,
        )

    # 情况 2：竖排列表（每行一个算式）
    if len(rows) >= 2:
        items = [min(row, key=lambda item: item.x1) for row in rows]
        return _make_expression_regions(start, items, lines, column_left, column_right,
                                        question_end_y, reaches_page_bottom, vertical=True)

    # 情况 3：单行多列（同行多个算式，如 2.4-0.2=  409÷7≈  655÷5=）
    if len(columns) >= 2:
        items = [min(col, key=lambda item: item.y1) for col in columns]
        return _make_expression_regions(start, items, lines, column_left, column_right,
                                        question_end_y, reaches_page_bottom, vertical=False)

    return []


def _make_expression_regions(
    start: QuestionStart,
    items: list[OcrLine],
    lines: list[OcrLine],
    column_left: int,
    column_right: int,
    question_end_y: int,
    reaches_page_bottom: bool,
    vertical: bool,
) -> list[dict]:
    """将算式列表制作为独立题目区域"""
    if len(items) < 2:
        return []

    regions: list[dict] = []
    for index, item in enumerate(items):
        if vertical:
            # 竖排：Y 方向切分
            end_y = items[index + 1].y1 if index + 1 < len(items) else question_end_y
            if end_y - item.y1 < 18:
                continue
            x = max(0, column_left - QUESTION_PADDING)
            w = min(99999, column_right + QUESTION_PADDING) - x
            y = max(0, item.y1 - QUESTION_PADDING)
            h = (end_y if reaches_page_bottom and index + 1 == len(items)
                 else max(item.y1 + 24, end_y - QUESTION_BOTTOM_GAP)) - y
        else:
            # 横排：X 方向切分
            next_x = items[index + 1].x1 if index + 1 < len(items) else column_right
            x = max(column_left, item.x1 - QUESTION_PADDING)
            w = min(next_x - 4, column_right) - x
            y = max(0, item.y1 - QUESTION_PADDING)
            h = max(item.y1 + 24, question_end_y - QUESTION_BOTTOM_GAP) - y

        region_text = _text_in_box(lines, int(x), int(x + w), item.y1,
                                   items[index + 1].y1 if vertical and index + 1 < len(items) else question_end_y)
        region_text = _clean_expression_region_text(region_text, item.text)
        regions.append({
            "question_no": f"{start.question_no}.{index + 1}",
            "question_text": region_text,
            "question_stem": _section_instruction_for_region(start, lines, column_left, column_right, question_end_y),
            "question_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        })
    return regions


def _clean_expression_region_text(region_text: str, expression_text: str) -> str:
    stripped = _strip_section_titles(region_text)
    if stripped and not ARABIC_SECTION_TITLE_PATTERN.search(stripped):
        return stripped
    expression_match = EXPRESSION_LINE.search(expression_text) or EXPRESSION_LINE.search(region_text)
    if expression_match:
        return expression_match.group(0).strip()
    return stripped


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
        and not _is_section_title_line(line.text)
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
            bottom = row_end_y if reaches_page_bottom and row_index + 1 == len(rows) else max(top + 24, row_end_y - QUESTION_BOTTOM_GAP)
            box = {"x": left, "y": top, "width": right - left, "height": bottom - top}
            regions.append(
                {
                    "question_no": f"{start.question_no}.{question_index}",
                    "question_text": _text_in_box(lines, left, right, row_starts[row_index], row_end_y),
                    "question_stem": _section_instruction_for_region(start, lines, column_left, column_right, question_end_y),
                    "question_box": box,
                }
            )
            question_index += 1
    return regions


def _is_calculation_expression(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return (
        not normalized.startswith(("=", "＝"))
        and
        (normalized[0].isdigit() or normalized.startswith(("*", "(", "（")))
        and
        bool(CALCULATION_EXPRESSION_PATTERN.fullmatch(normalized))
        and sum(character.isdigit() for character in normalized) >= 2
        and any(symbol in normalized for symbol in "+-*±×xX÷/≈≒")
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
