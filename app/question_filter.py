import re
from typing import Any


PLACEHOLDER_TEXTS = {
    "(题目文本缺失)",
    "（题目文本缺失）",
    "题目文本缺失",
    "未识别",
}
TITLE_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"细心计算|计算|口算|直接写出得数|写出得数|列竖式计算|用竖式计算|脱式计算|"
    r"填空|选择|判断|解决问题|看图列式|按要求画图|操作题|画图题"
    r")(?:[。（(][^）)]*(?:分|验算)[）)]|[，,、\s]*带[*＊※]的要验算|[，,、\s]*每题\d+分)*[。．.：:；;、\s]*$"
)
INSTRUCTION_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"用?竖式计算|列竖式计算|脱式计算|递等式计算|计算下面各题|直接写出得数|细心计算|"
    r"选择正确答案|填一填|按要求画图"
    r").*(?:分|验算|每题|带[*＊※]).*$"
)
LEADING_NUMBER_PATTERN = re.compile(r"^\s*(?:\d{1,2}\s*(?:[．、]|[.](?!\d))\s*)")
EMBEDDED_ARITHMETIC_TITLE_PATTERN = re.compile(
    r"^\s*\d{1,2}\s*(?:[．、]|[.](?!\d))\s*"
    r"(?:直接写出得数|写出得数|细心计算|计算|口算)"
    r"[。．.：:；;、\s]*"
    r"(?P<expr>\d+(?:\.\d+)?\s*[+\-−×xX*÷/]\s*\d+(?:\.\d+)?\s*[=＝])"
)


def normalize_question_candidate(item: dict[str, Any]) -> dict[str, Any]:
    normalized_item = dict(item)
    normalized_item["question_text"] = _normalize_question_text(normalized_item.get("question_text"))
    if not _has_choice_options(normalized_item):
        return normalized_item

    text = str(normalized_item.get("question_text") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and any(_is_meaningful_line(line) for line in lines):
        return normalized_item

    fallback = str(normalized_item.get("question_stem") or "").strip()
    normalized_item["question_text"] = fallback or "选择题（题干待复核）"
    normalized_item["needs_review"] = True
    normalized_item["review_reason"] = str(normalized_item.get("review_reason") or "选择题题干识别不完整").strip()
    return normalized_item


def is_meaningful_question(item: dict[str, Any]) -> bool:
    if _has_choice_options(item):
        return True

    text = str(item.get("question_text") or "").strip()
    if not text:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if all(_is_placeholder_line(line) for line in lines):
        return False

    return any(_is_meaningful_line(line) for line in lines)


def _is_meaningful_line(line: str) -> bool:
    normalized = LEADING_NUMBER_PATTERN.sub("", line).strip()
    if not normalized or _is_placeholder_line(normalized):
        return False
    if TITLE_ONLY_PATTERN.fullmatch(normalized):
        return False
    if _is_instruction_only(normalized):
        return False
    return True


def _is_placeholder_line(line: str) -> bool:
    return LEADING_NUMBER_PATTERN.sub("", line).strip() in PLACEHOLDER_TEXTS


def _normalize_question_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_lines: list[str] = []
    for line in lines:
        match = EMBEDDED_ARITHMETIC_TITLE_PATTERN.match(line)
        if match:
            normalized_lines.append(match.group("expr").strip())
        elif not _is_meaningful_line(line):
            continue
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def _is_instruction_only(text: str) -> bool:
    if not INSTRUCTION_ONLY_PATTERN.fullmatch(text):
        return False
    return not any(marker in text for marker in ("（ ）", "( )", "____", "？", "?", "="))


def _has_choice_options(item: dict[str, Any]) -> bool:
    return sum(1 for key in ("A", "B", "C", "D") if str(item.get(key) or "").strip()) >= 2
