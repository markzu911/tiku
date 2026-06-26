import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Any


NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
VALUE_EXPRESSION_PATTERN = (
    rf"{NUMBER_PATTERN}(?:\s*[+\-−×xX*÷/]\s*{NUMBER_PATTERN})*"
)
CHINESE_NUMBER_PATTERN = re.compile(r"^[零〇一二三四五六七八九十]+$")
VISUAL_REVIEW_PATTERN = re.compile(
    r"按要求.*(?:画|作)|作图|画图|画出|画一画|连一连|涂一涂|圈一圈|描一描|"
    r"用(?:圆规|直尺|三角尺|量角器)|射线|线段|垂线|垂直"
)
SIMPLE_EXPRESSION = re.compile(
    r"(?P<expr>(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*[+\-−×xX*÷/]\s*(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)(?:[)）]\s*)*)+)"
    r"\s*[=＝≈≒]"
)
ANSWER_EXPRESSION = re.compile(
    r"(?P<expr>(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*[+\-−×xX*÷/]\s*(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)(?:[)）]\s*)*)+)"
    r"\s*[=＝≈≒]\s*(?:[（(]\s*[)）]|$|[，,。；;\s])"
)
# 无等号版本：86-26-60（ ） 直接跟空白括号的算式
ANSWER_EXPRESSION_BLANK = re.compile(
    r"(?P<expr>(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*[+\-−×xX*÷/]\s*(?:[（(]\s*)*(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)(?:[)）]\s*)*)+)"
    r"\s*[（(]\s*[)）]"
)
PRODUCT_DIGIT_COUNT = re.compile(
    r"(?P<left>\d+)\s*[×xX*]\s*(?P<right>\d+)\s*的?积是\s*[（(]\s*[)）]\s*位数"
)
PRODUCT_TRAILING_ZERO_COUNT = re.compile(
    r"(?P<left>\d+)\s*[×xX*]\s*(?P<right>\d+)\s*的?积的末尾有\s*[（(]\s*[)）]\s*个0"
)
COMPARISON_EXPRESSION = re.compile(
    rf"(?P<left>{VALUE_EXPRESSION_PATTERN})\s*(?:[○〇]|[（(]\s*[)）])\s*(?P<right>{VALUE_EXPRESSION_PATTERN})"
)
CIRCLE_COMPARISON_EXPRESSION = re.compile(
    rf"(?P<left>{VALUE_EXPRESSION_PATTERN})\s*[○〇]\s*(?P<right>{VALUE_EXPRESSION_PATTERN})"
)
FRACTION_UNIT_COUNT = re.compile(
    r"(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)\s*里有\s*[（(]\s*[)）]\s*个\s*1\s*/\s*(?P=denominator)"
    r".*?再添上\s*[（(]\s*[)）]\s*个.*?就是\s*(?P<target>\d+)\s*/\s*(?P=denominator)"
)
RECTANGLE_TO_SQUARE_WIDTH_INCREASE = re.compile(
    r"长为\s*(?P<length>\d+)\s*米.*?宽为\s*(?P<width>\d+)\s*米.*?长不变.*?正方形.*?宽(?:要)?增加\s*[（(]\s*[)）]\s*米"
)
CALENDAR_FACTS = re.compile(
    r"1\s*年有\s*[（(]\s*[)）]\s*个月.*?其中有\s*[（(]\s*[)）]\s*个月是大月"
)
CURRENT_YEAR_DAYS = re.compile(r"今年全年共有\s*[（(]\s*[)）]\s*天")
TOKEN = re.compile(r"\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|[+\-−×xX*÷/()（）]")
SUBQUESTION_MARKER = re.compile(r"(?m)(?:^|[\s，,、;；])[(（]\s*\d{1,2}\s*[)）]")
BLANK_MARKER = re.compile(r"[（(]\s*[)）]")
CHINESE_CHARACTER = re.compile(r"[\u4e00-\u9fff]")
ANSWER_TOKEN = re.compile(r"\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|[零〇一二三四五六七八九十]+|[<>＝=＞＜]")
CALCULATION_CUE = re.compile(
    r"[+\-−×xX*÷/=<>＞＜]|"
    r"\d+\s*/\s*\d+|"
    r"(?:计算|得数|积|和|差|商|倍|末尾|位数|共有|增加|减少|分数|面积|周长)"
)
CHINESE_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


@dataclass(frozen=True)
class ComputedAnswer:
    expression: str
    expected: str


def apply_answer_validation(question: dict[str, Any]) -> dict[str, Any]:
    item = dict(question)
    text = _combined_text(item)
    current_answer = _clean(item.get("answer"))

    computed_answers = _compute_answer_results(text)
    if _is_standalone_arithmetic_question(text) and computed_answers:
        item["answer"] = computed_answers[0].expected
        item["answer_source"] = "program"
        item["answer_confidence"] = 1.0
        item["needs_review"] = False
        item["review_reason"] = ""
        _set_correctness_from_answers(item)
        return item

    if computed_answers:
        _apply_program_checks(item, computed_answers, current_answer)
        return item

    if VISUAL_REVIEW_PATTERN.search(text):
        item.setdefault("answer_source", "visual_review")
        if item.get("answer_confidence") in (None, ""):
            item["answer_confidence"] = 0.4
        item["needs_review"] = True
        item.setdefault("review_reason", "作图/视觉题需要人工复核")
        if item.get("is_correct") is None:
            item["is_correct"] = None
        return item

    if current_answer and _is_calculation_like(text):
        item["answer_source"] = "calculation_unchecked"
        item["answer_confidence"] = 0.4
        item["needs_review"] = True
        item["review_reason"] = "涉及计算，但未能完成程序校验"
        item["is_correct"] = None
        return item

    if current_answer:
        item.setdefault("answer_source", "ai")
        if item.get("answer_confidence") in (None, ""):
            item["answer_confidence"] = 0.7
        if item.get("needs_review") is None:
            item["needs_review"] = False
        item.setdefault("review_reason", "")
        return item

    item.setdefault("answer_source", "unverified")
    if item.get("answer_confidence") in (None, ""):
        item["answer_confidence"] = 0.0
    item["needs_review"] = True
    item.setdefault("review_reason", "未识别到可靠答案")
    return item


def _apply_program_checks(item: dict[str, Any], computed_answers: list[ComputedAnswer], current_answer: str) -> None:
    if not current_answer:
        item["answer"] = "；".join(f"{answer.expression}={answer.expected}" for answer in computed_answers)
        item["answer_source"] = "program"
        item["answer_confidence"] = 1.0
        item["needs_review"] = False
        item["review_reason"] = ""
        _set_correctness_from_answers(item)
        return

    missing = _missing_expected_answers(computed_answers, current_answer)
    if missing:
        item["answer_source"] = "ai_program_conflict"
        item["answer_confidence"] = 0.4
        item["needs_review"] = True
        item["review_reason"] = "程序计算发现答案可能不一致，缺少结果：" + "；".join(
            answer.expected for answer in missing[:6]
        )
        return

    text = _combined_text(item)
    unchecked_count = _answer_slot_count(text) - len(computed_answers)
    if unchecked_count > 0 and _is_calculation_like(text):
        item["answer_source"] = "partial_program_checked"
        item["answer_confidence"] = 0.6
        item["needs_review"] = True
        item["review_reason"] = f"部分计算结果已程序校验，仍有{unchecked_count}处未能程序确认"
        item["is_correct"] = None
        return

    item["answer_source"] = "program_checked"
    if item.get("answer_confidence") in (None, ""):
        item["answer_confidence"] = 0.95
    if item.get("needs_review") is None:
        item["needs_review"] = False
    item.setdefault("review_reason", "")


def _missing_expected_answers(computed_answers: list[ComputedAnswer], answer_text: str) -> list[ComputedAnswer]:
    answer_tokens = _normalized_answer_tokens(answer_text)
    missing: list[ComputedAnswer] = []
    for answer in computed_answers:
        expected = _normalize_answer(answer.expected)
        if answer_tokens[expected] > 0:
            answer_tokens[expected] -= 1
        else:
            missing.append(answer)
    return missing


def _normalized_answer_tokens(answer_text: str) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for token in ANSWER_TOKEN.findall(answer_text):
        tokens[_normalize_answer(_normalize_answer_token(token))] += 1
    return tokens


def _normalize_answer_token(token: str) -> str:
    if CHINESE_NUMBER_PATTERN.fullmatch(token):
        return str(_parse_chinese_number(token))
    return token


def _parse_chinese_number(token: str) -> int:
    if token in CHINESE_DIGITS:
        return int(CHINESE_DIGITS[token])
    if "十" not in token:
        return 0
    tens_text, _, ones_text = token.partition("十")
    tens = 1 if not tens_text else int(CHINESE_DIGITS.get(tens_text, "0"))
    ones = 0 if not ones_text else int(CHINESE_DIGITS.get(ones_text, "0"))
    return tens * 10 + ones


def _compute_answer_results(text: str) -> list[ComputedAnswer]:
    results: list[ComputedAnswer] = []
    seen: set[tuple[str, str]] = set()

    for match in ANSWER_EXPRESSION.finditer(text):
        expression = match.group("expr")
        expected = _compute_expression(expression)
        if expected is not None:
            key = (expression, expected)
            if key not in seen:
                results.append(ComputedAnswer(expression, expected))
                seen.add(key)

    for match in ANSWER_EXPRESSION_BLANK.finditer(text):
        expression = match.group("expr")
        expected = _compute_expression(expression)
        if expected is not None:
            key = (expression, expected)
            if key not in seen:
                results.append(ComputedAnswer(expression, expected))
                seen.add(key)

    for match in COMPARISON_EXPRESSION.finditer(text):
        expression = f"{match.group('left')}〇{match.group('right')}"
        expected = _compare_expression_values(match.group("left"), match.group("right"))
        if expected is not None:
            key = (expression, expected)
            if key not in seen:
                results.append(ComputedAnswer(expression, expected))
                seen.add(key)

    for match in PRODUCT_DIGIT_COUNT.finditer(text):
        product = int(match.group("left")) * int(match.group("right"))
        expected = str(len(str(abs(product))))
        expression = f"{match.group('left')}×{match.group('right')}的积位数"
        key = (expression, expected)
        if key not in seen:
            results.append(ComputedAnswer(expression, expected))
            seen.add(key)

    for match in PRODUCT_TRAILING_ZERO_COUNT.finditer(text):
        product = int(match.group("left")) * int(match.group("right"))
        expected = str(_trailing_zero_count(product))
        expression = f"{match.group('left')}×{match.group('right')}积的末尾0个数"
        key = (expression, expected)
        if key not in seen:
            results.append(ComputedAnswer(expression, expected))
            seen.add(key)

    for match in FRACTION_UNIT_COUNT.finditer(text):
        numerator = int(match.group("numerator"))
        target = int(match.group("target"))
        denominator = match.group("denominator")
        expected_values = [
            (f"{numerator}/{denominator}里有几个1/{denominator}", str(numerator)),
            (f"{target}/{denominator}还需几个1/{denominator}", str(target - numerator)),
        ]
        for expression, expected in expected_values:
            key = (expression, expected)
            if int(expected) >= 0 and key not in seen:
                results.append(ComputedAnswer(expression, expected))
                seen.add(key)

    for match in RECTANGLE_TO_SQUARE_WIDTH_INCREASE.finditer(text):
        length = int(match.group("length"))
        width = int(match.group("width"))
        expected = str(length - width)
        expression = f"{length}米长方形扩成正方形宽增加"
        key = (expression, expected)
        if length >= width and key not in seen:
            results.append(ComputedAnswer(expression, expected))
            seen.add(key)

    if CALENDAR_FACTS.search(text):
        for expression, expected in (("1年月份数", "12"), ("一年大月数", "7")):
            key = (expression, expected)
            if key not in seen:
                results.append(ComputedAnswer(expression, expected))
                seen.add(key)

    if CURRENT_YEAR_DAYS.search(text):
        year = date.today().year
        expected = "366" if _is_leap_year(year) else "365"
        expression = f"{year}年全年天数"
        key = (expression, expected)
        if key not in seen:
            results.append(ComputedAnswer(expression, expected))
            seen.add(key)

    return results


def _compute_arithmetic_answer(text: str) -> str | None:
    if not _is_standalone_arithmetic_question(text):
        return None
    match = SIMPLE_EXPRESSION.search(text) or ANSWER_EXPRESSION_BLANK.search(text)
    if not match:
        return None
    return _compute_expression(match.group("expr"))


def _compute_expression(expression: str) -> str | None:
    cleaned = _normalize_expression_text(expression)
    if not cleaned:
        return None
    try:
        value = _evaluate_expression(cleaned)
    except (ZeroDivisionError, ValueError):
        return None
    return _format_fraction(value)


def _normalize_expression_text(expr: str) -> str:
    """标准化表达式：统一括号、移除空格、确保括号平衡"""
    text = expr.replace(" ", "").replace("　", "")
    # 统一括号为 ASCII
    text = text.replace("（", "(").replace("）", ")")
    # 补全不匹配的括号
    open_count = text.count("(")
    close_count = text.count(")")
    if open_count > close_count:
        text += ")" * (open_count - close_count)
    elif close_count > open_count:
        text = "(" * (close_count - open_count) + text
    # 去掉 leading/trailing 空白和多余的 =
    text = re.sub(r"[=＝≈≒]$", "", text)
    return text


def _compare_expression_values(left: str, right: str) -> str | None:
    try:
        left_value = _evaluate_expression(left)
        right_value = _evaluate_expression(right)
    except (ZeroDivisionError, ValueError):
        return None
    if left_value > right_value:
        return ">"
    if left_value < right_value:
        return "<"
    return "="


def _is_leap_year(year: int) -> bool:
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)


def _is_standalone_arithmetic_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if SUBQUESTION_MARKER.search(stripped) or len(BLANK_MARKER.findall(stripped)) >= 2:
        return False

    matches = list(SIMPLE_EXPRESSION.finditer(stripped))
    # 如果没找到带 = 的，尝试无等号算式
    if not matches:
        matches = list(ANSWER_EXPRESSION_BLANK.finditer(stripped))
    if len(matches) != 1:
        return False

    remainder = (stripped[: matches[0].start()] + stripped[matches[0].end() :]).strip()
    remainder = re.sub(r"^\s*(?:\d{1,2}\s*[.．、]\s*)", "", remainder).strip()
    remainder = re.sub(r"^[=＝≈≒?？（）()\s]*", "", remainder)
    remainder = re.sub(r"[=＝≈≒?？（）()\s]*$", "", remainder)
    return not CHINESE_CHARACTER.search(remainder)


def _answer_slot_count(text: str) -> int:
    return len(BLANK_MARKER.findall(text)) + len(CIRCLE_COMPARISON_EXPRESSION.findall(text))


def _is_calculation_like(text: str) -> bool:
    return bool(CALCULATION_CUE.search(text))


def _evaluate_expression(expression: str) -> Fraction:
    tokens = _tokenize_expression(expression)
    if not tokens:
        raise ValueError("empty expression")
    result, pos = _parse_expression(tokens, 0)
    if pos < len(tokens):
        raise ValueError(f"unexpected token at position {pos}: {tokens[pos]}")
    return result


def _tokenize_expression(expression: str) -> list[str]:
    return [token.replace(" ", "") for token in TOKEN.findall(expression)]


def _parse_expression(tokens: list[str], pos: int) -> tuple[Fraction, int]:
    """递归下降解析：处理 +/- 的表达式"""
    left, pos = _parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-", "−"):
        operator = _normalize_operator(tokens[pos])
        right, pos = _parse_term(tokens, pos + 1)
        left = left + right if operator == "+" else left - right
    return left, pos


def _parse_term(tokens: list[str], pos: int) -> tuple[Fraction, int]:
    """解析乘除项，处理 ×/÷"""
    left, pos = _parse_factor(tokens, pos)
    while pos < len(tokens) and _normalize_operator(tokens[pos]) in ("*", "/"):
        operator = _normalize_operator(tokens[pos])
        right, pos = _parse_factor(tokens, pos + 1)
        left = left * right if operator == "*" else left / right
    return left, pos


def _parse_factor(tokens: list[str], pos: int) -> tuple[Fraction, int]:
    """解析因子：数字 或 (表达式)"""
    if pos >= len(tokens):
        raise ValueError("unexpected end of expression")
    token = tokens[pos]
    # 括号表达式
    if token in ("(", "（"):
        value, pos = _parse_expression(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] not in (")", "）"):
            raise ValueError(f"missing closing parenthesis at position {pos}")
        return value, pos + 1
    # 负数：-3 或 (-3)
    if _normalize_operator(token) == "-" and pos + 1 < len(tokens):
        next_token = tokens[pos + 1]
        if next_token not in ("+", "-", "−", "*", "/", "×", "÷", "(", "（", ")", "）"):
            value, pos = _parse_factor(tokens, pos + 1)
            return -value, pos
    # 数字
    return _parse_number(token), pos + 1


def _parse_number(text: str) -> Fraction:
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(numerator) / Fraction(denominator)
    return Fraction(text)


def _normalize_operator(operator: str) -> str:
    if operator in {"×", "x", "X", "*"}:
        return "*"
    if operator in {"÷", "/"}:
        return "/"
    if operator in {"-", "−"}:
        return "-"
    return "+"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _trailing_zero_count(value: int) -> int:
    text = str(abs(value))
    return len(text) - len(text.rstrip("0"))


def _set_correctness_from_answers(item: dict[str, Any]) -> None:
    student_answer = _clean(item.get("student_answer"))
    answer = _clean(item.get("answer"))
    if not student_answer or not answer:
        return
    item["is_correct"] = _normalize_answer(student_answer) == _normalize_answer(answer)


def _normalize_answer(value: str) -> str:
    return (
        re.sub(r"\s+", "", value)
        .replace("；", ";")
        .replace("＞", ">")
        .replace("＜", "<")
        .replace("＝", "=")
        .lower()
    )


def _combined_text(item: dict[str, Any]) -> str:
    return "\n".join(
        _clean(item.get(key))
        for key in ("question_stem", "question_text", "text")
        if _clean(item.get(key))
    )


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()
