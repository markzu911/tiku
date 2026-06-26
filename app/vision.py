import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.answer_validation import _compute_expression
from app.model_settings import get_active_model_config
from app.question_filter import is_meaningful_question, normalize_question_candidate
from app.question_ocr import extract_question_regions


logger = logging.getLogger(__name__)
CHOICE_OPTION_LINE_PATTERN = re.compile(r"^\s*[A-D][、.．）)]\s*")
CHOICE_OPTION_PREFIX = re.compile(r"^[A-D][、.．）)]\s*")
COMPARISON_FILL_SYMBOL = re.compile(r"[○〇]")
DIAGRAM_KEYWORD = re.compile(
    r"如图|下图|上图|右图|左图|看图|表中|下表|上表|统计图|图表|线段图|示意图|"
    r"条形图|折线图|饼图|图形|方格|格子|数一数|圈一圈|连一连|涂一涂|画一画|"
    r"钟面|时钟|数轴|计数器|算盘|小棒|积木|七巧板|天平|尺子"
)
VISUAL_OPERATION_KEYWORD = re.compile(
    r"按要求.*(?:画|作)|作图|画图|画出|画一画|连一连|涂一涂|圈一圈|描一描|"
    r"用(?:圆规|直尺|三角尺|量角器)|射线|线段|垂线|垂直"
)
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
    "image/gif",
    "image/tiff",
    "image/x-tiff",
}
VISION_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ANALYSIS_BATCH_SIZE = 4


async def extract_questions_from_image(
    file: UploadFile,
    grade_level: int,
    allowed_category_names: list[str],
) -> dict[str, Any]:
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="请上传 JPG、PNG、WebP、BMP、GIF 或 TIFF 格式的试卷图片")

    image_bytes = await file.read()
    image_bytes, image_mime_type = normalize_uploaded_image(image_bytes, file.content_type)
    regions = extract_question_regions(image_bytes, Path("debug/debug_question_boxes.png"))
    if not regions:
        return {"paper_title": "", "page_mark": "", "questions": []}

    questions = [
        {
            "question_no": region["question_no"],
            "question_text": region["question_text"],
            "question_stem": region.get("question_stem", ""),
            "answer": "",
            "student_answer": "",
            "is_correct": None,
            "A": "",
            "B": "",
            "C": "",
            "D": "",
            "grade_level": grade_level,
            "category_name": "",
            "question_type": "",
            "answer_confidence": None,
            "needs_review": region["question_no"] == "fallback",
            "review_reason": "OCR未能稳定定位题目区域，已保留整页/整块内容待复核" if region["question_no"] == "fallback" else "",
            "question_box": region["question_box"],
            "has_image": False,
            "_image_bytes": _crop_question_image(image_bytes, region["question_box"]),
        }
        for region in regions
    ]
    analyzed = await _analyze_question_images(questions, grade_level, image_mime_type, allowed_category_names)
    for index, question in enumerate(questions):
        analysis = analyzed.get(index, {})
        for field in (
            "question_text",
            "answer",
            "student_answer",
            "is_correct",
            "A",
            "B",
            "C",
            "D",
            "category_name",
            "question_type",
            "answer_confidence",
            "needs_review",
            "review_reason",
        ):
            if field in analysis:
                question[field] = analysis[field]
        if not str(question.get("question_stem") or "").strip() and "question_stem" in analysis:
            question["question_stem"] = analysis["question_stem"]
        question["category_name"] = _allowed_category_name(analysis.get("category_name"), allowed_category_names)
        question["question_type"] = _chinese_question_type(analysis.get("question_type"))
        question["question_text"] = _without_embedded_choice_options(question["question_text"], question)
        # 清理选项字段中的 A. B. C. D. 前缀
        for key in ("A", "B", "C", "D"):
            question[key] = _strip_option_prefix(question.get(key))
        if _has_choice_options(question):
            # 选择题的 answer 和 student_answer 只保留字母
            question["answer"] = _normalize_choice_letter(question.get("answer"), question)
            question["student_answer"] = _normalize_choice_letter(question.get("student_answer"), question)
        ai_has_image = _as_bool(analysis.get("has_image")) is True
        question["has_image"] = _should_store_question_image(question, ai_has_image)
        _fill_visual_operation_answer(question)
        question.pop("_image_bytes", None)

    questions = _split_final_merged_expressions(questions)
    questions = [_strip_trailing_answer(q) for q in questions]
    questions = [_validate_and_fix_question(q) for q in questions]
    questions = [_normalize_comparison_circle(q) for q in questions]
    questions = [normalize_question_candidate(question) for question in questions]
    questions = [question for question in questions if is_meaningful_question(question)]
    return {"paper_title": "", "page_mark": "", "questions": questions}


async def _analyze_question_images(
    questions: list[dict[str, Any]],
    grade_level: int,
    image_mime_type: str,
    allowed_category_names: list[str],
) -> dict[int, dict[str, Any]]:
    model_config = get_active_model_config()
    api_key = model_config["api_key"]
    if not api_key:
        logger.warning("%s is not configured; storing OCR text without AI analysis", model_config["label"])
        return {}

    base_url = model_config["base_url"]
    model = model_config["model"]
    analyzed: dict[int, dict[str, Any]] = {}
    for start in range(0, len(questions), ANALYSIS_BATCH_SIZE):
        batch = questions[start : start + ANALYSIS_BATCH_SIZE]
        content = [
            {
                "type": "text",
                "text": _analysis_prompt(grade_level, start, batch, allowed_category_names),
            }
        ]
        for offset, question in enumerate(batch):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Question index {start + offset}; OCR text: {question['question_text']}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{base64.b64encode(question['_image_bytes']).decode('ascii')}"
                        },
                    },
                ]
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
            result = _extract_json(response.json()["choices"][0]["message"]["content"])
        except (HTTPException, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Question image analysis failed for batch %s: %s: %s", start, type(exc).__name__, exc)
            continue

        for item in result.get("questions", []):
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if start <= index < start + len(batch):
                analyzed[index] = item
    return analyzed


def _analysis_prompt(
    grade_level: int,
    start: int,
    batch: list[dict[str, Any]],
    allowed_category_names: list[str],
) -> str:
    indexes = list(range(start, start + len(batch)))
    categories = "、".join(allowed_category_names) or "无"
    return f"""Analyze each complete question screenshot for a primary-school grade {grade_level} worksheet.

Return JSON only:
{{"questions":[{{"index":{indexes[0] if indexes else 0},"question_text":"","question_stem":"","answer":"","student_answer":"","is_correct":true,"A":"","B":"","C":"","D":"","category_name":"","question_type":"","answer_confidence":0.7,"needs_review":false,"review_reason":"","has_image":false}}]}}

Rules:
1. Each image is one complete question region. Read the image itself; OCR text is only a hint.
2. question_text: the question body WITHOUT any A/B/C/D option lines. For non-choice questions, include the full text and leave A/B/C/D empty. question_stem: only when a shared instruction spans multiple questions (e.g. "计算下面各题").
3. FOR CHOICE QUESTIONS (questions with printed A. B. C. D. options):
   a. Put each option's FULL text into its own field (A/B/C/D), WITHOUT the "A."/"B."/"C."/"D." prefix.
   b. answer: the letter ONLY — "A", "B", "C", or "D" (NOT the option text). If the correct answer is printed (✓, ★, underlined, or in an answer key), use that letter.
   c. student_answer: the letter the student wrote, circled, ticked, or filled. If no student mark is visible, leave empty.
   d. is_correct: true if student_answer equals answer, false if they differ, null if student_answer is empty.
4. FOR NON-CHOICE questions: answer is the full correct answer text. student_answer is what the student wrote. is_correct: true if answers match, false if different, null if no student work.
   For drawing/construction/operation questions (作图、画图、连线、涂色、几何构造), do NOT leave answer empty just because the answer is visual. Put a concise textual description of the required construction in answer, describe the visible student drawing/work in student_answer, and use null for is_correct when you cannot determine correctness from the image.
5. Do not return image coordinates or image-cropping instructions.
6. category_name must be exactly one of these existing Chinese categories: {categories}. Never invent, translate, or return an English category.
7. question_type is an AI analysis of the knowledge point. You MUST return a concise Chinese label (e.g. "两位数乘法", "分数比较", "时间计算"). NEVER return English, pinyin, numbers-only, or an empty string. If uncertain, use a short Chinese description of the main math skill tested.
8. has_image is CRITICAL — it controls whether the question image is saved. Set has_image to true when the question contains ANY of: a printed diagram, chart, table, grid, geometric shape, number line, clock face, bar/line/pie chart, coordinate grid, object illustration, or visual counting aid. Set has_image to false for text-only comparison/fill-in questions such as "63+25 ○ 63+22"; the ○/〇 comparison placeholder itself is not a diagram.
9. answer_confidence is your confidence in the answer/student_answer/is_correct fields from 0 to 1. Set needs_review true and give a short Chinese review_reason when the answer is visual, ambiguous, not fully visible, or you are less than 0.8 confident.
10. Use the supplied global indexes exactly: {indexes}.
"""


def _crop_question_image(image_bytes: bytes, question_box: dict[str, int]) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        x = max(0, int(question_box["x"]))
        y = max(0, int(question_box["y"]))
        right = min(image.width, x + int(question_box["width"]))
        bottom = min(image.height, y + int(question_box["height"]))
        output = BytesIO()
        image.convert("RGB").crop((x, y, right, bottom)).save(output, format="PNG")
        return output.getvalue()


def _extract_json(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise HTTPException(status_code=502, detail="模型没有返回可解析的 JSON")
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="模型返回的 JSON 格式不正确") from exc


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0"}:
        return False
    return None


def _should_store_question_image(question: dict[str, Any], ai_has_image: bool) -> bool:
    image_hint_text = "\n".join(
        str(question.get(key) or "").strip()
        for key in ("question_stem", "question_text")
        if str(question.get(key) or "").strip()
    )
    text_has_image = (
        DIAGRAM_KEYWORD.search(image_hint_text) is not None
        or VISUAL_OPERATION_KEYWORD.search(image_hint_text) is not None
    )
    if _is_text_only_comparison_question(question, image_hint_text):
        return False
    return ai_has_image or text_has_image


def _is_text_only_comparison_question(question: dict[str, Any], text: str) -> bool:
    if _has_choice_options(question) or not COMPARISON_FILL_SYMBOL.search(text):
        return False
    if DIAGRAM_KEYWORD.search(text) or VISUAL_OPERATION_KEYWORD.search(text):
        return False
    meaningful_chars = re.sub(r"[\d\s.+\-−×xX*÷/＝=<>＞＜○〇（）()，,。；;：“”\"'、分米厘米元角时年月日平方立方]", "", text)
    allowed_words = (
        "在里填或大小比较比较填上填入"
        "米厘米分米元角时分年月日平方厘米平方分米平方米"
    )
    remainder = "".join(char for char in meaningful_chars if char not in allowed_words)
    return not remainder


# 最终兜底：拆分含多个 = 或 ≈ 的合并算式，分数优先作为一个数处理
_FINAL_NUMBER = r"(?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
_FINAL_EXPRESSION = re.compile(
    rf"{_FINAL_NUMBER}(?:\s*[+\-−×xX*÷/±]\s*{_FINAL_NUMBER})+\s*[≈≒＝=]"
)
_SUB_QUESTION_MARKER = re.compile(
    r"(?m)(?:^|[\s，,、;；])(?:[(（]\s*\d{1,2}\s*[)）]|[①-⑳])"
)


def _validate_and_fix_question(question: dict[str, Any]) -> dict[str, Any]:
    """校验并自动修复单个题目的常见 AI 错误"""
    options = {
        key: str(question.get(key) or "").strip()
        for key in ("A", "B", "C", "D")
    }
    has_options = any(options.values())

    if has_options:
        # 1. 选择题：确保 answer 是 A-D 之一
        answer = str(question.get("answer") or "").strip().upper()
        if answer not in ("A", "B", "C", "D"):
            # 尝试匹配选项文本
            answer = _match_answer_to_option(answer, options)
        # 如果还不对，检查答案是否藏在选项文本中
        if answer not in ("A", "B", "C", "D"):
            for key, text in options.items():
                if text and str(question.get("answer") or "") in text:
                    answer = key
                    break
        question["answer"] = answer if answer in ("A", "B", "C", "D") else ""

        # 2. 确保至少 2 个选项非空，否则清空——不是选择题
        filled = [k for k, v in options.items() if v]
        if len(filled) < 2:
            for key in ("A", "B", "C", "D"):
                question[key] = ""
            question["answer"] = str(question.get("answer") or "")
        else:
            # 3. 确保 answer 对应的选项非空
            if answer and not options.get(answer):
                # 找一个非空选项作为 fallback
                question["answer"] = filled[0] if filled else ""

        # 4. student_answer 规范化
        sa = str(question.get("student_answer") or "").strip().upper()
        if sa and sa not in ("A", "B", "C", "D"):
            sa = _match_answer_to_option(sa, options)
        question["student_answer"] = sa if sa in ("A", "B", "C", "D") else ""

    # 5. 非选择题：answer 不应过短（可能是截断的）
    if not has_options:
        answer = str(question.get("answer") or "").strip()
        if len(answer) <= 1 and answer not in ("", "√", "×", "○"):
            # 单字符答案对非选题通常不对，清空等人工复核
            pass  # 保留，因为可能是简单的数字答案

    # 6. question_text 不应为空
    if not str(question.get("question_text") or "").strip():
        # 从 OCR 文本或其他字段恢复
        fallback = str(question.get("question_stem") or "").strip()
        if not fallback:
            options_text = " ".join(
                f"{k}. {options[k]}" for k in ("A", "B", "C", "D") if options[k]
            )
            fallback = options_text
        question["question_text"] = fallback

    return question


def _match_answer_to_option(text: str, options: dict[str, str]) -> str:
    """尝试将答案文本匹配到 A/B/C/D 选项"""
    text = str(text or "").strip()
    if not text:
        return ""
    text_upper = text.upper()
    # 直接是字母
    if text_upper in ("A", "B", "C", "D"):
        return text_upper
    # 去掉常见前缀: A. A、 A) A．
    cleaned = CHOICE_OPTION_PREFIX.sub("", text).strip()
    if cleaned.upper() in ("A", "B", "C", "D"):
        return cleaned.upper()
    # 匹配选项文本内容
    for key in ("A", "B", "C", "D"):
        opt = options.get(key, "")
        if opt and (cleaned == opt or cleaned in opt or opt in cleaned):
            return key
    return ""


# 匹配算式后面紧跟的数字答案  e.g. "44+36=80" → answer "80"
_TRAILING_EQ_ANSWER = re.compile(r"\s*[=＝≈≒]\s*(\d+(?:\.\d+)?(?:/\d+)?)\s*$")
# OCR 常把 = 误读为 -，e.g. "44+36-80" 实际应该是 "44+36=80"
_TRAILING_MINUS_ANSWER = re.compile(r"\s*[-\-−]\s*(\d+(?:\.\d+)?(?:/\d+)?)\s*$")
_SIMPLE_ARITHMETIC = re.compile(r"^[\d.]+\s*[+\-×÷±]\s*[\d.]+$")


def _normalize_comparison_circle(question: dict[str, Any]) -> dict[str, Any]:
    """统一比较题中的 ○ 为 〇"""
    for key in ("question_text", "question_stem", "answer", "student_answer"):
        value = question.get(key)
        if isinstance(value, str) and "○" in value:
            question[key] = value.replace("○", "〇")
    return question


def _strip_trailing_answer(question: dict[str, Any]) -> dict[str, Any]:
    """剥离题目文本末尾的学生答案（包括 OCR 误读 = 为 - 的情况）"""
    text = str(question.get("question_text") or "")
    existing_student = str(question.get("student_answer") or "").strip()
    if existing_student and existing_student != "未作答":
        return question

    # 1. 正常情况：= 后面跟数字
    match = _TRAILING_EQ_ANSWER.search(text)
    if match:
        answer_value = match.group(1)
        question["question_text"] = text[:match.start()].strip()
        question["student_answer"] = answer_value
        existing_answer = str(question.get("answer") or "").strip()
        if existing_answer:
            question["is_correct"] = existing_answer.strip() == answer_value
        return question

    # 2. OCR 误读：= 被识别为 -，如 "44+36-80"
    match = _TRAILING_MINUS_ANSWER.search(text)
    if match:
        prefix = text[:match.start()].strip()
        # 检查前缀是否是简单算式
        arith_match = _SIMPLE_ARITHMETIC.match(prefix)
        if arith_match:
            answer_value = match.group(1)
            try:
                # 用程序验证：prefix 的计算结果是否等于末尾数字
                computed = _compute_expression(prefix)
                if computed and computed.strip() == answer_value:
                    # 确认：末尾数字是正确计算结果 → OCR 误读 = 为 -
                    question["question_text"] = prefix
                    question["student_answer"] = answer_value
                    existing_answer = str(question.get("answer") or "").strip()
                    if existing_answer:
                        question["is_correct"] = existing_answer.strip() == answer_value
                    elif computed:
                        question["answer"] = computed
                        question["is_correct"] = True
                        question["answer_source"] = "program"
                        question["answer_confidence"] = 1.0
            except Exception:
                pass  # 计算失败不处理
    return question


def _split_final_merged_expressions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """最终防线：无小题编号时，连续算式必须拆成独立题目。"""
    result: list[dict[str, Any]] = []
    for question in questions:
        text = str(question.get("question_text") or "")
        expressions = _expression_segments(text)
        if len(expressions) < 2 or _has_choice_options(question):
            result.append(question)
            continue

        # 只有 (1) / （1） / ① 这种小题编号才允许多道算式留在同一题里。
        if _has_sub_question_marker(text):
            result.append(question)
            continue

        answers = _split_parallel_answer_field(question.get("answer"), len(expressions))
        student_answers = _split_parallel_answer_field(question.get("student_answer"), len(expressions))
        for idx, expr_text in enumerate(expressions):
            new_q = dict(question)
            new_q["question_text"] = expr_text
            new_q["question_no"] = f"{question.get('question_no', '')}.{idx + 1}".lstrip(".")
            new_q["question_box"] = dict(question.get("question_box") or {})
            if answers is not None:
                new_q["answer"] = answers[idx]
            if student_answers is not None:
                new_q["student_answer"] = student_answers[idx]
            if answers is not None and student_answers is not None:
                new_q["is_correct"] = _normalize_answer_text(answers[idx]) == _normalize_answer_text(
                    student_answers[idx]
                )
            result.append(new_q)
    return result


def _expression_segments(text: str) -> list[str]:
    matches = list(_FINAL_EXPRESSION.finditer(text))
    if len(matches) < 2:
        return []

    segments: list[str] = []
    for index, match in enumerate(matches):
        end_pos = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start():end_pos].strip().rstrip(";；。")
        if segment:
            segments.append(segment)
    return segments if len(segments) >= 2 else []


def _has_sub_question_marker(text: str) -> bool:
    return _SUB_QUESTION_MARKER.search(text) is not None


def _has_choice_options(question: dict[str, Any]) -> bool:
    return any(str(question.get(key) or "").strip() for key in ("A", "B", "C", "D"))


def _split_parallel_answer_field(value: Any, expected_count: int) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None

    parts = [part.strip() for part in re.split(r"[;；\n]+", text) if part.strip()]
    if len(parts) == expected_count:
        return parts

    parts = [part.strip() for part in re.split(r"\s*(?:，|、)\s*", text) if part.strip()]
    if len(parts) == expected_count:
        return parts

    return None


def _normalize_answer_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _fill_visual_operation_answer(question: dict[str, Any]) -> None:
    if _has_choice_options(question):
        return

    text = "\n".join(
        str(question.get(key) or "").strip()
        for key in ("question_stem", "question_text")
        if str(question.get(key) or "").strip()
    )
    if not text or VISUAL_OPERATION_KEYWORD.search(text) is None:
        return

    if not str(question.get("answer") or "").strip():
        question["answer"] = _visual_operation_expected_answer(text)
    if not str(question.get("student_answer") or "").strip() and _as_bool(question.get("has_image")) is True:
        question["student_answer"] = "见题图中的作图痕迹"
    if "is_correct" not in question:
        question["is_correct"] = None


def _visual_operation_expected_answer(text: str) -> str:
    lines = [
        re.sub(r"^\s*(?:\d{1,2}\s*[.．、]|[(（]\s*\d{1,2}\s*[)）]|[①-⑳])\s*", "", line).strip()
        for line in text.splitlines()
    ]
    task_lines = [
        line
        for line in lines
        if line
        and VISUAL_OPERATION_KEYWORD.search(line)
        and not re.fullmatch(r"按要求[画作]图[。.]?", line)
    ]
    if task_lines:
        answer = "按要求完成作图：" + "；".join(task_lines)
    else:
        answer = "按题目要求完成作图，答案需结合题图人工复核"
    return answer[:500]


def _allowed_category_name(value: Any, allowed_category_names: list[str]) -> str:
    name = str(value or "").strip()
    return name if name in allowed_category_names else "\u586b\u7a7a\u9898"


def _chinese_question_type(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    # Keep Chinese labels as-is
    if any("\u4e00" <= character <= "\u9fff" for character in name):
        return name
    # If the AI returned English, keep it \u2014 better than showing "\u672a\u8bc6\u522b\u9898\u578b"
    # But filter out obviously invalid values (very long text, pure numbers, coordinate-like strings)
    if len(name) <= 30 and not name.isdigit() and not name.startswith(("[", "(", "{")):
        return name
    return ""


def _strip_option_prefix(value: Any) -> str:
    text = str(value or "").strip()
    return CHOICE_OPTION_PREFIX.sub("", text).strip()


def _normalize_choice_letter(value: Any, question: dict[str, Any]) -> str:
    """将选择题答案规范化为纯字母 A/B/C/D，去掉前缀和多余文本"""
    text = str(value or "").strip()
    if not text:
        return ""
    # 已是纯字母
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()
    # 带前缀如 "A." "B、" "C)"
    cleaned = CHOICE_OPTION_PREFIX.sub("", text).strip()
    if cleaned.upper() in ("A", "B", "C", "D"):
        return cleaned.upper()
    # 可能是完整选项文本，尝试匹配 A/B/C/D 字段
    for key in ("A", "B", "C", "D"):
        option_text = str(question.get(key) or "").strip()
        if option_text and (cleaned == option_text or cleaned in option_text or option_text in cleaned):
            return key
    return ""


def _without_embedded_choice_options(question_text: Any, question: dict[str, Any]) -> str:
    text = str(question_text or "").strip()
    if not any(str(question.get(key) or "").strip() for key in ("A", "B", "C", "D")):
        return text
    return "\n".join(line for line in text.splitlines() if not CHOICE_OPTION_LINE_PATTERN.match(line)).strip()


def normalize_uploaded_image(image_bytes: bytes, mime_type: str | None) -> tuple[bytes, str]:
    normalized_mime_type = mime_type or ""
    if normalized_mime_type in VISION_IMAGE_TYPES:
        return image_bytes, normalized_mime_type

    try:
        image = Image.open(BytesIO(image_bytes))
        image.seek(0)
        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue(), "image/png"
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取该图片，请尝试转换为 JPG 或 PNG 后上传") from exc
