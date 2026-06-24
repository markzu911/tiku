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

from app.model_settings import get_active_model_config
from app.question_ocr import extract_question_regions


logger = logging.getLogger(__name__)
CHOICE_OPTION_LINE_PATTERN = re.compile(r"^\s*[A-D][、.．）)]\s*")
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
            "question_stem": "",
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
            "question_stem",
            "answer",
            "student_answer",
            "is_correct",
            "A",
            "B",
            "C",
            "D",
            "category_name",
            "question_type",
        ):
            if field in analysis:
                question[field] = analysis[field]
        question["category_name"] = _allowed_category_name(analysis.get("category_name"), allowed_category_names)
        question["question_type"] = _chinese_question_type(analysis.get("question_type"))
        question["question_text"] = _without_embedded_choice_options(question["question_text"], question)
        question["has_image"] = _as_bool(analysis.get("has_image")) is True
        question.pop("_image_bytes", None)

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
{{"questions":[{{"index":{indexes[0] if indexes else 0},"question_text":"","question_stem":"","answer":"","student_answer":"","is_correct":true,"A":"","B":"","C":"","D":"","category_name":"","question_type":"","has_image":false}}]}}

Rules:
1. Each image is one complete question region. Read the image itself; OCR text is only a hint.
2. Preserve all printed question text in question_text. For choice questions, question_text must contain only the stem; put A/B/C/D only in their separate fields and never repeat options in question_text. Put a shared instruction in question_stem only when it is clearly present.
3. Read student work when visible. Set is_correct only when it can be determined; otherwise use null.
4. Do not return image coordinates or image-cropping instructions.
5. category_name must be exactly one of these existing Chinese categories: {categories}. Never invent, translate, or return an English category.
6. question_type is an AI analysis of the knowledge point. You MUST return a concise Chinese label (e.g. "两位数乘法", "分数比较", "时间计算"). NEVER return English, pinyin, numbers-only, or an empty string. If uncertain, use a short Chinese description of the main math skill tested.
7. Set has_image to true only when the question includes a printed diagram, table, chart, geometry figure, or object illustration needed for the question. Set it to false for pure text, answer blanks, ruled lines, and student handwriting.
8. Use the supplied global indexes exactly: {indexes}.
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
