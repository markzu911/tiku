import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.question_ocr import extract_question_regions


logger = logging.getLogger(__name__)
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


async def extract_questions_from_image(file: UploadFile, grade_level: int) -> dict[str, Any]:
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
            "has_image": True,
            "_image_bytes": _crop_question_image(image_bytes, region["question_box"]),
        }
        for region in regions
    ]
    analyzed = await _analyze_question_images(questions, grade_level, image_mime_type)
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
        question.pop("_image_bytes", None)

    return {"paper_title": "", "page_mark": "", "questions": questions}


async def _analyze_question_images(
    questions: list[dict[str, Any]],
    grade_level: int,
    image_mime_type: str,
) -> dict[int, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not configured; storing OCR text without AI analysis")
        return {}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    analyzed: dict[int, dict[str, Any]] = {}
    for start in range(0, len(questions), ANALYSIS_BATCH_SIZE):
        batch = questions[start : start + ANALYSIS_BATCH_SIZE]
        content = [
            {
                "type": "text",
                "text": _analysis_prompt(grade_level, start, batch),
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


def _analysis_prompt(grade_level: int, start: int, batch: list[dict[str, Any]]) -> str:
    indexes = list(range(start, start + len(batch)))
    return f"""Analyze each complete question screenshot for a primary-school grade {grade_level} worksheet.

Return JSON only:
{{"questions":[{{"index":{indexes[0] if indexes else 0},"question_text":"","question_stem":"","answer":"","student_answer":"","is_correct":true,"A":"","B":"","C":"","D":"","category_name":"","question_type":""}}]}}

Rules:
1. Each image is one complete question region. Read the image itself; OCR text is only a hint.
2. Preserve all printed question text in question_text. Put a shared instruction in question_stem only when it is clearly present.
3. Read student work when visible. Set is_correct only when it can be determined; otherwise use null.
4. Do not return image coordinates or image-cropping instructions.
5. Use the supplied global indexes exactly: {indexes}.
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
