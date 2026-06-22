import json
import os
import base64
import re
from typing import Any
from xml.sax.saxutils import escape

import httpx
from fastapi import HTTPException


def _extract_json(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise HTTPException(status_code=502, detail="model did not return JSON")

    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="model returned invalid JSON") from exc


def _build_generation_prompt(source_questions: list[dict[str, Any]], count: int) -> str:
    source_json = json.dumps(source_questions, ensure_ascii=False, indent=2)
    return f"""
You are a primary-school question generator for Chinese students.
Generate {count} new questions based on the source wrong questions below.

Rules:
1. Keep the same grade level, category_name, and question_type as the source questions.
2. Keep the same knowledge point and a similar difficulty.
3. Do not copy the original numbers, wording, or scenario.
4. If the source is a choice question, generate choice questions with A/B/C/D.
5. If the source is not a choice question, keep A/B/C/D empty strings.
6. Generate concise Chinese questions suitable for the grade.
7. Keep text and pictures separate. Do not write visual descriptions like "图中有..." as a substitute for the picture.
8. If the question needs a picture, put the real drawing in diagram_svg. The SVG should contain only the diagram/visual objects, not the whole question text.
9. diagram_svg may include simple labels such as A/B/C/D only when they are part of visual options.
10. Use SVG shapes such as rect, circle, polygon, path, line, ellipse, and text. Do not use external images, scripts, foreignObject, or Markdown.
11. Return JSON only. Do not return Markdown or explanations outside JSON.

Return exactly this JSON object shape:
{{
  "questions": [
    {{
      "question_text": "new question text",
      "question_stem": "shared stem or empty string",
      "answer": "correct answer",
      "A": "",
      "B": "",
      "C": "",
      "D": "",
      "grade_level": 1,
      "category_name": "category",
      "question_type": "knowledge point",
      "analysis": "short solution explanation",
      "diagram_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"160\" viewBox=\"0 0 720 160\">...</svg> or empty string"
    }}
  ]
}}

Source wrong questions:
{source_json}
""".strip()


def _wrap_text(text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").replace("\r", "").split("\n"):
        if not paragraph:
            lines.append("")
            continue

        current = ""
        current_width = 0
        for char in paragraph:
            char_width = 2 if ord(char) > 127 else 1
            if current and current_width + char_width > max_width:
                lines.append(current)
                current = char
                current_width = char_width
            else:
                current += char
                current_width += char_width
        if current:
            lines.append(current)
    return lines or [""]


def _svg_text_line(text: str, x: int, y: int, size: int = 24, weight: int = 500, fill: str = "#1f2937") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" font-family="Microsoft YaHei, SimHei, Arial, sans-serif">'
        f"{escape(text)}</text>"
    )


def _sanitize_diagram_svg(svg: str) -> str:
    value = str(svg or "").strip()
    if not value:
        return ""

    match = re.search(r"<svg\b[\s\S]*?</svg>", value, re.IGNORECASE)
    if not match:
        return ""

    cleaned = match.group(0)
    cleaned = re.sub(r"<script\b[\s\S]*?</script>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<foreignObject\b[\s\S]*?</foreignObject>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", cleaned)
    cleaned = re.sub(r"\s+(?:href|xlink:href)\s*=\s*(['\"])(?!#).*?\1", "", cleaned, flags=re.IGNORECASE)
    if "xmlns=" not in cleaned[:160]:
        cleaned = cleaned.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return cleaned


def _build_svg_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


async def generate_similar_questions(source_questions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="please set OPENAI_API_KEY before generating questions")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    safe_count = max(1, min(20, int(count or 1)))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_generation_prompt(source_questions, safe_count)}],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"model API error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"model request failed: {exc}") from exc

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    result = _extract_json(content)
    questions = result.get("questions")
    if not isinstance(questions, list):
        raise HTTPException(status_code=502, detail="model JSON missing questions list")

    normalized = []
    for item in questions[:safe_count]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "question_text": str(item.get("question_text") or "").strip(),
                "question_stem": str(item.get("question_stem") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "A": str(item.get("A") or "").strip(),
                "B": str(item.get("B") or "").strip(),
                "C": str(item.get("C") or "").strip(),
                "D": str(item.get("D") or "").strip(),
                "grade_level": item.get("grade_level") or source_questions[0].get("grade_level"),
                "category_name": str(item.get("category_name") or source_questions[0].get("category_name") or "").strip(),
                "question_type": str(item.get("question_type") or source_questions[0].get("question_type") or "").strip(),
                "analysis": str(item.get("analysis") or "").strip(),
                "diagram_svg": str(item.get("diagram_svg") or "").strip(),
            }
        )

    output = [item for item in normalized if item["question_text"]]
    for item in output:
        diagram_svg = _sanitize_diagram_svg(item.pop("diagram_svg", ""))
        if diagram_svg:
            item["image_url"] = _build_svg_data_url(diagram_svg)
            item["has_image"] = True
            item["image_generated"] = True
            item["image_kind"] = "diagram"
        else:
            item["image_url"] = ""
            item["has_image"] = False
            item["image_generated"] = False

    return output
