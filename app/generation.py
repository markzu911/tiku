import base64
import json
import os
import re
from typing import Any

import httpx
from fastapi import HTTPException


def _extract_json(value: str) -> dict[str, Any]:
    content = str(value or "").strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise HTTPException(status_code=502, detail="generation model did not return JSON")
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="generation model returned invalid JSON") from exc


def _build_generation_prompt(source_questions: list[dict[str, Any]], count: int) -> str:
    sources = json.dumps(source_questions, ensure_ascii=False)
    return f"""
You generate Chinese primary-school mathematics practice questions.
Create {count} new questions from the supplied wrong-answer examples.

Rules:
1. Preserve the source grade_level, category_name, and question_type.
2. Keep the same knowledge point and a comparable difficulty, but do not reuse the source wording, numbers, or scenario.
3. Return a correct answer and A/B/C/D only for choice questions; otherwise keep A/B/C/D as empty strings.
4. Write concise, natural Chinese suitable for primary-school students.
5. When a question genuinely requires a visual, return it in diagram_svg as a standalone SVG containing only the diagram, chart, or table. Do not place question prose in the SVG.
6. SVG may use svg, rect, line, circle, ellipse, polygon, path, text, and g only. Never use scripts, external URLs, foreignObject, or Markdown.
7. Provide a short Chinese analysis for each generated question.

Return JSON only in this exact shape:
{{
  "questions": [
    {{
      "question_text": "new question",
      "question_stem": "shared instruction or empty string",
      "answer": "correct answer",
      "A": "",
      "B": "",
      "C": "",
      "D": "",
      "grade_level": 3,
      "category_name": "category",
      "question_type": "knowledge point",
      "analysis": "short solution explanation",
      "diagram_svg": "<svg ...></svg> or empty string"
    }}
  ]
}}

Source wrong questions:
{sources}
""".strip()


def _sanitize_svg(value: Any) -> str:
    svg = str(value or "").strip()
    if not svg:
        return ""
    match = re.search(r"<svg\b[\s\S]*?</svg>", svg, re.IGNORECASE)
    if not match:
        return ""
    svg = match.group(0)
    svg = re.sub(r"<script\b[\s\S]*?</script>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<foreignObject\b[\s\S]*?</foreignObject>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", svg)
    svg = re.sub(r"\s+(?:href|xlink:href)\s*=\s*(['\"])(?!#).*?\1", "", svg, flags=re.IGNORECASE)
    if "xmlns=" not in svg[:160]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


def _svg_data_url(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


async def generate_similar_questions(source_questions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not source_questions:
        raise HTTPException(status_code=400, detail="at least one source question is required")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    safe_count = max(1, min(20, int(count or 1)))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_generation_prompt(source_questions, safe_count)}],
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
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"generation model error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"generation model request failed: {exc}") from exc

    result = _extract_json(response.json()["choices"][0]["message"]["content"])
    generated = result.get("questions")
    if not isinstance(generated, list):
        raise HTTPException(status_code=502, detail="generation model JSON is missing questions")

    output = []
    fallback = source_questions[0]
    for item in generated[:safe_count]:
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question_text") or "").strip()
        if not question_text:
            continue
        diagram_svg = _sanitize_svg(item.get("diagram_svg"))
        output.append(
            {
                "question_text": question_text,
                "question_stem": str(item.get("question_stem") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "A": str(item.get("A") or "").strip(),
                "B": str(item.get("B") or "").strip(),
                "C": str(item.get("C") or "").strip(),
                "D": str(item.get("D") or "").strip(),
                "grade_level": item.get("grade_level") or fallback.get("grade_level"),
                "category_name": str(item.get("category_name") or fallback.get("category_name") or "").strip(),
                "question_type": str(item.get("question_type") or fallback.get("question_type") or "").strip(),
                "analysis": str(item.get("analysis") or "").strip(),
                "image_url": _svg_data_url(diagram_svg) if diagram_svg else "",
                "has_image": bool(diagram_svg),
                "image_generated": bool(diagram_svg),
            }
        )
    return output
