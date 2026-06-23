import json
import base64
import re
from typing import Any
from xml.sax.saxutils import escape

from fastapi import HTTPException

from app.ai_provider import is_image_generation_enabled, request_chat_completion, request_image_generation


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
Generate {count} new questions based on the source questions below.

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
11. For 观察物体 / 小正方体堆叠 / 立方体透视图 / 几何体三视图 questions, do not freely draw the cube stack in diagram_svg. Instead set diagram_kind to "cube_stack" and provide cube_layout as coordinates.
12. cube_layout must be a list of cubes. Each cube is [x, y, z], where x/y are ground positions and z is the layer height starting at 0. Keep it simple: 1 to 30 cubes, integer coordinates only.
13. For cube_stack questions, diagram_svg may be an empty string because the backend will render a fixed isometric cube template from cube_layout.
14. If a raster image is more suitable than SVG, set image_prompt to a concise Chinese prompt for an image generation model. Prefer clean worksheet-style diagrams on a white background. Do not put question text in the generated image.
15. For non-cube-stack questions, set diagram_kind to an empty string and cube_layout to an empty list.
16. Return JSON only. Do not return Markdown or explanations outside JSON.

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
      "diagram_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"160\" viewBox=\"0 0 720 160\">...</svg> or empty string",
      "diagram_kind": "cube_stack or empty string",
      "cube_layout": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "image_prompt": "prompt for image generation or empty string"
    }}
  ]
}}

Source questions:
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


def _normalize_cube_layout(value: Any) -> list[tuple[int, int, int]]:
    if not isinstance(value, list):
        return []

    cubes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in value:
        if isinstance(item, dict):
            raw_x = item.get("x")
            raw_y = item.get("y")
            raw_z = item.get("z")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            raw_x, raw_y, raw_z = item[:3]
        else:
            continue

        try:
            x = int(raw_x)
            y = int(raw_y)
            z = int(raw_z)
        except (TypeError, ValueError):
            continue

        if not (-8 <= x <= 8 and -8 <= y <= 8 and 0 <= z <= 8):
            continue

        cube = (x, y, z)
        if cube in seen:
            continue
        seen.add(cube)
        cubes.append(cube)
        if len(cubes) >= 30:
            break

    if not cubes:
        return []

    min_x = min(x for x, _, _ in cubes)
    min_y = min(y for _, y, _ in cubes)
    min_z = min(z for _, _, z in cubes)
    normalized = sorted((x - min_x, y - min_y, z - min_z) for x, y, z in cubes)
    span_x = max(x for x, _, _ in normalized) + 1
    span_y = max(y for _, y, _ in normalized) + 1
    span_z = max(z for _, _, z in normalized) + 1
    if span_x > 8 or span_y > 8 or span_z > 8:
        return []
    return normalized


def _is_cube_stack_kind(item: dict[str, Any]) -> bool:
    diagram_kind = str(item.get("diagram_kind") or "").strip().lower().replace("-", "_")
    if diagram_kind == "cube_stack":
        return True

    text = " ".join(
        str(item.get(key) or "")
        for key in ("question_text", "question_stem", "category_name", "question_type")
    )
    cube_keywords = (
        "观察物体",
        "小正方体",
        "立方体",
        "正方体搭",
        "几何体",
        "三视图",
        "从正面",
        "从上面",
        "从左面",
        "透视图",
    )
    return any(keyword in text for keyword in cube_keywords)


def _polygon(points: list[tuple[float, float]], fill: str) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{point_text}" fill="{fill}" stroke="#374151" stroke-width="1.8" />'


def _cube_faces(cube: tuple[int, int, int]) -> dict[str, list[tuple[float, float]]]:
    x, y, z = cube
    width = 32
    depth = 18
    height = 36
    sx = (x - y) * width
    sy = (x + y) * depth - z * height

    top = [(sx, sy), (sx + width, sy + depth), (sx, sy + depth * 2), (sx - width, sy + depth)]
    left = [top[3], top[2], (top[2][0], top[2][1] + height), (top[3][0], top[3][1] + height)]
    right = [top[2], top[1], (top[1][0], top[1][1] + height), (top[2][0], top[2][1] + height)]
    return {"top": top, "left": left, "right": right}


def _render_cube_stack_svg(cubes: list[tuple[int, int, int]]) -> str:
    if not cubes:
        return ""

    cube_shapes = []
    all_points: list[tuple[float, float]] = []
    for cube in sorted(cubes, key=lambda item: (item[0] + item[1] + item[2], item[2], item[0])):
        faces = _cube_faces(cube)
        cube_shapes.append(faces)
        for points in faces.values():
            all_points.extend(points)

    min_x = min(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    pad = 14
    shift_x = pad - min_x
    shift_y = pad - min_y
    width = int(max_x - min_x + pad * 2)
    height = int(max_y - min_y + pad * 2)

    polygons: list[str] = []
    for faces in cube_shapes:
        shifted = {
            name: [(x + shift_x, y + shift_y) for x, y in points]
            for name, points in faces.items()
        }
        polygons.append(_polygon(shifted["left"], "#e5e7eb"))
        polygons.append(_polygon(shifted["right"], "#d1d5db"))
        polygons.append(_polygon(shifted["top"], "#f9fafb"))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="立方体堆叠图">'
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />'
        '<g stroke-linejoin="round" stroke-linecap="round">'
        f'{"".join(polygons)}'
        "</g>"
        "</svg>"
    )


def _build_svg_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


async def generate_similar_questions(source_questions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    safe_count = max(1, min(20, int(count or 1)))

    content = await request_chat_completion(
        [{"role": "user", "content": _build_generation_prompt(source_questions, safe_count)}],
        task="generation",
        json_response=True,
    )
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
                "diagram_kind": str(item.get("diagram_kind") or "").strip(),
                "cube_layout": item.get("cube_layout"),
                "image_prompt": str(item.get("image_prompt") or "").strip(),
            }
        )

    output = [item for item in normalized if item["question_text"]]
    for item in output:
        cube_layout = _normalize_cube_layout(item.pop("cube_layout", []))
        raw_diagram_svg = item.pop("diagram_svg", "")
        image_prompt = item.pop("image_prompt", "")
        if cube_layout and _is_cube_stack_kind(item):
            diagram_svg = _render_cube_stack_svg(cube_layout)
            item["diagram_kind"] = "cube_stack"
        else:
            diagram_svg = _sanitize_diagram_svg(raw_diagram_svg)
            item.pop("diagram_kind", None)

        if diagram_svg:
            item["image_url"] = _build_svg_data_url(diagram_svg)
            item["has_image"] = True
            item["image_generated"] = True
            item["image_kind"] = "diagram"
        elif image_prompt and is_image_generation_enabled():
            item["image_url"] = await request_image_generation(image_prompt)
            item["has_image"] = True
            item["image_generated"] = True
            item["image_kind"] = "generated"
        else:
            item["image_url"] = ""
            item["has_image"] = False
            item["image_generated"] = False

    return output
