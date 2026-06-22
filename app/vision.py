import base64
import json
import os
from io import BytesIO
from typing import Any

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image


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


CLEAN_QUESTION_IMAGE_RULES = """
Image crop rules for every returned question:
1. Never return a crop that contains student handwriting, filled answers, pencil marks, red pen ticks/crosses/circles, teacher corrections, scores, question numbers, or surrounding question text.
2. For every question where has_image is true, return question_image_bboxes: an array of normalized [x1, y1, x2, y2] boxes. Each box must cover one complete printed visual component, such as a diagram, table, chart, ruler, geometry figure, or picture.
3. A visual component must never be partial. For a table, include the complete table: every row, every column, all outer borders, headers, labels, and values. For a chart, include the full chart area, legend, axes, and labels. For a diagram, include every connected line, arrow, dimension label, and marked point.
4. For word problems containing a table or picture, crop the complete table or picture only. Do not crop a fragment of question prose together with only part of the table.
5. Use multiple boxes when one large box would capture handwriting, correction marks, question text, or unrelated content. Include every required printed visual component so the saved image is complete.
6. Every visual box must leave a visible clean margin around the full printed component. Never cut off arrows, dimension lines, labels, axis marks, table borders, scale ticks, or figure edges, even when they are close to the boundary.
7. Do not return cleanup boxes. The service stores only the selected visual regions and never paints white blocks over an image.
8. Include question_image_bboxes in every question JSON object, including when has_image is false (use []). Keep question_image_bbox for backward compatibility; set it to the union only when that union is clean.
""".strip()


ANSWER_EVALUATION_RULES = """
Answer evaluation rules:
- When a question has no student response, set student_answer to the exact text "未作答" and set is_correct to false.
- An unanswered question is always an incorrect question, including multi-blank questions where no answer was written.
- Use an empty student_answer only when the source image is unreadable; in that case still set is_correct to false.
""".strip()


VISUAL_ASSET_RULES = """
Visual asset extraction rules:
- Return question_visuals for every question. It must be an array of objects in this exact shape: {"kind":"table|diagram|chart|image","bbox":[x1,y1,x2,y2]}.
- Each item is one necessary non-text visual asset only. Never use a bbox for the whole question area.
- A word-problem table must be returned as one table asset containing the full table and nothing else. Do not include question prose, question numbers, answer space, student work, correction marks, or nearby content.
- Example: for a fruit word problem with a three-row fruit-count table, return exactly the complete three-row table as kind "table". Do not return the surrounding question text or the student's calculations.
- The question text is stored separately in question_text. question_visuals is only for information that cannot be preserved as text.
- Return [] when no visual asset is necessary. Keep question_image_bboxes only for backward compatibility; question_visuals is the source of truth for cropping.
""".strip()


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


def _build_extraction_prompt(grade_level: int) -> str:
    return f"""
你是一个小学试卷识别与判题助手。用户选择的年级是小学 {grade_level} 年级。请严格根据图片内容，从上到下、从左到右识别试卷中的题目，并只返回 JSON。

试卷命名规则：
1. paper_title 从图片顶部或明显标题位置提取试卷名称，例如“人教版三年级下册数学期末测试卷”。
2. 不要把学校、姓名、班级、考号、装订线、密封线、分数栏当成 paper_title。
3. page_mark 从卷子的角标、页码、页眉页脚中的页序信息提取，例如“第1页”“P1”“1/2”“A卷”“正面”“背面”。没有则为空字符串。
4. 如果图片中没有明确标题，paper_title 返回空字符串，不要编造。

题目提取规则：
1. 先识别大题题干，也就是一组题共同的作答要求。例如“列竖式计算，带☆的要验算。”这是 question_stem，不是一道具体题。
2. 不要把计分说明、小题数量说明、总分说明提取为题干或题目。例如“本题共10小题，每空1分，共28分”“每题2分”“共5分”必须忽略。
3. 大题下面有多个独立小题时，每个小题单独返回一条题目，并继承同一个 question_stem。
4. 同一道题中有多个填空时，仍然只返回一条题目，不要按空拆题。answer 和 student_answer 按空的顺序用中文分号“；”分隔。
5. question_text 只写具体题目内容，不要把大题题干重复拼进去。
6. category_name 是题目大类，只能优先使用：选择题、填空题、应用题、计算题、判断题。
7. question_type 是细题型/知识点，不是大类。示例：1-50加法、小数读写、小数加法、三位数除以一位数、统计表读数、分数加减法、角的认识。
8. grade_level 必须返回 {grade_level}，不要根据图片自行改成年级。
9. 如果图片上有学生作答痕迹，请提取 student_answer；如果没有学生答案，student_answer 必须为“未作答”，is_correct 必须为 false。
10. 请计算或判断正确答案 answer，并比较 student_answer 是否正确。多空题任意一空错误，整题 is_correct 为 false。
11. 如果题目有 A/B/C/D 选项，分别填入 A、B、C、D；没有选项则为空字符串。
12. 不要识别页眉、页脚、装饰文字、分数栏、批改符号为题目。
13. 保留题目前的特殊要求符号，例如 ☆817÷4= 中的 ☆ 必须保留在 question_text 里。
14. 如果一道题包含无法完整转成文字的信息，例如几何图、角度图、线段图、统计图、表格、示意图、图片材料，必须设置 has_image 为 true。
15. has_image=true 时，必须通过 question_visuals 返回这道题所需的独立视觉素材。每个素材只能是完整表格、完整图形、完整图表或完整图片之一，不能是整道题的版面范围。
16. question_visuals 中的 bbox 使用归一化坐标 [x1, y1, x2, y2]，取值 0 到 1。表格必须包含全部行列；图形必须包含全部边线、箭头、刻度和标注；不得包含题干、题号、答题区、学生作答、批改痕迹或相邻题目。
17. 如果题目是纯文字题，has_image 为 false，question_visuals 必须为 []。
18. 对于几何图题，question_text 只写题目文字要求；图形本身通过 question_visuals 保存。

返回 JSON 格式必须完全如下：
{{
  "paper_title": "试卷标题；没有则为空字符串",
  "page_mark": "角标/页码/卷面标记；没有则为空字符串",
  "questions": [
    {{
      "question_text": "具体题目内容",
      "question_stem": "所属大题题干/要求；没有则为空字符串；不要包含计分说明",
      "answer": "正确答案；多空题用；分隔；无法确定则为空字符串",
      "student_answer": "学生作答；多空题用；分隔；没有则为空字符串",
      "is_correct": true,
      "has_image": false,
      "question_visuals": [],
      "question_image_bboxes": [],
      "question_image_bbox": null,
      "A": "A选项；没有则为空字符串",
      "B": "B选项；没有则为空字符串",
      "C": "C选项；没有则为空字符串",
      "D": "D选项；没有则为空字符串",
      "grade_level": {grade_level},
      "category_name": "选择题/填空题/应用题/计算题/判断题之一；无法确定则为空字符串",
      "question_type": "细题型/知识点；无法确定则为空字符串"
    }}
  ]
}}

is_correct 只能是 true、false 或 null。
has_image 只能是 true 或 false。
question_visuals 必须始终存在。每个元素必须为 {{"kind":"table|diagram|chart|image","bbox":[x1,y1,x2,y2]}}；没有必要图像时为 []。
question_image_bboxes 必须始终存在；它和 question_image_bbox 仅为兼容字段，不作为新题图的裁图依据。
不要返回 Markdown。不要返回解释。不要返回 JSON 以外的任何内容。
""".strip()


async def _refine_question_visuals(
    image_bytes: bytes,
    image_mime_type: str,
    questions: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
) -> dict[int, list[dict[str, Any]]]:
    candidates = [
        {
            "index": index,
            "question_text": question.get("question_text", ""),
            "question_stem": question.get("question_stem", ""),
        }
        for index, question in enumerate(questions)
        if isinstance(question, dict) and question.get("has_image") is True
    ]
    if not candidates:
        return {}

    prompt = """You are reviewing only visual assets in a worksheet image. For each target question below, locate only the essential non-text visual material. Do not locate the whole question area.

Return JSON exactly as:
{"questions":[{"index":0,"question_visuals":[{"kind":"table|diagram|chart|image","bbox":[x1,y1,x2,y2]}]}]}

Rules:
1. A table asset must include its complete outer border, every row, every column, and all cell values, but no surrounding question prose, question number, answer space, handwriting, or correction marks.
2. A diagram/chart asset must include the complete figure and its necessary labels, arrows, axes, and scales, but no surrounding prose or student work.
3. Return question_visuals as [] when an asset cannot be isolated cleanly. Never return a partial table or a whole-question crop.
4. Use normalized coordinates against the full source image.

Target questions:
""" + json.dumps(candidates, ensure_ascii=False)

    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ],
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
        content = response.json()["choices"][0]["message"]["content"]
        refined = _extract_json(content)
    except (HTTPException, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return {}

    candidate_indexes = {candidate["index"] for candidate in candidates}
    return {
        int(item["index"]): item.get("question_visuals", [])
        for item in refined.get("questions", [])
        if isinstance(item, dict)
        and isinstance(item.get("index"), int)
        and item["index"] in candidate_indexes
        and isinstance(item.get("question_visuals"), list)
    }


async def extract_questions_from_image(file: UploadFile, grade_level: int) -> dict[str, Any]:
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="请上传 JPG、PNG、WebP、BMP、GIF 或 TIFF 格式的试卷图片")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="请先在 .env 中配置 OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    image_bytes = await file.read()
    image_bytes, image_mime_type = normalize_uploaded_image(image_bytes, file.content_type)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{_build_extraction_prompt(grade_level)}\n\n{CLEAN_QUESTION_IMAGE_RULES}\n\n{ANSWER_EVALUATION_RULES}\n\n{VISUAL_ASSET_RULES}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ],
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
        raise HTTPException(status_code=502, detail=f"模型接口返回错误：{exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"模型接口请求失败：{exc}") from exc

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    result = _extract_json(content)
    result.setdefault("paper_title", "")
    result.setdefault("page_mark", "")
    result.setdefault("questions", [])
    refined_visuals = await _refine_question_visuals(
        image_bytes,
        image_mime_type,
        result["questions"],
        api_key,
        base_url,
        model,
    )
    for index, question in enumerate(result["questions"]):
        if isinstance(question, dict) and question.get("has_image") is True:
            question["question_visuals"] = refined_visuals.get(index, [])
            question["has_image"] = bool(question["question_visuals"])
    return result


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
