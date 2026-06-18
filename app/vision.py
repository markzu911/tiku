import base64
import json
import os
from typing import Any

import httpx
from fastapi import HTTPException, UploadFile


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


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
9. 如果图片上有学生作答痕迹，请提取 student_answer；如果没有学生答案，student_answer 为空字符串，is_correct 为 null。
10. 请计算或判断正确答案 answer，并比较 student_answer 是否正确。多空题任意一空错误，整题 is_correct 为 false。
11. 如果题目有 A/B/C/D 选项，分别填入 A、B、C、D；没有选项则为空字符串。
12. 不要识别页眉、页脚、装饰文字、分数栏、批改符号为题目。
13. 保留题目前的特殊要求符号，例如 ☆817÷4= 中的 ☆ 必须保留在 question_text 里。
14. 如果一道题包含无法完整转成文字的信息，例如几何图、角度图、线段图、统计图、表格、示意图、图片材料，必须设置 has_image 为 true。
15. has_image=true 时，question_image_bbox 必须给出“只包含这道题中非文字图形/表格/示意图信息”的裁剪框，不要包含整张卷子，也不要包含大量题干文字。
16. question_image_bbox 使用归一化坐标 [x1, y1, x2, y2]，取值 0 到 1，分别表示相对整张图片宽高的位置。坐标要略微留白，确保图形完整。
17. 如果题目是纯文字题，has_image 为 false，question_image_bbox 为 null。
18. 对于几何图题，question_text 只写题目文字要求；图形本身通过 question_image_bbox 保存。

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
question_image_bbox 在 has_image=true 时必须是 4 个数字组成的数组，在 has_image=false 时必须是 null。
不要返回 Markdown。不要返回解释。不要返回 JSON 以外的任何内容。
""".strip()


async def extract_questions_from_image(file: UploadFile, grade_level: int) -> dict[str, Any]:
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="请上传 JPG、PNG 或 WebP 格式的试卷图片")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="请先在 .env 中配置 OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_extraction_prompt(grade_level)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{file.content_type};base64,{image_base64}"},
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
    return result
