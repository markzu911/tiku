import json
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from typing import Any

from PIL import Image
from sqlalchemy.orm import Session

from app.models import Category, Paper, Question, QuestionType


def save_extracted_questions(
    db: Session,
    extracted_questions: list[dict[str, Any]],
    grade_level: int,
    source_image: bytes | None = None,
    source_image_mime_type: str | None = None,
    paper_name: str | None = None,
    paper_info: dict[str, Any] | None = None,
    paper_group_id: str | None = None,
    paper_group_name: str | None = None,
    paper_page_index: int | None = None,
) -> list[dict[str, Any]]:
    saved_questions = []
    crop_cache = _QuestionCropCache(source_image)
    paper = _create_paper(
        db,
        paper_name,
        source_image,
        source_image_mime_type,
        group_id=paper_group_id,
        group_name=paper_group_name,
        page_index=paper_page_index,
    )
    if paper and paper_info is not None:
        paper_info.update(
            {
                "id": paper.id,
                "name": paper.name,
                "group_id": paper.group_id or str(paper.id),
                "group_name": paper.group_name or paper.name,
                "page_index": paper.page_index,
                "image_url": f"/api/papers/{paper.id}/image",
            }
        )

    for item in extracted_questions:
        question_text = _clean(item.get("question_text"))
        question_box = _normalize_question_box(item.get("question_box"))
        if not question_text or question_box is None:
            continue

        student_answer = _clean(item.get("student_answer"))
        is_correct = _as_bool(item.get("is_correct"))
        if not student_answer:
            student_answer = "未作答"
            is_correct = False

        category = _get_or_create_category(db, _clean(item.get("category_name")))
        question_type = _get_or_create_question_type(db, _clean(item.get("question_type")))
        _update_question_type_stats(question_type, is_correct)
        question_image, question_image_mime_type = crop_cache.crop(question_box)

        question = Question(
            question_no=_clean(item.get("question_no")) or None,
            question_box=json.dumps(question_box, ensure_ascii=False),
            question_text=question_text,
            answer=_clean(item.get("answer")),
            student_answer=student_answer,
            is_correct=is_correct,
            A=_clean(item.get("A")) or None,
            B=_clean(item.get("B")) or None,
            C=_clean(item.get("C")) or None,
            D=_clean(item.get("D")) or None,
            grade_level=grade_level,
            question_stem=_clean(item.get("question_stem")) or None,
            question_image=question_image,
            question_image_mime_type=question_image_mime_type,
            paper=paper,
            category=category,
            type=question_type,
        )
        db.add(question)
        db.flush()

        saved_item = dict(item)
        saved_item.pop("_image_bytes", None)
        saved_item.update(
            {
                "id": question.id,
                "grade_level": grade_level,
                "category_id": category.id if category else None,
                "type_id": question_type.id if question_type else None,
                "paper_id": paper.id if paper else None,
                "paper_name": paper.name if paper else "",
                "paper_group_id": paper.group_id if paper else "",
                "paper_group_name": paper.group_name if paper else "",
                "has_image": question.question_image is not None,
                "question_image_saved": question.question_image is not None,
                "image_url": f"/api/questions/{question.id}/image" if question.question_image else "",
            }
        )
        saved_questions.append(saved_item)

    db.commit()
    return saved_questions


class _QuestionCropCache:
    def __init__(self, image_bytes: bytes | None):
        self.image_bytes = image_bytes
        self._image: Image.Image | None = None

    def crop(self, question_box: dict[str, int]) -> tuple[bytes | None, str | None]:
        image = self._get_image()
        if image is None:
            return None, None

        left = question_box["x"]
        top = question_box["y"]
        right = min(image.width, left + question_box["width"])
        bottom = min(image.height, top + question_box["height"])
        if right <= left or bottom <= top:
            return None, None

        output = BytesIO()
        image.crop((left, top, right, bottom)).save(output, format="PNG")
        return output.getvalue(), "image/png"

    def _get_image(self) -> Image.Image | None:
        if self._image is not None:
            return self._image
        if self.image_bytes is None:
            return None
        try:
            self._image = Image.open(BytesIO(self.image_bytes)).convert("RGB")
        except Exception:
            return None
        return self._image


def _normalize_question_box(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        x = max(0, int(value["x"]))
        y = max(0, int(value["y"]))
        width = int(value["width"])
        height = int(value["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def _create_paper(
    db: Session,
    name: str | None,
    image_bytes: bytes | None,
    mime_type: str | None,
    group_id: str | None = None,
    group_name: str | None = None,
    page_index: int | None = None,
) -> Paper | None:
    if not image_bytes:
        return None
    paper = Paper(
        name=_clean(name) or "未命名试卷",
        group_id=_clean(group_id) or None,
        group_name=_clean(group_name) or _clean(name) or "未命名试卷",
        page_index=page_index,
        paper_image=image_bytes,
        paper_image_mime_type=mime_type or "application/octet-stream",
    )
    db.add(paper)
    db.flush()
    return paper


def _get_or_create_category(db: Session, name: str) -> Category | None:
    if not name:
        return None
    category = db.query(Category).filter(Category.name == name).first()
    if category:
        return category
    category = Category(name=name)
    db.add(category)
    db.flush()
    return category


def _get_or_create_question_type(db: Session, name: str) -> QuestionType | None:
    if not name:
        return None
    question_type = db.query(QuestionType).filter(QuestionType.question_type == name).first()
    if question_type:
        return question_type
    question_type = QuestionType(
        question_type=name,
        total=0,
        correct_count=0,
        error_count=0,
        accuracy=Decimal("0.00"),
    )
    db.add(question_type)
    db.flush()
    return question_type


def _update_question_type_stats(question_type: QuestionType | None, is_correct: Any) -> None:
    if question_type is None or is_correct is None:
        return
    question_type.total += 1
    if is_correct is True:
        question_type.correct_count += 1
    else:
        question_type.error_count += 1
    question_type.accuracy = (
        Decimal(question_type.correct_count * 100) / Decimal(question_type.total)
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
