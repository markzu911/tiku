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
    crop_cache = _ImageCropCache(source_image, source_image_mime_type)
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
        paper_info["id"] = paper.id
        paper_info["name"] = paper.name
        paper_info["group_id"] = paper.group_id or str(paper.id)
        paper_info["group_name"] = paper.group_name or paper.name
        paper_info["page_index"] = paper.page_index
        paper_info["image_url"] = f"/api/papers/{paper.id}/image"

    for item in extracted_questions:
        question_text = _clean(item.get("question_text"))
        if not question_text:
            continue

        student_answer = _clean(item.get("student_answer"))
        is_correct = _as_bool(item.get("is_correct"))
        if not student_answer:
            student_answer = "未作答"
            is_correct = False

        category = _get_or_create_category(db, _clean(item.get("category_name")))
        question_type = _get_or_create_question_type(db, _clean(item.get("question_type")))
        _update_question_type_stats(question_type, is_correct)
        visual_bboxes = _normalize_visual_bboxes(item.get("question_visuals"))

        question_image, question_image_mime_type = crop_cache.crop(
            None,
            visual_bboxes,
        )

        question = Question(
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
            question_image=question_image if item.get("has_image") is True else None,
            question_image_mime_type=question_image_mime_type if item.get("has_image") is True else None,
            paper=paper,
            category=category,
            type=question_type,
        )
        db.add(question)
        db.flush()

        saved_item = dict(item)
        saved_item["id"] = question.id
        saved_item["grade_level"] = grade_level
        saved_item["category_id"] = category.id if category else None
        saved_item["type_id"] = question_type.id if question_type else None
        saved_item["paper_id"] = paper.id if paper else None
        saved_item["paper_name"] = paper.name if paper else ""
        saved_item["paper_group_id"] = paper.group_id if paper else ""
        saved_item["paper_group_name"] = paper.group_name if paper else ""
        saved_item["has_image"] = question.question_image is not None
        saved_item["question_image_saved"] = question.question_image is not None
        saved_questions.append(saved_item)

    db.commit()
    return saved_questions


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
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


class _ImageCropCache:
    def __init__(self, image_bytes: bytes | None, mime_type: str | None):
        self.image_bytes = image_bytes
        self.mime_type = mime_type or "image/png"
        self._image: Image.Image | None = None

    def crop(self, bbox: Any, visual_bboxes: Any = None) -> tuple[bytes | None, str | None]:
        box = _normalize_bbox(bbox)
        boxes = _normalize_bboxes(visual_bboxes) or ([box] if box else [])
        if self.image_bytes is None or not boxes:
            return None, None

        image = self._get_image()
        if image is None:
            return None, None

        cropped_regions = self._crop_visual_regions(image, boxes)
        if not cropped_regions:
            return None, None

        cropped = self._compose_visual_regions(cropped_regions)
        output = BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue(), "image/png"

    @staticmethod
    def _crop_visual_regions(
        image: Image.Image,
        boxes: list[tuple[float, float, float, float]],
    ) -> list[Image.Image]:
        width, height = image.size
        regions = []
        for x1, y1, x2, y2 in boxes:
            # Model boxes already identify complete visual assets; only retain a slim safety margin.
            padding = 0.012
            left = max(0, int((x1 - padding) * width))
            top = max(0, int((y1 - padding) * height))
            right = min(width, int((x2 + padding) * width))
            bottom = min(height, int((y2 + padding) * height))
            if right > left and bottom > top:
                regions.append(image.crop((left, top, right, bottom)))
        return regions

    @staticmethod
    def _compose_visual_regions(regions: list[Image.Image]) -> Image.Image:
        if len(regions) == 1:
            return regions[0]

        gap = 12
        canvas_width = max(region.width for region in regions)
        canvas_height = sum(region.height for region in regions) + gap * (len(regions) - 1)
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        top = 0
        for region in regions:
            canvas.paste(region, ((canvas_width - region.width) // 2, top))
            top += region.height + gap
        return canvas

    def _get_image(self) -> Image.Image | None:
        if self._image is not None:
            return self._image

        try:
            image = Image.open(BytesIO(self.image_bytes))
            self._image = image.convert("RGB")
        except Exception:
            return None

        return self._image


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None

    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None

    return x1, y1, x2, y2


def _normalize_bboxes(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list):
        return []
    return [box for item in value if (box := _normalize_bbox(item)) is not None]


def _normalize_visual_bboxes(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list):
        return []
    return [
        box
        for item in value
        if isinstance(item, dict) and (box := _normalize_bbox(item.get("bbox"))) is not None
    ]


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
    elif is_correct is False:
        question_type.error_count += 1
    else:
        return

    question_type.accuracy = (
        Decimal(question_type.correct_count * 100) / Decimal(question_type.total)
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
