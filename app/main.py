from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session, joinedload

from app.database import get_db, init_db
from app.generation import generate_similar_questions
from app.models import GeneratedPaper, Paper, Question, QuestionType
from app.question_service import save_extracted_questions
from app.vision import extract_questions_from_image


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Exam Bank API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class GenerateSimilarRequest(BaseModel):
    question_ids: list[int] = Field(default_factory=list)
    category_name: str | None = None
    question_type: str | None = None
    count: int = Field(default=5, ge=1, le=20)


class SaveGeneratedPaperRequest(BaseModel):
    title: str = Field(default="错题巩固练习卷", max_length=255)
    source_label: str | None = Field(default=None, max_length=500)
    questions: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/")
def upload_page():
    return FileResponse(Path("static/index.html"))


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


@app.get("/api/questions")
def list_questions(db: Session = Depends(get_db)):
    questions = (
        db.query(Question)
        .options(joinedload(Question.category), joinedload(Question.type), joinedload(Question.paper))
        .order_by(Question.id.desc())
        .all()
    )
    return {
        "total": len(questions),
        "questions": [_serialize_question(question) for question in questions],
    }


@app.post("/api/generate-similar")
async def generate_similar(request: GenerateSimilarRequest, db: Session = Depends(get_db)):
    query = db.query(Question).options(
        joinedload(Question.category),
        joinedload(Question.type),
        joinedload(Question.paper),
    )

    if request.question_ids:
        source_ids = list(dict.fromkeys([item for item in request.question_ids if item > 0]))[:50]
        query = query.filter(Question.id.in_(source_ids))
    else:
        if request.category_name:
            query = query.filter(Question.category.has(name=request.category_name))
        if request.question_type:
            query = query.filter(Question.type.has(question_type=request.question_type))

    candidates = query.order_by(Question.id.desc()).limit(200).all()
    wrong_questions = [question for question in candidates if _is_wrong_question(question)]
    if not wrong_questions:
        raise HTTPException(status_code=404, detail="no wrong questions found for generation")

    source_questions = [_serialize_question(question) for question in wrong_questions[:10]]
    generated_questions = await generate_similar_questions(source_questions, request.count)
    return {
        "source_count": len(wrong_questions),
        "source_questions": source_questions,
        "questions": generated_questions,
    }


@app.get("/api/papers")
def list_papers(db: Session = Depends(get_db)):
    papers = (
        db.query(Paper)
        .options(joinedload(Paper.questions))
        .order_by(Paper.id.desc())
        .all()
    )
    paper_groups = _serialize_paper_groups(papers)
    return {"total": len(paper_groups), "papers": paper_groups}


@app.get("/api/generated-papers")
def list_generated_papers(db: Session = Depends(get_db)):
    generated_papers = db.query(GeneratedPaper).order_by(GeneratedPaper.id.desc()).all()
    return {
        "total": len(generated_papers),
        "papers": [_serialize_generated_paper(paper) for paper in generated_papers],
    }


@app.post("/api/generated-papers")
def save_generated_paper(request: SaveGeneratedPaperRequest, db: Session = Depends(get_db)):
    questions = [question for question in request.questions if isinstance(question, dict)]
    if not questions:
        raise HTTPException(status_code=400, detail="generated paper requires at least one question")

    title = request.title.strip() or "错题巩固练习卷"
    paper = GeneratedPaper(
        title=title[:255],
        source_label=(request.source_label or "")[:500] or None,
        question_count=len(questions),
        content_json=json.dumps({"questions": questions}, ensure_ascii=False),
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return _serialize_generated_paper(paper)


@app.get("/api/papers/{paper_id}/image")
def get_paper_image(paper_id: int, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None or paper.paper_image is None:
        raise HTTPException(status_code=404, detail="paper image not found")

    return Response(
        content=paper.paper_image,
        media_type=paper.paper_image_mime_type or "application/octet-stream",
    )


@app.get("/api/question-types")
def list_question_types(db: Session = Depends(get_db)):
    question_types = (
        db.query(QuestionType)
        .order_by(QuestionType.total.desc(), QuestionType.question_type.asc())
        .all()
    )
    return {
        "total": len(question_types),
        "question_types": [_serialize_question_type(question_type) for question_type in question_types],
    }


@app.get("/api/questions/{question_id}/image")
def get_question_image(question_id: int, db: Session = Depends(get_db)):
    question = db.query(Question).filter(Question.id == question_id).first()
    if question is None or question.question_image is None:
        raise HTTPException(status_code=404, detail="question image not found")

    return Response(
        content=question.question_image,
        media_type=question.question_image_mime_type or "application/octet-stream",
    )


@app.post("/api/extract-questions")
async def extract_questions(
    grade_level: int = Form(...),
    paper_name: str | None = Form(None),
    files: list[UploadFile] = File(..., alias="file"),
    db: Session = Depends(get_db),
):
    if grade_level not in {1, 2, 3}:
        raise HTTPException(status_code=400, detail="grade_level must be 1, 2, or 3")

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张试卷图片")

    all_questions = []
    saved_papers = []
    group_id = uuid4().hex
    group_name = None

    for index, file in enumerate(files, start=1):
        image_bytes = await file.read()
        await file.seek(0)
        result = await extract_questions_from_image(file, grade_level)

        if group_name is None:
            group_name = _build_paper_group_name(
                result,
                file.filename or f"试卷{index}",
                custom_name=paper_name,
            )

        final_paper_name = _build_paper_name(
            result,
            file.filename or f"试卷{index}",
            index,
            len(files),
            custom_name=group_name,
        )
        paper_info = {}
        saved_questions = save_extracted_questions(
            db,
            result["questions"],
            grade_level,
            source_image=image_bytes,
            source_image_mime_type=file.content_type,
            paper_name=final_paper_name,
            paper_info=paper_info,
            paper_group_id=group_id,
            paper_group_name=group_name,
            paper_page_index=index,
        )
        paper_info.setdefault("name", final_paper_name)
        paper_info["saved_count"] = len(saved_questions)
        saved_papers.append(paper_info)
        all_questions.extend(saved_questions)

    return {
        "saved_count": len(all_questions),
        "paper_count": len(saved_papers),
        "paper_group_id": group_id,
        "paper_group_name": group_name,
        "papers": saved_papers,
        "questions": all_questions,
    }


def _serialize_question(question: Question):
    has_image = question.question_image is not None
    is_correct = _infer_correctness(question)
    return {
        "id": question.id,
        "question_text": question.question_text,
        "question_stem": question.question_stem or "",
        "answer": question.answer,
        "student_answer": question.student_answer or "",
        "A": question.A or "",
        "B": question.B or "",
        "C": question.C or "",
        "D": question.D or "",
        "grade_level": question.grade_level,
        "category_name": question.category.name if question.category else "",
        "question_type": question.type.question_type if question.type else "",
        "paper_id": question.paper_id,
        "paper_name": question.paper.name if question.paper else "",
        "paper_group_id": question.paper.group_id if question.paper else "",
        "paper_group_name": question.paper.group_name if question.paper else "",
        "has_image": has_image,
        "image_url": f"/api/questions/{question.id}/image" if has_image else "",
        "is_correct": is_correct,
        "is_wrong": is_correct is False,
    }


def _serialize_generated_paper(paper: GeneratedPaper):
    try:
        content = json.loads(paper.content_json or "{}")
    except json.JSONDecodeError:
        content = {}

    questions = content.get("questions")
    if not isinstance(questions, list):
        questions = []

    return {
        "id": paper.id,
        "title": paper.title,
        "source_label": paper.source_label or "",
        "question_count": paper.question_count,
        "questions": questions,
        "created_at": paper.created_at.isoformat(sep=" ", timespec="seconds") if paper.created_at else "",
    }


def _is_wrong_question(question: Question) -> bool:
    return _infer_correctness(question) is False


def _infer_correctness(question: Question) -> bool | None:
    answer = _normalize_answer(question.answer)
    student_answer = _normalize_answer(question.student_answer)
    if not student_answer:
        return False
    if not answer:
        return None
    return answer == student_answer


def _normalize_answer(value) -> str:
    text_value = str(value or "").strip().lower()
    replacements = {
        " ": "",
        "\t": "",
        "\r": "",
        "\n": "",
        "；": ";",
        "，": ",",
        "。": "",
        "：": ":",
    }
    for source, target in replacements.items():
        text_value = text_value.replace(source, target)
    return text_value


def _serialize_paper_groups(papers: list[Paper]):
    groups = {}
    for paper in papers:
        key = paper.group_id or str(paper.id)
        if key not in groups:
            groups[key] = {
                "id": key,
                "name": paper.group_name or paper.name,
                "paper_ids": [],
                "images": [],
                "question_count": 0,
            }

        groups[key]["paper_ids"].append(paper.id)
        groups[key]["question_count"] += len(paper.questions)
        groups[key]["images"].append(
            {
                "paper_id": paper.id,
                "name": paper.name,
                "page_index": paper.page_index or len(groups[key]["images"]) + 1,
                "image_url": f"/api/papers/{paper.id}/image",
            }
        )

    result = list(groups.values())
    for group in result:
        group["images"].sort(key=lambda image: (image["page_index"], image["paper_id"]))
    return result


def _build_paper_group_name(result: dict, fallback_name: str, custom_name: str | None = None) -> str:
    custom = _clean_name(custom_name)
    title = _clean_name(result.get("paper_title"))
    return custom or title or _strip_image_extension(fallback_name) or "未命名试卷"


def _build_paper_name(
    result: dict,
    fallback_name: str,
    index: int,
    total: int,
    custom_name: str | None = None,
) -> str:
    base_name = _build_paper_group_name(result, fallback_name, custom_name=custom_name)
    page_mark = _clean_name(result.get("page_mark"))

    if page_mark:
        return f"{base_name}-{page_mark}"
    if total > 1:
        return f"{base_name}-第{index}张"
    return base_name


def _clean_name(value) -> str:
    return str(value or "").strip().replace("/", "-").replace("\\", "-")


def _strip_image_extension(filename: str) -> str:
    name = Path(filename or "").name
    for suffix in [".jpeg", ".jpg", ".png", ".webp"]:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _serialize_question_type(question_type: QuestionType):
    total = question_type.total or 0
    correct_count = question_type.correct_count or 0
    error_count = question_type.error_count or 0
    accuracy = float(question_type.accuracy or 0)
    error_rate = round(error_count * 100 / total, 2) if total else 0.0
    return {
        "id": question_type.id,
        "question_type": question_type.question_type,
        "total": total,
        "correct_count": correct_count,
        "error_count": error_count,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "priority": question_type.priority or "",
    }
