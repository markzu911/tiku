from datetime import datetime

from sqlalchemy import BOOLEAN, DECIMAL, DateTime, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Category(Base):
    __tablename__ = "category"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)

    questions: Mapped[list["Question"]] = relationship(back_populates="category")


class QuestionType(Base):
    __tablename__ = "question_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_type: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[float] = mapped_column(DECIMAL(5, 2), nullable=False, default=0)
    priority: Mapped[str | None] = mapped_column(String(50), nullable=True)

    questions: Mapped[list["Question"]] = relationship(back_populates="type")


class Paper(Base):
    __tablename__ = "paper"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    group_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    group_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    page_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paper_image: Mapped[bytes] = mapped_column(LargeBinary(length=16_777_215), nullable=False)
    paper_image_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    questions: Mapped[list["Question"]] = relationship(back_populates="paper")


class GeneratedPaper(Base):
    __tablename__ = "generated_paper"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_label: Mapped[str | None] = mapped_column(String(500), nullable=True)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_json: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT, "mysql"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


class Question(Base):
    __tablename__ = "question"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_text: Mapped[str] = mapped_column(String(2000), nullable=False)
    answer: Mapped[str] = mapped_column(String(500), nullable=False)
    student_answer: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(BOOLEAN, nullable=True)
    A: Mapped[str | None] = mapped_column(String(500), nullable=True)
    B: Mapped[str | None] = mapped_column(String(500), nullable=True)
    C: Mapped[str | None] = mapped_column(String(500), nullable=True)
    D: Mapped[str | None] = mapped_column(String(500), nullable=True)
    grade_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    question_stem: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    question_image: Mapped[bytes | None] = mapped_column(LargeBinary(length=16_777_215), nullable=True)
    question_image_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    paper_id: Mapped[int | None] = mapped_column(ForeignKey("paper.id"), nullable=True, index=True)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("category.id"), nullable=True, index=True)
    type_id: Mapped[int | None] = mapped_column(ForeignKey("question_type.id"), nullable=True, index=True)

    paper: Mapped[Paper | None] = relationship(back_populates="questions")
    category: Mapped[Category | None] = relationship(back_populates="questions")
    type: Mapped[QuestionType | None] = relationship(back_populates="questions")
