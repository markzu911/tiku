import os
import re

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@127.0.0.1:3306/exam_bank?charset=utf8mb4",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
CATEGORY_NAMES = ("\u586b\u7a7a\u9898", "\u9009\u62e9\u9898", "\u5224\u65ad\u9898", "\u8ba1\u7b97\u9898", "\u5e94\u7528\u9898", "\u753b\u56fe\u9898")


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import Category, Question

    Base.metadata.create_all(bind=engine)
    _ensure_question_columns()
    _ensure_paper_columns()

    default_categories = CATEGORY_NAMES
    with SessionLocal() as db:
        existing = {
            name
            for (name,) in db.query(Category.name)
            .filter(Category.name.in_(default_categories))
            .all()
        }
        for name in default_categories:
            if name not in existing:
                db.add(Category(name=name))
        db.flush()
        categories = {
            category.name: category
            for category in db.query(Category).filter(Category.name.in_(CATEGORY_NAMES)).all()
        }
        for question in db.query(Question).all():
            question.category = categories[_infer_category_name(question)]
        db.commit()


def _infer_category_name(question) -> str:
    text = "\n".join(
        str(value or "")
        for value in (question.question_text, question.question_stem, question.A, question.B, question.C, question.D)
    )
    if any(str(getattr(question, key) or "").strip() for key in ("A", "B", "C", "D")) or re.search(r"(?m)^\s*[A-D][\u3001.\uff0e]", text):
        return "\u9009\u62e9\u9898"
    if re.search(r"\u5224\u65ad|\u5bf9\u7684|\u9519\u7684|\u6b63\u786e|\u9519\u8bef|\u221a|\xd7", text):
        return "\u5224\u65ad\u9898"
    if re.search(r"\u753b\u56fe|\u4f5c\u56fe|\u753b\u4e00\u753b|\u8fde\u7ebf|\u6d82\u8272|\u62fc\u56fe|\u6298\u4e00\u6298", text):
        return "\u753b\u56fe\u9898"
    if re.search(r"\u5e94\u7528\u9898|\u89e3\u51b3\u95ee\u9898|\u4e00\u5171|\u8fd8\u5269|\u6bd4.*(?:\u591a|\u5c11)|\u4e70|\u79df|\u8def\u7a0b|\u6bcf.*(?:\u5143|\u4eba|\u4e2a|\u7c73|\u5343\u514b|\u5206\u949f)", text):
        return "\u5e94\u7528\u9898"
    if re.search(r"\u8ba1\u7b97|\u53e3\u7b97|\u7ad6\u5f0f|\u8131\u5f0f|\u7b97\u5f0f|\u5f97\u6570|\d\s*[+\-\xb1\xd7xX\xf7/]\s*\d", text):
        return "\u8ba1\u7b97\u9898"
    return "\u586b\u7a7a\u9898"


def _ensure_question_columns():
    inspector = inspect(engine)
    if not inspector.has_table("question"):
        return

    columns = {column["name"] for column in inspector.get_columns("question")}
    with engine.begin() as connection:
        if "student_answer" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN student_answer VARCHAR(500) NULL"))
        if "is_correct" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN is_correct BOOLEAN NULL"))
        if "question_image" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN question_image LONGBLOB NULL"))
        if "question_image_mime_type" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN question_image_mime_type VARCHAR(100) NULL"))
        if "question_no" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN question_no VARCHAR(50) NULL"))
        if "question_box" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN question_box TEXT NULL"))
        if "paper_id" not in columns:
            connection.execute(text("ALTER TABLE question ADD COLUMN paper_id INT NULL"))
            connection.execute(text("CREATE INDEX ix_question_paper_id ON question (paper_id)"))


def _ensure_paper_columns():
    inspector = inspect(engine)
    if not inspector.has_table("paper"):
        return

    columns = {column["name"] for column in inspector.get_columns("paper")}
    with engine.begin() as connection:
        if "group_id" not in columns:
            connection.execute(text("ALTER TABLE paper ADD COLUMN group_id VARCHAR(64) NULL"))
            connection.execute(text("CREATE INDEX ix_paper_group_id ON paper (group_id)"))
        if "group_name" not in columns:
            connection.execute(text("ALTER TABLE paper ADD COLUMN group_name VARCHAR(255) NULL"))
            connection.execute(text("CREATE INDEX ix_paper_group_name ON paper (group_name)"))
        if "page_index" not in columns:
            connection.execute(text("ALTER TABLE paper ADD COLUMN page_index INT NULL"))
