import os

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


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import Category

    Base.metadata.create_all(bind=engine)
    _ensure_question_columns()
    _ensure_paper_columns()

    default_categories = [
        "\u9009\u62e9\u9898",
        "\u586b\u7a7a\u9898",
        "\u5e94\u7528\u9898",
        "\u8ba1\u7b97\u9898",
        "\u5224\u65ad\u9898",
    ]
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
        db.commit()


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
