# Exam Bank API

FastAPI project for storing scanned exam questions and later generating papers from the question bank.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

The default database is MySQL. Create a database named `exam_bank`, then set `.env`:

```env
DATABASE_URL=mysql+pymysql://root:你的密码@127.0.0.1:3306/exam_bank?charset=utf8mb4
```

Set the model credentials before using image extraction:

```powershell
copy .env.example .env
# Fill OPENAI_API_KEY in .env
```

## Database Tables

- `question`: questions, answers, options, grade level, stem, category and type links
- `category`: question categories
- `question_type`: type statistics, including total/correct/error counts and accuracy

## Check Connection

Open `http://127.0.0.1:8000/health`.

## Upload Page

Open `http://127.0.0.1:8000/` and upload a scanned paper image.

The extraction API is:

```http
POST /api/extract-questions
Content-Type: multipart/form-data
grade_level=<1|2|3>
file=<JPG|PNG|WebP>
```
