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

Set the model credentials before using image extraction and question generation.
For Zhipu AI:

```powershell
copy .env.example .env
# Fill ZHIPU_API_KEY in .env
AI_PROVIDER=zhipu
ZHIPU_API_KEY=your_key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_VISION_MODEL=glm-4v-plus
ZHIPU_TEXT_MODEL=glm-4-plus
ZHIPU_IMAGE_MODEL=cogview-3-flash
```

The project also supports OpenAI-compatible settings by switching `AI_PROVIDER=openai`
and filling `OPENAI_API_KEY`.

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
