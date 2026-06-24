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

To access the project from another device on the same LAN, start it with:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://<this-computer's-LAN-IP>:8000/` from the other device. Allow inbound TCP port 8000 on the Windows Private network if prompted.

## Run with Docker

Docker Desktop must be running. This starts both the API and a persistent MySQL container. The application database is stored in the `exam-bank-mysql-data` Docker volume.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker-stack.ps1
```

To copy the current host MySQL data into the Docker database once, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker-stack.ps1 -MigrateHostData
```

Open `http://127.0.0.1:8000/`. Model credentials remain outside the image in `.env`.

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
