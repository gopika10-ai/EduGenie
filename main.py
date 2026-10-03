"""
EduGenie - Google Gemini Powered Learning Assistant
---------------------------------------------------
FastAPI backend exposing one REST endpoint per module:

    /qa                       -> qna.answer_question
    /explain                  -> explanation_module.explain_concept
    /quiz                     -> quiz_module.generate_quiz
    /summarize                -> summary_module.summarize_text
    /learn/recommendations    -> learning_path.get_learning_recommendations

Run locally with:  uvicorn main:app --reload
Then open:         http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import explanation_module
import learning_path
import qna
import quiz_module
import summary_module

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
# The frontend posts the same text under several possible keys so that the
# API stays forgiving regardless of which task is selected.
class QaRequest(BaseModel):
    question: str | None = Field(default=None, description="The student's question")
    query: str | None = None
    text: str | None = None

    def value(self) -> str:
        return (self.question or self.query or self.text or "").strip()


class ExplainRequest(BaseModel):
    topic: str | None = None
    question: str | None = None
    text: str | None = None

    def value(self) -> str:
        return (self.topic or self.question or self.text or "").strip()


class QuizRequest(BaseModel):
    passage: str | None = None
    text: str | None = None
    topic: str | None = None

    def value(self) -> str:
        return (self.passage or self.text or self.topic or "").strip()


class SummarizeRequest(BaseModel):
    text: str | None = None
    passage: str | None = None
    topic: str | None = None

    def value(self) -> str:
        return (self.text or self.passage or self.topic or "").strip()


class LearningPathRequest(BaseModel):
    topic: str | None = None
    question: str | None = None
    text: str | None = None
    level: str = Field(default="beginner", description="beginner | intermediate | advanced")

    def value(self) -> str:
        return (self.topic or self.question or self.text or "").strip()


def fail(message: str, status: int = 400) -> JSONResponse:
    """Return a friendly error instead of the raw Gemini/SDK message."""
    text = str(message or "")

    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return JSONResponse(
            status_code=429,
            content={
                "error": (
                    "Gemini's free daily quota is used up (20 requests per day). "
                    "Please try again later, or use a different API key."
                )
            },
        )

    if "503" in text or "UNAVAILABLE" in text:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "Gemini is busy right now (high demand). "
                    "Please wait a few seconds and try again."
                )
            },
        )

    if "API_KEY" in text and "not set" in text:
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "No Gemini API key found. Add your key to the .env file as "
                    "GEMINI_API_KEY=\"your-key\"."
                )
            },
        )

    return JSONResponse(status_code=status, content={"error": text})


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the single-page frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------
@app.post("/qa")
async def qa_endpoint(payload: QaRequest):
    """Answer a question with Gemini."""
    answer, error = qna.answer_question(payload.value())
    if error:
        return fail(error)
    return {"answer": answer}


@app.post("/explain")
async def explain_endpoint(payload: ExplainRequest):
    """Explain a concept (LaMini-Flan-T5 local, Gemini fallback)."""
    explanation, error, source = explanation_module.explain_concept(payload.value())
    if error:
        return fail(error)
    return {"explanation": explanation, "source_model": source}


@app.post("/quiz")
async def quiz_endpoint(payload: QuizRequest):
    """Generate three 4-option MCQs from a passage."""
    questions, error = quiz_module.generate_quiz(payload.value())
    if error:
        return fail(error)
    return {"quiz": questions}


@app.post("/summarize")
async def summarize_endpoint(payload: SummarizeRequest):
    """Summarise a long passage."""
    summary, error = summary_module.summarize_text(payload.value())
    if error:
        return fail(error)
    return {"summary": summary}


@app.post("/learn/recommendations")
async def learning_recommendations_endpoint(payload: LearningPathRequest):
    """Generate a personalised beginner -> advanced learning path."""
    path, error = learning_path.get_learning_recommendations(
        payload.value(), payload.level
    )
    if error:
        return fail(error)
    return {"learning_path": path, "level": payload.level, "topic": payload.value()}