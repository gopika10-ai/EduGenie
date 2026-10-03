"""
EduGenie - Learning Path Module
-------------------------------
Generates a personalised, structured learning path for any topic using Google
Gemini: beginner to advanced concepts, organised by difficulty and supported
with resources such as videos, articles and books.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()  # reads the .env file so the API key stays out of your code

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Gemini's free tier often answers 503 "high demand"; retry a few times.
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

PROMPT_TEMPLATE = """You are a learning-path designer for EduGenie.

Create a personalised, structured learning path for the topic below.

Rules:
- Organise the path into Beginner, Intermediate and Advanced stages.
- For each stage list 3 to 5 concrete topics in the order they should be learned.
- Under each topic add one useful resource: a video, article, book or free course.
- Mark the estimated time for each stage.
- End with a short "How to practise" section.
- Adapt the depth to a {level} learner.
- Be specific and practical. Do not add filler.

TOPIC: {topic}
LEARNER LEVEL: {level}

Respond in Markdown.
"""


def _attempt(prompt, temperature, max_tokens, api_key):
    """One Gemini call. Returns (text, error). Supports both Google GenAI SDKs."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return None, f"Learning path generation failed: {exc}"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Learning path generation failed: {exc}"


def _generate(prompt, temperature=0.5, max_tokens=2048):
    """Call Gemini with retries. Returns (text, error)."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, (
            "GEMINI_API_KEY is not set. Create a key at "
            "https://aistudio.google.com/apikey and add it to your environment."
        )

    text, error = None, None
    for attempt in range(MAX_RETRIES):
        text, error = _attempt(prompt, temperature, max_tokens, api_key)
        if not error:
            return text, None
        if "503" not in error and "UNAVAILABLE" not in error.upper():
            break
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None, error


def get_learning_recommendations(topic, level="beginner"):
    """Return a personalised learning path for a topic.

    Returns (learning_path_markdown, error_message).
    """
    topic = (topic or "").strip()
    if not topic:
        return None, "Please enter a topic you want to learn."

    level = (level or "beginner").strip().lower()
    if level not in {"beginner", "intermediate", "advanced"}:
        level = "beginner"

    prompt = PROMPT_TEMPLATE.format(topic=topic, level=level)
    path, error = _generate(prompt, temperature=0.5, max_tokens=2048)
    if error:
        return None, error
    if not path or not path.strip():
        return None, "Gemini returned an incomplete learning path. Please try again."
    return path, None