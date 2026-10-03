"""
EduGenie - Summary Module
-------------------------
Summarises long educational passages into concise, revision-friendly text
using Google Gemini's generative capabilities.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()  # reads the .env file so the API key stays out of your code

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Gemini's free tier often answers 503 "high demand"; retry a few times.
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

PROMPT_TEMPLATE = """Summarise the passage below for a student who wants quick revision.

Rules:
- Keep every core fact, term and conclusion.
- Remove repetition, filler and hedging.
- Use short bullet points, then one closing sentence.
- Stay faithful to the passage; never add outside information.
- Aim for about 25% of the original length.

PASSAGE:
{text}
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
        return None, f"Summarisation failed: {exc}"

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
        return None, f"Summarisation failed: {exc}"


def _generate(prompt, temperature=0.3, max_tokens=1024):
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


def summarize_text(text):
    """Summarise a passage.

    Returns (summary, error_message).
    """
    text = (text or "").strip()
    if not text:
        return None, "Please enter the text you want summarised."
    if len(text) < 60:
        return None, "Please provide a longer passage (at least 60 characters)."

    prompt = PROMPT_TEMPLATE.format(text=text)
    summary, error = _generate(prompt, temperature=0.3, max_tokens=1024)
    if error:
        return None, error
    return summary, None