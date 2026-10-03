"""
EduGenie - QnA Module
--------------------
Answers general knowledge and academic questions using Google Gemini 1.5 Pro
through the Google AI Studio API.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()  # reads the .env file so the API key stays out of your code

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Gemini's free tier often answers 503 "high demand"; retry a few times.
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

SYSTEM_PROMPT = (
    "You are EduGenie, a friendly and precise AI learning assistant for students. "
    "Answer questions accurately, clearly and concisely. Use short paragraphs or "
    "bullet points. If a question is unclear, state your assumption and still help."
)


def _attempt(prompt, temperature, max_tokens, api_key):
    """One Gemini call. Returns (text, error). Supports both Google GenAI SDKs."""
    # SDK v1: google-genai
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "system_instruction": SYSTEM_PROMPT,
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini request failed: {exc}"

    # SDK v0: google-generativeai
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            DEFAULT_MODEL, system_instruction=SYSTEM_PROMPT
        )
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
        return None, f"Gemini request failed: {exc}"


def _generate(prompt, temperature=0.4, max_tokens=1024):
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
            break  # not a transient overload, do not retry
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None, error


def answer_question(question):
    """Return a clear answer to a student question.

    Returns (answer_text, error_message).
    """
    question = (question or "").strip()
    if not question:
        return None, "Please enter a question."

    prompt = f"Answer the following student question clearly and concisely.\n\nQuestion: {question}"
    text, error = _generate(prompt, temperature=0.3, max_tokens=1024)
    if error:
        return None, error
    return text, None