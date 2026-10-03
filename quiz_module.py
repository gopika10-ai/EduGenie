"""
EduGenie - Quiz Module
----------------------
Generates three multiple-choice questions (MCQs), each with four options,
from a given passage using Google Gemini. Output is parsed into JSON so the
frontend can grade the learner.
"""

import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()  # reads the .env file so the API key stays out of your code

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Gemini's free tier often answers 503 "high demand"; retry a few times.
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

PROMPT_TEMPLATE = """You are an exam setter for EduGenie.

Read the passage below and create exactly 3 multiple-choice questions.

Rules:
- Each question must have exactly 4 options.
- Only one option is correct.
- Distractors must be plausible but clearly wrong.
- Questions must be answerable from the passage alone.
- "answer" must be the 0-based index of the correct option.

PASSAGE:
{passage}

Respond with JSON only, in exactly this shape, and nothing else:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0}}]}}
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
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return None, f"Quiz generation failed: {exc}"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Quiz generation failed: {exc}"


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


def clean_json_block(text):
    """Strip Markdown code fences and surrounding prose from a model response."""
    if not text:
        return ""

    cleaned = text.strip()

    # Remove ```json ... ``` fences
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Keep only the outermost JSON array or object
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    if start_obj == -1 and start_arr == -1:
        return cleaned

    if start_arr == -1 or (start_obj != -1 and start_obj < start_arr):
        start = start_obj
        end_char = "}"
    else:
        start = start_arr
        end_char = "]"

    end = cleaned.rfind(end_char)
    if end == -1 or end < start:
        return cleaned[start:]

    return cleaned[start : end + 1]


def _normalize(raw_questions):
    """Force every item into {question, options[4], answer:int}."""
    normalized = []
    for item in raw_questions or []:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or item.get("q") or "").strip()
        options = item.get("options") or item.get("choices") or []
        if not isinstance(options, list):
            options = [options]
        options = [str(o).strip() for o in options if str(o).strip()][:4]
        while len(options) < 4:
            options.append(f"Option {len(options) + 1}")

        answer = item.get("answer", item.get("correct_answer", item.get("correct")))
        try:
            answer = int(answer)
        except (TypeError, ValueError):
            answer = -1

        # Accept a letter or full option text as the answer
        if answer < 0:
            for idx, opt in enumerate(options):
                if str(item.get("answer", "")).strip().lower() == opt.lower():
                    answer = idx
                    break
        if answer == -1:
            for idx, opt in enumerate(options):
                letter = chr(ord("a") + idx)
                if str(item.get("answer", "")).strip().lower() == letter:
                    answer = idx
                    break
        if not 0 <= answer < 4:
            answer = 0

        if question:
            normalized.append({"question": question, "options": options, "answer": answer})

    return normalized[:3]


def generate_quiz(passage):
    """Generate a 3-question MCQ quiz from a passage.

    Returns (questions_list, error_message).
    """
    passage = (passage or "").strip()
    if not passage:
        return None, "Please enter a passage or topic to build a quiz from."
    if len(passage) < 40:
        return None, "Please provide a longer passage (at least 40 characters)."

    prompt = PROMPT_TEMPLATE.format(passage=passage)
    raw, error = _generate(prompt, temperature=0.5, max_tokens=2048)
    if error:
        return None, error

    cleaned = clean_json_block(raw)

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Could not parse the quiz JSON: {exc} (raw response: {raw[:300]})"

    if isinstance(data, dict):
        data = data.get("questions") or data.get("quiz") or []
    if not isinstance(data, list) or not data:
        return None, f"Quiz response was empty or malformed (raw response: {raw[:300]})"

    questions = _normalize(data)
    if not questions:
        return None, f"Quiz response could not be normalised (raw response: {raw[:300]})"

    return questions, None