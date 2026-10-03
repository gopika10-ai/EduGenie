"""
EduGenie - Explanation Module
-----------------------------
Uses the local LaMini-Flan-T5-783M model to break complex topics into simple,
readable explanations. The model is instruction-tuned, lightweight and
CPU-compatible, so it runs on modest hardware.

If the local model or its dependencies are unavailable, the module falls back
to Google Gemini so the feature never breaks the application.
"""

import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()  # reads the .env file so the API key stays out of your code

MODEL_NAME = os.getenv("EXPLANATION_MODEL", "MBZUAI/LaMini-Flan-T5-783M")

# Gemini's free tier often answers 503 "high demand"; retry a few times.
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

INSTRUCTION = (
    "Provide a clear, simple explanation of the following concept for a beginner. "
    "Use short sentences and plain words.\n\nConcept: "
)

_lock = threading.Lock()
_pipeline = None
_load_failed = False


def _load_pipeline():
    """Load the LaMini-Flan-T5 text2text pipeline once, lazily."""
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    with _lock:
        if _pipeline is None and not _load_failed:
            try:
                from transformers import pipeline as hf_pipeline

                _pipeline = hf_pipeline(
                    "text2text-generation",
                    model=MODEL_NAME,
                    max_length=512,
                )
            except Exception:  # noqa: BLE001
                _load_failed = True
                _pipeline = None
    return _pipeline


def _gemini_attempt(topic, api_key):
    """One Gemini call for the explanation fallback."""
    prompt = (
        "Explain the following concept to a beginner in simple, clear language. "
        "Use short bullet points and one everyday analogy.\n\n"
        f"Concept: {topic}"
    )
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.4, "max_output_tokens": 768},
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini request failed: {exc}"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 768},
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini request failed: {exc}"


def _gemini_explanation(topic):
    """Fallback explanation path using Google Gemini, with retries."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, (
            "No explanation backend available. Install 'transformers' and 'torch' "
            "for local LaMini-Flan-T5, or set GEMINI_API_KEY for cloud fallback."
        )

    text, error = None, None
    for attempt in range(MAX_RETRIES):
        text, error = _gemini_attempt(topic, api_key)
        if not error:
            return text, None
        if "503" not in error and "UNAVAILABLE" not in error.upper():
            break
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None, error


def _format_bullets(raw):
    """Turn a single blob of generated text into readable bullets."""
    text = (raw or "").strip()
    if not text:
        return ""

    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    out = []
    for i, sentence in enumerate(sentences[:5]):
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith("."):
            sentence += "."
        out.append(f"- {sentence}")
    return "\n".join(out)


def explain_concept(topic):
    """Explain a topic in simple language.

    Returns (explanation_text, error_message, source_model).
    """
    topic = (topic or "").strip()
    if not topic:
        return None, "Please enter a topic to explain.", None

    pipe = _load_pipeline()
    if pipe is not None:
        try:
            result = pipe(INSTRUCTION + topic, max_length=512, num_beams=4)
            raw = result[0]["generated_text"] if result else ""
            text = _format_bullets(raw)
            if text:
                return text, None, MODEL_NAME
        except Exception as exc:  # noqa: BLE001
            # Local model broke at inference time - try cloud instead.
            text, error = _gemini_explanation(topic)
            if text:
                return text, None, os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            return None, f"Local explanation failed ({exc}). {error}", None

    text, error = _gemini_explanation(topic)
    if text:
        return text, None, os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    return None, error, None