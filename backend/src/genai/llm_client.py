import json
import logging
import os

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(dotenv_path=env_path)
except ImportError:
    logging.warning("python-dotenv module is not installed. Will use system environment variables.")

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logging.warning("google-generativeai module is not installed. LLM features will be disabled.")

MODEL_NAME = "gemini-2.5-flash"


def _api_key() -> str | None:
    key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


def is_llm_available() -> bool:
    return HAS_GENAI and _api_key() is not None


def get_gemini_client():
    """Initializes and returns the Gemini generative model, or None without a key."""
    if not is_llm_available():
        return None
    genai.configure(api_key=_api_key())
    return genai.GenerativeModel(MODEL_NAME)


def generate_text(prompt: str) -> str | None:
    client = get_gemini_client()
    if client is None:
        return None
    response = client.generate_content(prompt)
    return response.text or None


def generate_json(prompt: str) -> dict | None:
    """Gemini in JSON mode; returns None if unavailable or the reply isn't a JSON object."""
    client = get_gemini_client()
    if client is None:
        return None
    response = client.generate_content(
        prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
    try:
        parsed = json.loads(response.text)
    except (TypeError, ValueError):
        logging.warning("Gemini returned non-JSON output")
        return None
    return parsed if isinstance(parsed, dict) else None
