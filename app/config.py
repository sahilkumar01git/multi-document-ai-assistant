import os

from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "Multi-Document AI Assistant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_api_key(secrets=None):
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if api_key:
        return api_key

    if secrets is not None:
        return secrets.get("GROQ_API_KEY") or secrets.get("groq_api_key")

    return None
