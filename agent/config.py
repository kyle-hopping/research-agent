"""
config.py — All the settings in one file.

Values are from the local `.env` file (see `.env.example`).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Read the .env file in the project root (if it exists) into os.environ.
load_dotenv()

@dataclass(frozen=True)
class Settings:
    """
    Immutable bundle of settings.
    `frozen=True` stops accidental changes at runtime.
    """

    # Groq key from https://console.groq.com/keys.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Groq model to use. openai/gpt-oss-120b is the default with tool-calling
    # support.
    model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Folder where Chroma stores its vector database on disk.
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")

    # Notes further than this from the question aren't auto-added to context.
    # Lower = stricter. Tune with --verbose if recall feels too loose or tight.
    memory_max_distance: float = float(os.getenv("MEMORY_MAX_DISTANCE", "1.5"))

    # Safety: the most "think -> use tool -> think" rounds per question.
    # Stops the agent from looping forever (and burning tokens).
    max_steps: int = int(os.getenv("MAX_STEPS", "5"))

settings = Settings()