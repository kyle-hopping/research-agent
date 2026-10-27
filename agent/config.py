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

    # Multi-agent research team (research.py)
    # Model for the team. Defaults to GROQ_MODEL. A model with a higher
    # tokens-per-minute limit hits fewer rate limits, since several agents
    # share one quota.
    team_model: str = os.getenv("TEAM_MODEL", "") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # How many researcher agents may call Groq at the same moment.
    team_max_concurrency: int = int(os.getenv("TEAM_MAX_CONCURRENCY", "3"))

    # Seconds before a researcher is cancelled and reported as failed.
    researcher_timeout: float = float(os.getenv("RESEARCHER_TIMEOUT", "120"))

    # Tool rounds (web searches) each researcher gets before it must answer.
    researcher_max_steps: int = int(os.getenv("RESEARCHER_MAX_STEPS", "4"))

    # How many times the writer may revise after the critic finds problems.
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "1"))

settings = Settings()