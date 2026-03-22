import os
from pathlib import Path

from dotenv import load_dotenv

# Project Root (Assumes this file is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")

# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default to Google AI Studio endpoint for Gemini 3
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
)

# OpenAI Compatible API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1/chat/completions",
)
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o")

# vLLM Settings
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH")

# Translation Provider Selection
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "gemini")

# Tools Directory
TOOLS_DIR = PROJECT_ROOT / "tools"
FFMPEG_LOCAL_PATH = TOOLS_DIR / "ffmpeg" / "ffmpeg"

# Models Directory
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Inject local ffmpeg into PATH so third-party libs (e.g. Whisper) can find it
if FFMPEG_LOCAL_PATH.exists():
    os.environ["PATH"] = (
        str(FFMPEG_LOCAL_PATH.parent) + os.pathsep + os.environ.get("PATH", "")
    )

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"
