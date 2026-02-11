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

# Tools Directory
TOOLS_DIR = PROJECT_ROOT / "tools"
FFMPEG_LOCAL_PATH = TOOLS_DIR / "ffmpeg" / "ffmpeg"

# Models Directory
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"
