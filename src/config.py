from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_CLI_PATH: Final = PROJECT_ROOT / ".venv" / "bin" / "whisper-cli"
WHISPER_MODEL_PATH: Final = MODELS_DIR / "whisper" / "ggml-large-v3-turbo.bin"
WHISPER_VAD_MODEL_PATH: Final = MODELS_DIR / "whisper" / "ggml-silero-v6.2.0.bin"
TRANSLATION_MODEL_PATH: Final = MODELS_DIR / "Qwen3.5-9B-AWQ-4bit"

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"
