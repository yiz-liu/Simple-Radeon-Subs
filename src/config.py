from pathlib import Path
from typing import Final, Literal

type ASRBackend = Literal["whisper", "qwen"]

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_OUTPUT_TAIL_BYTES: Final = 8192

# Audio extraction
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"
AUDIO_TIMELINE_FILTER: Final = (
    f"aresample={AUDIO_SAMPLE_RATE}:async=1:first_pts=0:min_hard_comp=0.001"
)

# Whisper transcription
WHISPER_CLI_PATH: Final = PROJECT_ROOT / ".venv" / "bin" / "whisper-cli"
WHISPER_MODEL_PATH: Final = MODELS_DIR / "whisper" / "ggml-large-v3-turbo.bin"
WHISPER_VAD_MODEL_PATH: Final = MODELS_DIR / "whisper" / "ggml-silero-v6.2.0.bin"

# Qwen model inference
QWEN_ASR_MODEL_PATH: Final = MODELS_DIR / "Qwen3-ASR-1.7B"
QWEN_ALIGNER_MODEL_PATH: Final = MODELS_DIR / "Qwen3-ForcedAligner-0.6B"
QWEN_GPU_MEMORY_UTILIZATION: Final = 0.5
QWEN_MAX_MODEL_LEN: Final = 8192
QWEN_MAX_NUM_SEQS: Final = 4
QWEN_MAX_BATCHED_TOKENS: Final = 8192
QWEN_MAX_OUTPUT_TOKENS: Final = 512
QWEN_REPETITION_PENALTY: Final = 1.2
QWEN_REPETITION_MIN_PATTERN_SIZE: Final = 1
QWEN_REPETITION_MAX_PATTERN_SIZE: Final = 20
QWEN_REPETITION_MIN_COUNT: Final = 30

# Qwen speech detection
QWEN_VAD_MODEL_PATH: Final = MODELS_DIR / "silero-vad" / "silero_vad.onnx"
QWEN_VAD_THRESHOLD: Final = 0.01
QWEN_VAD_MIN_SPEECH_MS: Final = 250
QWEN_VAD_MIN_SILENCE_MS: Final = 100
QWEN_VAD_PAD_MS: Final = 50
QWEN_VAD_MAX_SECONDS: Final = 180

# Qwen windows and subtitle timing
QWEN_WINDOW_TARGET_SECONDS: Final = 30
QWEN_WINDOW_MERGE_GAP_SECONDS: Final = 1
QWEN_SHORT_WINDOW_SECONDS: Final = 3
QWEN_FALLBACK_MIN_SECONDS: Final = 0.32

# Translation
TRANSLATION_MODEL_PATH: Final = MODELS_DIR / "Qwen3.5-9B-AWQ-4bit"
