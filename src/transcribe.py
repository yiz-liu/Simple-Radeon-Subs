import argparse
from pathlib import Path
from typing import Optional

import torch
import whisper
from whisper.utils import get_writer

from src.config import MODELS_DIR
from src.logger import logger


class Transcriber:
    """
    Handles audio transcription using OpenAI Whisper.
    Optimized for GPU (ROCm/CUDA) and high-quality SRT output.
    """

    def __init__(
        self, model_name: str = "large-v3-turbo", device: Optional[str] = None
    ):
        """
        Initialize the Transcriber.

        Args:
            model_name: The name of the Whisper model to use.
            device: The device to run on ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None  # Lazy loading

    def _load_model(self):
        """Internal method to load the model into memory."""
        if self.model is None:
            logger.info(
                "Loading Whisper model '%s' on %s...", self.model_name, self.device
            )
            self.model = whisper.load_model(
                self.model_name, device=self.device, download_root=str(MODELS_DIR)
            )
            logger.info("Model loaded successfully.")

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: Optional[str | Path] = None,
        language: Optional[str] = None,
        verbose: bool = True,
    ) -> Path:
        """
        Transcribes the audio file and saves the result as an SRT file.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the SRT file. Defaults to audio file's directory.
            language: Optional language code (e.g., 'en', 'ja'). If None, auto-detects.
            verbose: Whether to print progress to console.

        Returns:
            Path: The path to the generated SRT file.
        """
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        output_dir = Path(output_dir or audio_file.parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

        logger.info("Transcribing %s...", audio_file.name)

        # Performance notes:
        # fp16=True is generally faster on supported GPUs.
        # language=None triggers auto-detection.
        result = self.model.transcribe(
            str(audio_file),
            fp16=True if self.device == "cuda" else False,
            language=language,
            verbose=verbose,
            condition_on_previous_text=False,
        )

        # Export to SRT using Whisper's official utility
        # This ensures we follow standard formatting correctly.
        writer = get_writer("srt", str(output_dir))

        # writer() expects (result, original_audio_path_or_filename, options)
        # We pass an empty dict for options as we use defaults.
        writer(result, str(audio_file), {})

        srt_path = output_dir / f"{audio_file.stem}.srt"
        logger.info("Transcription complete: %s", srt_path)

        return srt_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper."
    )
    parser.add_argument("input", help="Path to the input audio file.")
    parser.add_argument("-o", "--output-dir", help="Directory to save the SRT file.")
    parser.add_argument(
        "-m",
        "--model",
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo).",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Language of the audio (e.g., 'en', 'zh'). Auto-detects if omitted.",
    )
    parser.add_argument("--device", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Minimize console output."
    )

    args = parser.parse_args()

    try:
        transcriber = Transcriber(model_name=args.model, device=args.device)
        transcriber.transcribe(
            args.input,
            output_dir=args.output_dir,
            language=args.language,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
