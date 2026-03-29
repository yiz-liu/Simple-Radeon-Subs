import argparse
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import whisper
from whisper.utils import get_writer

from src.config import MODELS_DIR
from src.logger import logger
from src.vad import get_clip_timestamps, validate_clip_timestamps

whisper_transcribe_module = importlib.import_module("whisper.transcribe")


def _parse_clip_timestamps(clip_timestamps: list[float] | str) -> list[float]:
    if isinstance(clip_timestamps, str):
        raw_values = [float(value) for value in clip_timestamps.split(",") if value.strip()]
        return validate_clip_timestamps(raw_values)

    return validate_clip_timestamps(clip_timestamps)


def _get_clip_progress_total_frames(clip_timestamps: list[float] | str) -> int:
    parsed_timestamps = _parse_clip_timestamps(clip_timestamps)
    total_seconds = 0.0
    for index in range(0, len(parsed_timestamps), 2):
        total_seconds += parsed_timestamps[index + 1] - parsed_timestamps[index]

    return max(1, round(total_seconds * whisper.audio.FRAMES_PER_SECOND))


@contextmanager
def _clip_aware_progress_bar(clip_timestamps: Optional[list[float] | str], verbose: Optional[bool]):
    if clip_timestamps is None or verbose is not False:
        yield
        return

    original_tqdm = whisper_transcribe_module.tqdm.tqdm
    progress_total = _get_clip_progress_total_frames(clip_timestamps)

    def clip_aware_tqdm(*args, **kwargs):
        kwargs["total"] = progress_total
        return original_tqdm(*args, **kwargs)

    whisper_transcribe_module.tqdm.tqdm = clip_aware_tqdm
    try:
        yield
    finally:
        whisper_transcribe_module.tqdm.tqdm = original_tqdm


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
        verbose: Optional[bool] = False,
        clip_timestamps: Optional[list[float] | str] = None,
    ) -> Path:
        """
        Transcribes the audio file and saves the result as an SRT file.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the SRT file. Defaults to audio file's directory.
            language: Optional language code (e.g., 'en', 'ja'). If None, auto-detects.
            verbose: Whisper console output mode. True prints decoded text,
                False shows Whisper's built-in tqdm progress bar, and None is silent.
            clip_timestamps: Optional Whisper clip_timestamps value used to constrain
                decoding to known speech regions.

        Returns:
            Path: The path to the generated SRT file.
        """
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        output_dir = Path(output_dir or audio_file.parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = output_dir / f"{audio_file.stem}.srt"

        if clip_timestamps == [] or clip_timestamps == "":
            srt_path.write_text("", encoding="utf-8")
            logger.warning(
                "No speech regions were provided for %s. Wrote empty transcription: %s",
                audio_file.name,
                srt_path,
            )
            return srt_path

        self._load_model()

        logger.info("Transcribing %s...", audio_file.name)

        # Performance notes:
        # fp16=True is generally faster on supported GPUs.
        # language=None triggers auto-detection.
        transcribe_options: dict[str, object] = {
            "fp16": True if self.device == "cuda" else False,
            "language": language,
            "verbose": verbose,
            "condition_on_previous_text": False,
        }
        if clip_timestamps is not None:
            transcribe_options["clip_timestamps"] = clip_timestamps
            if isinstance(clip_timestamps, list):
                logger.info(
                    "Constraining Whisper to %d speech segments.",
                    len(clip_timestamps) // 2,
                )
            else:
                logger.info("Constraining Whisper using provided clip timestamps.")

        with _clip_aware_progress_bar(clip_timestamps, verbose):
            result = self.model.transcribe(str(audio_file), **transcribe_options)

        # Export to SRT using Whisper's official utility
        # This ensures we follow standard formatting correctly.
        writer = get_writer("srt", str(output_dir))

        # writer() expects (result, original_audio_path_or_filename, options)
        # We pass an empty dict for options as we use defaults.
        writer(result, str(audio_file), {})

        logger.info("Transcription complete: %s", srt_path)

        return srt_path


def _load_clip_timestamps_from_file(file_path: str | Path) -> list[float]:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clip timestamps file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            parsed_value = json.load(handle)

        if isinstance(parsed_value, list):
            return validate_clip_timestamps(parsed_value)

        if isinstance(parsed_value, dict):
            return get_clip_timestamps(parsed_value)

        raise ValueError(
            "JSON clip timestamps file must contain either a list of floats or a VAD report object."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw_value = handle.read().strip()

    if not raw_value:
        return []

    parsed_value = json.loads(raw_value)
    if not isinstance(parsed_value, list):
        raise ValueError("Clip timestamps file must contain a JSON list of floats.")

    return validate_clip_timestamps(parsed_value)


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
        "--clip-timestamps-file",
        help="Path to a VAD JSON report or JSON list containing Whisper clip timestamps.",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable Whisper progress output.",
    )
    output_group.add_argument(
        "--verbose-text",
        action="store_true",
        help="Print Whisper decoded text instead of the tqdm progress bar.",
    )

    args = parser.parse_args()

    try:
        verbose: Optional[bool] = False
        if args.quiet:
            verbose = None
        elif args.verbose_text:
            verbose = True

        transcriber = Transcriber(model_name=args.model, device=args.device)
        clip_timestamps = (
            _load_clip_timestamps_from_file(args.clip_timestamps_file)
            if args.clip_timestamps_file
            else None
        )
        transcriber.transcribe(
            args.input,
            output_dir=args.output_dir,
            language=args.language,
            verbose=verbose,
            clip_timestamps=clip_timestamps,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
