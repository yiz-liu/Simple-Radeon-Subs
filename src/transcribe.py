import argparse
from pathlib import Path

from src.config import (
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_VAD_MODEL_PATH,
)
from src.logger import logger
from src.whisper_batch import (
    TranscriptionFailure,
    TranscriptionTask,
    WhisperBatchRunner,
    WhisperRuntime,
    parse_progress,
)

__all__ = ["Transcriber", "TranscriptionFailure", "TranscriptionTask", "parse_progress"]


class Transcriber:
    """Transcribe audio through the repository-managed whisper.cpp CLI."""

    def transcribe_many(
        self,
        tasks: tuple[TranscriptionTask, ...],
        language: str | None = None,
        quiet: bool = False,
        enable_vad: bool = True,
    ) -> list[TranscriptionFailure]:
        """Create multiple SRT files with one native model load."""
        if tasks:
            logger.info("Transcribing %d audio file(s)...", len(tasks))
        runner = WhisperBatchRunner(
            WhisperRuntime(
                cli_path=WHISPER_CLI_PATH,
                model_path=WHISPER_MODEL_PATH,
                vad_model_path=WHISPER_VAD_MODEL_PATH,
            )
        )
        failures = runner.run(tasks, language, quiet, enable_vad)
        logger.info(
            "Transcription batch complete: %d succeeded, %d failed.",
            len(tasks) - len(failures),
            len(failures),
        )
        return failures

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None,
        language: str | None = None,
        quiet: bool = False,
        enable_vad: bool = True,
    ) -> Path:
        """Create an SRT beside the audio or in the requested output directory."""
        audio_file = Path(audio_path).resolve()
        output_directory = Path(output_dir or audio_file.parent).resolve()
        destination = output_directory / f"{audio_file.stem}.srt"
        task = TranscriptionTask(audio_file, destination)
        failures = self.transcribe_many(
            (task,),
            language=language,
            quiet=quiet,
            enable_vad=enable_vad,
        )
        if failures:
            raise RuntimeError(failures[0].reason)
        return destination


class _Arguments(argparse.Namespace):
    input: str = ""
    output_dir: str | None = None
    language: str | None = None
    quiet: bool = False
    enable_vad: bool = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using the managed whisper.cpp CLI."
    )
    _ = parser.add_argument("input", help="Path to the input audio file.")
    _ = parser.add_argument(
        "-o", "--output-dir", help="Directory to save the SRT file."
    )
    _ = parser.add_argument(
        "-l",
        "--language",
        help="Source language code. Auto-detects if omitted.",
    )
    _ = parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable transcription progress output.",
    )
    _ = parser.add_argument(
        "--disable-vad",
        action="store_false",
        dest="enable_vad",
        help="Disable integrated VAD and transcribe the complete audio stream.",
    )
    arguments = parser.parse_args(namespace=_Arguments())
    try:
        _ = Transcriber().transcribe(
            arguments.input,
            output_dir=arguments.output_dir,
            language=arguments.language,
            quiet=arguments.quiet,
            enable_vad=arguments.enable_vad,
        )
    except (OSError, RuntimeError) as error:
        logger.error("Transcription failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
