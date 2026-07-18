import argparse
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from io import TextIOWrapper
from pathlib import Path
from typing import Final

from tqdm import tqdm

from src.config import (
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_VAD_MODEL_PATH,
)
from src.logger import logger

PROGRESS_PATTERN: Final = re.compile(
    r"whisper_print_progress_callback: progress =\s*(\d{1,3})%"
)
DIAGNOSTIC_TAIL_LINES: Final = 20


def parse_progress(line: str) -> int | None:
    """Parse one pinned whisper.cpp progress line."""
    matched = PROGRESS_PATTERN.fullmatch(line.rstrip("\r\n"))
    if matched is None:
        return None

    value = int(matched.group(1))
    return value if 0 <= value <= 100 else None


class Transcriber:
    """Transcribe audio through the repository-managed whisper.cpp CLI."""

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None,
        language: str | None = None,
        quiet: bool = False,
    ) -> Path:
        """Create an SRT beside the audio or in the requested output directory."""
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        for prerequisite in (WHISPER_MODEL_PATH, WHISPER_VAD_MODEL_PATH):
            if not prerequisite.is_file():
                raise FileNotFoundError(
                    f"Managed transcription prerequisite not found: {prerequisite}"
                )
        if not WHISPER_CLI_PATH.is_file() or not os.access(WHISPER_CLI_PATH, os.X_OK):
            raise FileNotFoundError(
                f"Managed transcription executable not found: {WHISPER_CLI_PATH}"
            )

        output_directory = Path(output_dir or audio_file.parent).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        destination = output_directory / f"{audio_file.stem}.srt"
        stderr_tail: deque[str] = deque(maxlen=DIAGNOSTIC_TAIL_LINES)

        logger.info("Transcribing %s...", audio_file.name)
        with tempfile.TemporaryDirectory(
            dir=output_directory,
            prefix=".transcribe-",
        ) as temporary_directory:
            output_prefix = Path(temporary_directory) / audio_file.stem
            generated_srt = Path(f"{output_prefix}.srt")
            command = [
                str(WHISPER_CLI_PATH),
                "--model",
                str(WHISPER_MODEL_PATH),
                "--vad-model",
                str(WHISPER_VAD_MODEL_PATH),
                "--file",
                str(audio_file),
                "--output-file",
                str(output_prefix),
                "--output-srt",
                "--language",
                language or "auto",
                "--processors",
                "1",
                "--beam-size",
                "5",
                "--max-context",
                "48",
                "--vad",
                "--vad-threshold",
                "0.15",
                "--vad-min-speech-duration-ms",
                "250",
                "--vad-min-silence-duration-ms",
                "120",
                "--vad-max-speech-duration-s",
                "30",
                "--vad-speech-pad-ms",
                "250",
                "--vad-samples-overlap",
                "0",
                "--no-prints",
                "--print-progress",
            ]
            process: subprocess.Popen[str] = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stderr_stream = process.stderr
            if not isinstance(stderr_stream, TextIOWrapper):
                process.terminate()
                _ = process.wait()
                error_message = "whisper-cli did not provide a text stderr stream"
                raise RuntimeError(error_message)

            try:
                current_progress = 0
                terminal_size = shutil.get_terminal_size()
                with tqdm(
                    total=100,
                    desc="Transcribing",
                    unit="%",
                    disable=quiet,
                    ncols=terminal_size.columns or 80,
                    nrows=terminal_size.lines or 24,
                ) as progress:
                    for line in stderr_stream:
                        stderr_tail.append(line.rstrip("\r\n"))
                        parsed_progress = parse_progress(line)
                        if (
                            parsed_progress is not None
                            and parsed_progress > current_progress
                        ):
                            _ = progress.update(parsed_progress - current_progress)
                            current_progress = parsed_progress
                return_code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        _ = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        _ = process.wait()
                stderr_stream.close()

            diagnostic = "\n".join(stderr_tail)
            if return_code != 0:
                error_message = (
                    f"whisper-cli failed with exit code {return_code}. stderr tail:\n"
                    f"{diagnostic}"
                )
                raise RuntimeError(error_message)
            if not generated_srt.is_file():
                error_message = (
                    "whisper-cli exited successfully without creating an SRT. "
                    f"stderr tail:\n{diagnostic}"
                )
                raise RuntimeError(error_message)

            _ = generated_srt.replace(destination)

        logger.info("Transcription complete: %s", destination)
        return destination


class _Arguments(argparse.Namespace):
    input: str = ""
    output_dir: str | None = None
    language: str | None = None
    quiet: bool = False


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
    arguments = parser.parse_args(namespace=_Arguments())

    try:
        _ = Transcriber().transcribe(
            arguments.input,
            output_dir=arguments.output_dir,
            language=arguments.language,
            quiet=arguments.quiet,
        )
    except (OSError, RuntimeError) as error:
        logger.error("Transcription failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
