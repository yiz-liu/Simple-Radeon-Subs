import os
import re
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from collections import deque
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Final

from tqdm import tqdm


PROGRESS_PATTERN: Final = re.compile(
    r"whisper_print_progress_callback: progress =\s*(\d{1,3})%"
)
INPUT_START_PATTERN: Final = re.compile(r"read_audio_data: reading audio data from .+")
DIAGNOSTIC_TAIL_LINES: Final = 20


class WhisperProcessError(RuntimeError):
    """Raised when the managed whisper.cpp process contract is broken."""


@dataclass(frozen=True, slots=True)
class TranscriptionTask:
    """One audio input and its published SRT destination."""

    audio_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class TranscriptionFailure:
    """A native transcription failure scoped to one task."""

    task: TranscriptionTask
    reason: str


def parse_progress(line: str) -> int | None:
    """Parse one pinned whisper.cpp progress line."""
    matched = PROGRESS_PATTERN.fullmatch(line.rstrip("\r\n"))
    if matched is None:
        return None
    value = int(matched.group(1))
    return value if 0 <= value <= 100 else None


@dataclass(frozen=True, slots=True)
class WhisperRuntime:
    """Pinned native executable and model paths."""

    cli_path: Path
    model_path: Path
    vad_model_path: Path

    def validate(self) -> None:
        for prerequisite in (self.model_path, self.vad_model_path):
            if not prerequisite.is_file():
                raise FileNotFoundError(
                    f"Managed transcription prerequisite not found: {prerequisite}"
                )
        if not self.cli_path.is_file() or not os.access(self.cli_path, os.X_OK):
            raise FileNotFoundError(
                f"Managed transcription executable not found: {self.cli_path}"
            )


class WhisperBatchRunner:
    """Run one whisper.cpp process for an ordered group of audio files."""

    def __init__(self, runtime: WhisperRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        tasks: tuple[TranscriptionTask, ...],
        language: str | None,
        quiet: bool,
    ) -> list[TranscriptionFailure]:
        if not tasks:
            return []
        self.runtime.validate()
        for task in tasks:
            if not task.audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {task.audio_path}")
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

        stderr_tail: deque[str] = deque(maxlen=DIAGNOSTIC_TAIL_LINES)
        with ExitStack() as stack:
            staged_audio: list[Path] = []
            staged_outputs: list[Path] = []
            for index, task in enumerate(tasks):
                temporary_directory = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            dir=task.output_path.parent,
                            prefix=".transcribe-",
                        )
                    )
                )
                staged = temporary_directory / f"input-{index}.wav"
                staged.symlink_to(task.audio_path.resolve())
                staged_audio.append(staged)
                staged_outputs.append(Path(f"{staged}.srt"))

            process: subprocess.Popen[str] = subprocess.Popen(
                self._command(tuple(staged_audio), language),
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
                raise WhisperProcessError(
                    "whisper-cli did not provide a text stderr stream"
                )

            try:
                self._consume_progress(
                    process,
                    stderr_stream,
                    stderr_tail,
                    len(tasks),
                    quiet,
                )
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
            failures: list[TranscriptionFailure] = []
            for task, generated in zip(tasks, staged_outputs, strict=True):
                if generated.is_file():
                    _ = generated.replace(task.output_path)
                else:
                    failures.append(
                        TranscriptionFailure(
                            task,
                            f"whisper-cli did not create an SRT (exit code {return_code}). "
                            f"stderr tail:\n{diagnostic}",
                        )
                    )
            return failures

    def _command(
        self,
        audio_paths: tuple[Path, ...],
        language: str | None,
    ) -> list[str]:
        return [
            str(self.runtime.cli_path),
            "--model",
            str(self.runtime.model_path),
            "--vad-model",
            str(self.runtime.vad_model_path),
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
            *(str(path) for path in audio_paths),
        ]

    @staticmethod
    def _consume_progress(
        process: subprocess.Popen[str],
        stderr_stream: TextIOWrapper,
        stderr_tail: deque[str],
        task_count: int,
        quiet: bool,
    ) -> None:
        terminal_size = shutil.get_terminal_size()
        with tqdm(
            total=100 * task_count,
            desc="Transcribing",
            unit="%",
            disable=quiet,
            ncols=terminal_size.columns or 80,
            nrows=terminal_size.lines or 24,
        ) as progress:
            file_index = -1
            current_progress = 0
            for line in stderr_stream:
                stderr_tail.append(line.rstrip("\r\n"))
                if INPUT_START_PATTERN.fullmatch(line.rstrip("\r\n")):
                    if file_index >= 0 and current_progress < 100:
                        _ = progress.update(100 - current_progress)
                    file_index += 1
                    current_progress = 0
                    continue
                parsed = parse_progress(line)
                if parsed is not None and parsed > current_progress:
                    _ = progress.update(parsed - current_progress)
                    current_progress = parsed
            if process.poll() is None and file_index >= 0 and current_progress < 100:
                _ = progress.update(100 - current_progress)
