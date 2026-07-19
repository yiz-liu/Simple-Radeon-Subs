import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from src.audio import AudioExtractor
from src.config import (
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_VAD_MODEL_PATH,
)
from src.transcribe import Transcriber, TranscriptionTask

EXAMPLE_VIDEO: Final = Path(__file__).with_name("example.mp4")
EXAMPLE_AUDIO: Final = Path(__file__).with_name("example_en.mp3")
REAL_ASR_AVAILABLE: Final = (
    WHISPER_CLI_PATH.is_file()
    and os.access(WHISPER_CLI_PATH, os.X_OK)
    and WHISPER_MODEL_PATH.is_file()
    and WHISPER_VAD_MODEL_PATH.is_file()
)
SRT_VALIDATION_SCRIPT: Final = """
import sys
from itertools import pairwise

import pysrt

subtitles = pysrt.open(sys.argv[1], encoding="utf-8")
assert subtitles
assert any(subtitle.text.strip() for subtitle in subtitles)
assert [subtitle.index for subtitle in subtitles] == list(range(1, len(subtitles) + 1))
assert all(subtitle.start.ordinal < subtitle.end.ordinal for subtitle in subtitles)
assert all(
    previous.end.ordinal <= current.start.ordinal
    for previous, current in pairwise(subtitles)
)
print(f"validated {len(subtitles)} subtitles")
"""


def test_module_cli_exposes_only_the_fixed_adapter_options() -> None:
    # Given: The transcribe module's public command-line surface.

    # When: A user asks for help.
    completed = subprocess.run(
        [sys.executable, "-m", "src.transcribe", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: Only input, output, language, and quiet controls remain.
    assert completed.returncode == 0
    assert all(
        flag in completed.stdout for flag in ("--output-dir", "--language", "--quiet")
    )
    assert all(
        flag not in completed.stdout
        for flag in ("--model", "--device", "--verbose-text")
    )


def test_pipeline_cli_does_not_expose_asr_model_selection() -> None:
    # Given: The end-to-end pipeline's public command-line surface.

    # When: A user asks for help.
    completed = subprocess.run(
        [sys.executable, "run.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: The fixed ASR model is not a user-selectable option.
    assert completed.returncode == 0
    assert "--model" not in completed.stdout


def test_module_cli_rejects_a_missing_audio_file_without_creating_an_srt(
    tmp_path: Path,
) -> None:
    # Given: A missing WAV path and a fresh subtitle directory.
    missing_audio = tmp_path / "missing.wav"
    output_dir = tmp_path / "subtitles"

    # When: The public module CLI receives the malformed input path.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(missing_audio),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: The CLI reports failure without publishing a subtitle.
    assert completed.returncode != 0
    assert not (output_dir / "missing.srt").exists()


def test_module_cli_reports_an_invalid_output_directory_without_a_traceback(
    tmp_path: Path,
) -> None:
    # Given: A valid audio path and a regular file where a directory is required.
    audio_path = tmp_path / "audio.wav"
    output_file = tmp_path / "not-a-directory"
    _ = audio_path.touch()
    _ = output_file.touch()

    # When: The public CLI reaches the filesystem output boundary.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(audio_path),
            "--output-dir",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: It reports a controlled failure without publishing an SRT.
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "audio.srt").exists()


@pytest.mark.skipif(
    not REAL_ASR_AVAILABLE,
    reason="managed whisper.cpp CLI, full model, or VAD model is unavailable",
)
def test_module_cli_transcribes_the_real_media_fixture_on_rocm(tmp_path: Path) -> None:
    # Given: The shared real-media fixture converted through the production extractor.
    audio_path = AudioExtractor().extract(EXAMPLE_VIDEO, tmp_path / "example.wav")
    output_dir = tmp_path / "subtitles"

    # When: The public module CLI transcribes through managed whisper.cpp.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(audio_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Then: Visible completion accompanies a well-formed, ordered subtitle timeline.
    subtitle_path = output_dir / "example.srt"
    assert completed.returncode == 0, completed.stderr
    assert "Transcribing" in completed.stderr
    assert "100%" in completed.stderr
    validation = subprocess.run(
        [sys.executable, "-c", SRT_VALIDATION_SCRIPT, str(subtitle_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert validation.returncode == 0, validation.stderr
    assert validation.stdout.startswith("validated ")


@pytest.mark.skipif(
    not REAL_ASR_AVAILABLE or not EXAMPLE_AUDIO.is_file(),
    reason="managed whisper.cpp runtime or the second media fixture is unavailable",
)
def test_transcribe_many_processes_two_real_audio_files_in_one_batch(
    tmp_path: Path,
) -> None:
    # Given: The video fixture converted to WAV and the independent MP3 fixture.
    first_audio = AudioExtractor().extract(EXAMPLE_VIDEO, tmp_path / "first.wav")
    tasks = (
        TranscriptionTask(first_audio, tmp_path / "first.srt"),
        TranscriptionTask(EXAMPLE_AUDIO, tmp_path / "second.srt"),
    )

    # When: Both files enter one managed whisper.cpp invocation.
    failures = Transcriber().transcribe_many(tasks, language="en", quiet=True)

    # Then: Both outputs contain valid ordered subtitle timelines.
    assert failures == []
    for task in tasks:
        validation = subprocess.run(
            [sys.executable, "-c", SRT_VALIDATION_SCRIPT, str(task.output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert validation.returncode == 0, validation.stderr
