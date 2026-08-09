import wave
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest

from src.audio import AudioExtractor, parse_out_time_seconds
from src.config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE

EXAMPLE_VIDEO: Final = Path(__file__).with_name("example.mp4")


def test_audio_extractor_rejects_a_missing_system_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A required executable is absent from PATH.
    missing_binaries: dict[str, str] = {}
    monkeypatch.setattr("src.audio.shutil.which", missing_binaries.get)

    # When / Then: Resolution fails at the system dependency boundary.
    with pytest.raises(FileNotFoundError):
        _ = AudioExtractor()


def test_parse_out_time_seconds_converts_microseconds() -> None:
    # Given: FFmpeg's machine-readable progress timestamp.
    line = "out_time_us=1500000"

    # When: The timestamp is parsed.
    result = parse_out_time_seconds(line)

    # Then: It is returned in seconds for the progress bar.
    assert result == 1.5


@pytest.mark.parametrize(
    "line",
    ("progress=continue", "out_time_us=unknown", "out_time_us=-1"),
)
def test_parse_out_time_seconds_ignores_non_timestamp_values(line: str) -> None:
    # Given: A progress field that cannot advance the progress bar.

    # When: The field is parsed as a timestamp.
    result = parse_out_time_seconds(line)

    # Then: It is ignored.
    assert result is None


def test_extract_converts_example_video_to_whisper_wav(tmp_path: Path) -> None:
    # Given: The shared real-media fixture and a fresh output path.
    output_path = tmp_path / "example.wav"
    extractor = AudioExtractor()

    # When: The audio extraction CLI adapter processes the fixture.
    extracted_path = extractor.extract(EXAMPLE_VIDEO, output_path)

    # Then: The result is a non-empty 16 kHz mono 16-bit PCM WAV.
    with wave.open(str(extracted_path), "rb") as audio:
        assert (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
        ) == (AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, 2)
        assert audio.getnframes() > 0


def test_extract_uses_a_reusable_nested_progress_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A batch caller reserves the second terminal line for the current file.
    progress = MagicMock()
    progress.__enter__.return_value = progress
    progress_factory = MagicMock(return_value=progress)
    monkeypatch.setattr("src.audio.tqdm", progress_factory)
    output_path = tmp_path / "example.wav"

    # When: The extractor processes a file within that batch display.
    _ = AudioExtractor().extract(
        EXAMPLE_VIDEO,
        output_path,
        progress_position=1,
    )

    # Then: The current-file bar occupies and clears the reusable second line.
    progress_factory.assert_called_once()
    progress_options = progress_factory.call_args.kwargs
    assert progress_options["desc"] == f"Current file: {EXAMPLE_VIDEO.name}"
    assert progress_options["unit"] == "s"
    assert progress_options["position"] == 1
    assert progress_options["leave"] is False
