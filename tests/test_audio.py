from pathlib import Path
import subprocess
import sys
from typing import Final
from unittest.mock import MagicMock
import wave

import numpy as np
import pytest

from src.audio import SampleSpan, processing_windows

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


@pytest.mark.parametrize("gap_seconds", (0.0, 0.025, 0.5))
def test_extract_preserves_source_timestamp_gaps(
    tmp_path: Path, gap_seconds: float
) -> None:
    extractor = AudioExtractor()
    source = tmp_path / "timestamp-gap.nut"
    subprocess.run(
        [
            extractor.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-af",
            "asetnsamples=n=160:p=0,"
            f"asetpts=PTS+if(gte(T\\,0.5)\\,{gap_seconds}/TB\\,0)",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    output = extractor.extract(source, tmp_path / "extracted.wav")

    with wave.open(str(output), "rb") as audio:
        assert audio.getnframes() == round((1 + gap_seconds) * AUDIO_SAMPLE_RATE)
        audio.setpos(AUDIO_SAMPLE_RATE // 2)
        gap_samples = round(gap_seconds * AUDIO_SAMPLE_RATE)
        assert audio.readframes(gap_samples) == bytes(gap_samples * 2)
        audio.setpos(round((0.75 + gap_seconds) * AUDIO_SAMPLE_RATE))
        assert any(audio.readframes(AUDIO_SAMPLE_RATE // 8))


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
    assert progress_options["disable"] is (not sys.stderr.isatty())
    assert progress_options["position"] == 1
    assert progress_options["leave"] is False


def test_windows_keep_all_speech_and_intervening_short_silence() -> None:
    audio = np.ones(66 * 16000, dtype=np.float32)
    audio[32 * 16000 : 33 * 16000] = 0
    spans = [SampleSpan(16000, 32000), SampleSpan(40000, 65 * 16000)]
    windows = processing_windows(audio, spans)
    assert windows[0].start == 16000
    assert windows[-1].end == 65 * 16000
    assert all(a.end == b.start for a, b in zip(windows, windows[1:]))
    assert sum(w.end - w.start for w in windows) == 64 * 16000
    assert max(w.end - w.start for w in windows) <= 34 * 16000


def test_long_silence_stays_outside_processing_windows() -> None:
    audio = np.ones(20 * 16000, dtype=np.float32)
    spans = [SampleSpan(16000, 5 * 16000), SampleSpan(15 * 16000, 19 * 16000)]
    assert processing_windows(audio, spans) == spans


def test_tiny_adjacent_segment_gets_continuous_context() -> None:
    audio = np.ones(10 * 16000, dtype=np.float32)
    spans = [SampleSpan(1600, 8000), SampleSpan(10000, 9 * 16000)]
    assert processing_windows(audio, spans) == [SampleSpan(1600, 9 * 16000)]


def test_invalid_or_unordered_spans_are_rejected() -> None:
    audio = np.ones(16000, dtype=np.float32)
    with pytest.raises(ValueError):
        processing_windows(audio, [SampleSpan(0, 20000)])
    with pytest.raises(ValueError):
        processing_windows(audio, [SampleSpan(500, 1000), SampleSpan(100, 400)])


def test_empty_vad_produces_no_windows() -> None:
    assert processing_windows(np.zeros(16000, dtype=np.float32), []) == []
