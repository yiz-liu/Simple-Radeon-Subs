import pytest

from src.audio import AudioExtractor


def test_audio_extractor_resolves_both_system_binaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Both FFmpeg executables are available on PATH.
    binaries = {
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffprobe": "/usr/bin/ffprobe",
    }
    monkeypatch.setattr("src.audio.shutil.which", binaries.get)

    # When: The extractor is initialized.
    extractor = AudioExtractor()

    # Then: It uses the two PATH-resolved executables.
    assert extractor.ffmpeg_path == binaries["ffmpeg"]
    assert extractor.ffprobe_path == binaries["ffprobe"]


def test_audio_extractor_rejects_a_missing_system_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A required executable is absent from PATH.
    monkeypatch.setattr("src.audio.shutil.which", lambda _: None)

    # When / Then: Resolution fails with an actionable error.
    with pytest.raises(FileNotFoundError, match="ffprobe.*PATH"):
        AudioExtractor._resolve_binary("ffprobe")
