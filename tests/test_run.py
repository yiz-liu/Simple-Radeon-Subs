from pathlib import Path

from run import scan_directory


def test_directory_scan_excludes_managed_sidecars(tmp_path: Path) -> None:
    # Given: Source media, ordinary nested media, and a retained pipeline sidecar.
    source = tmp_path / "movie.mp4"
    nested = tmp_path / "nested" / "soundtrack.mp3"
    cached_audio = tmp_path / ".movie.mp4.simple-radeon-subs" / "audio.wav"
    nested.parent.mkdir()
    cached_audio.parent.mkdir()
    source.touch()
    nested.touch()
    cached_audio.touch()

    # When: The input directory is scanned again.
    discovered = scan_directory(tmp_path)

    # Then: Only user media is returned; pipeline-owned audio stays internal.
    assert discovered == [source, nested]
