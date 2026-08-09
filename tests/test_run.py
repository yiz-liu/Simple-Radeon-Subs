from pathlib import Path

import pytest

from run import _parse_arguments, scan_directory


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


@pytest.mark.parametrize(
    ("extra_arguments", "expected"),
    (([], False), (["--enable-vad"], True)),
)
def test_vad_is_opt_in_from_the_pipeline_cli(
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: list[str],
    expected: bool,
) -> None:
    # Given: A normal pipeline invocation, optionally requesting integrated VAD.
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "movie.mkv", *extra_arguments],
    )

    # When / Then: VAD is disabled by default and enabled only by its flag.
    assert _parse_arguments().enable_vad is expected
