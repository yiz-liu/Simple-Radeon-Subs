from src.ffmpeg_progress import parse_out_time_seconds


def test_parse_out_time_seconds_converts_microseconds() -> None:
    # Given: FFmpeg's machine-readable progress timestamp.
    line = "out_time_us=1500000"

    # When: The timestamp is parsed.
    result = parse_out_time_seconds(line)

    # Then: It is returned in seconds for tqdm.
    assert result == 1.5


def test_parse_out_time_seconds_ignores_unrelated_progress_fields() -> None:
    # Given: A valid FFmpeg progress control field.
    line = "progress=continue"

    # When: The field is parsed as a timestamp.
    result = parse_out_time_seconds(line)

    # Then: It is ignored.
    assert result is None


def test_parse_out_time_seconds_ignores_invalid_values() -> None:
    # Given: Invalid timestamp values that must not move progress backwards.
    lines = ("out_time_us=unknown", "out_time_us=-1")

    # When: Each value is parsed.
    results = tuple(parse_out_time_seconds(line) for line in lines)

    # Then: Neither value is accepted.
    assert results == (None, None)
