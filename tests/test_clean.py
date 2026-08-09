from pathlib import Path

import pysrt
import pytest

from src.clean import clean_srt


def _subtitle(
    index: int,
    span_ms: tuple[int, int],
    text: str,
) -> pysrt.SubRipItem:
    start_ms, end_ms = span_ms
    return pysrt.SubRipItem(
        index=index,
        start=pysrt.SubRipTime(milliseconds=start_ms),
        end=pysrt.SubRipTime(milliseconds=end_ms),
        text=text,
    )


def _clean(
    tmp_path: Path,
    items: list[pysrt.SubRipItem],
) -> pysrt.SubRipFile:
    input_path = tmp_path / "input.srt"
    output_path = tmp_path / "output.srt"
    pysrt.SubRipFile(items=items).save(str(input_path), encoding="utf-8")
    clean_srt(input_path, output_path)
    return pysrt.open(str(output_path), encoding="utf-8")


def test_clean_srt_preserves_meaningful_text_and_markup(
    tmp_path: Path,
) -> None:
    # Given: Meaningful cues that the previous language and duration heuristics removed.
    items = [
        _subtitle(1, (0, 10_000), " I "),
        _subtitle(2, (10_100, 50_000), "<i>Stay</i> [aside] ♪"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: Content is preserved while long cue timestamps keep only their endings.
    assert [(item.text, item.start.ordinal, item.end.ordinal) for item in cleaned] == [
        ("I", 8_000, 10_000),
        ("<i>Stay</i> [aside] ♪", 48_000, 50_000),
    ]


@pytest.mark.parametrize(
    ("span_ms", "expected_span_ms"),
    (
        ((1_000, 6_000), (1_000, 6_000)),
        ((1_000, 6_001), (4_001, 6_001)),
    ),
)
def test_clean_srt_keeps_only_the_final_two_seconds_when_duration_exceeds_five(
    tmp_path: Path,
    span_ms: tuple[int, int],
    expected_span_ms: tuple[int, int],
) -> None:
    # Given: A cue at or immediately beyond the five-second boundary.
    items = [_subtitle(1, span_ms, "Keep the ending")]

    # When: The cleaner normalizes its timeline.
    cleaned = _clean(tmp_path, items)

    # Then: Only a duration over five seconds moves the start to two seconds
    # before the unchanged end.
    assert (cleaned[0].start.ordinal, cleaned[0].end.ordinal) == expected_span_ms


def test_clean_srt_removes_only_empty_punctuation_and_excessive_repetition(
    tmp_path: Path,
) -> None:
    # Given: Safely removable cues plus a short repeated vocalization.
    items = [
        _subtitle(1, (0, 1_000), "   "),
        _subtitle(2, (1_000, 2_000), "... ?!"),
        _subtitle(3, (2_000, 3_000), "oooo"),
        _subtitle(4, (3_000, 4_000), "ooo"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: Four identical effective characters are excessive; three are retained.
    assert [(item.index, item.text) for item in cleaned] == [(1, "ooo")]


def test_clean_srt_recovers_invalid_utf8(tmp_path: Path) -> None:
    # Given: A native SRT containing one incomplete UTF-8 sequence.
    input_path = tmp_path / "input.srt"
    output_path = tmp_path / "output.srt"
    input_path.write_bytes(
        b"1\n00:00:00,000 --> 00:00:01,000\n"
        + b"before"
        + b"\xe3\x81"
        + b"after\n"
    )

    # When: The cleaner reads the malformed native output.
    clean_srt(input_path, output_path)

    # Then: Surrounding text is preserved in a valid UTF-8 SRT.
    cleaned = pysrt.open(str(output_path), encoding="utf-8")
    assert [item.text for item in cleaned] == ["before�after"]


def test_clean_srt_removes_only_qualifying_phrase_loops(tmp_path: Path) -> None:
    # Given: Phrase loops around the text-length, phrase-length, and repeat boundaries.
    items = [
        _subtitle(1, (0, 1_000), "go!" * 16),
        _subtitle(2, (1_000, 2_000), "abcdefgh" * 4),
        _subtitle(3, (2_000, 3_000), "abcdefghi" * 4),
        _subtitle(4, (3_000, 4_000), "abcdefgh" * 3 + "ijklmnop"),
        _subtitle(5, (4_000, 5_000), "go!" * 15),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: Only the qualifying 32-character loop is removed; the shorter loop
    # crosses the normalization boundary and retains three repetitions.
    assert [item.text for item in cleaned] == [
        "abcdefghi" * 4,
        "abcdefgh" * 3 + "ijklmnop",
        "go!" * 3,
    ]


def test_clean_srt_normalizes_only_pathological_repetition(tmp_path: Path) -> None:
    # Given: Character and phrase runs on both sides of the repetition boundary.
    items = [
        _subtitle(1, (0, 1_000), "Wow" + "!" * 4),
        _subtitle(2, (2_000, 3_000), "very " * 4 + "nice."),
        _subtitle(3, (4_000, 5_000), "Wow" + "!" * 3),
        _subtitle(4, (6_000, 7_000), "very " * 3 + "nice."),
    ]

    # When: The conservative cleaner processes the native output.
    cleaned = _clean(tmp_path, items)

    # Then: More than three repeats collapse to three; shorter emphasis is unchanged.
    assert [item.text for item in cleaned] == [
        "Wow" + "!" * 3,
        "very " * 3 + "nice.",
        "Wow" + "!" * 3,
        "very " * 3 + "nice.",
    ]


@pytest.mark.parametrize(
    ("gap_ms", "expected_count"),
    ((250, 1), (251, 2)),
)
def test_clean_srt_merges_duplicate_pair_only_within_the_gap_limit(
    tmp_path: Path,
    gap_ms: int,
    expected_count: int,
) -> None:
    # Given: Two identical cues separated by a boundary-sized gap.
    items = [
        _subtitle(1, (0, 1_000), "Same line"),
        _subtitle(2, (1_000 + gap_ms, 2_000 + gap_ms), "Same line"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: At most 250 ms is one run; anything larger remains separate.
    assert len(cleaned) == expected_count
    if expected_count == 1:
        assert (cleaned[0].start.ordinal, cleaned[0].end.ordinal) == (
            0,
            2_000 + gap_ms,
        )


def test_clean_srt_discards_three_duplicates_in_one_close_run(tmp_path: Path) -> None:
    # Given: Three identical cues separated only by small gaps.
    items = [
        _subtitle(1, (0, 1_000), "Loop"),
        _subtitle(2, (1_100, 2_000), "Loop"),
        _subtitle(3, (2_200, 3_000), "Loop"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: The repeated run is removed entirely.
    assert cleaned == []


def test_clean_srt_filters_zero_duration_cue_inside_duplicate_run(
    tmp_path: Path,
) -> None:
    # Given: whisper.cpp emitted a zero-duration cue inside one repeated run.
    items = [
        _subtitle(1, (0, 1_000), "Repeated line"),
        _subtitle(2, (1_000, 1_000), "Repeated line"),
        _subtitle(3, (1_000, 2_000), "Repeated line"),
        _subtitle(4, (2_000, 3_000), "Next line"),
    ]

    # When: The conservative cleaner processes the native output.
    cleaned = _clean(tmp_path, items)

    # Then: The repeated hallucination is removed before strict timeline validation.
    assert [(item.text, item.start.ordinal, item.end.ordinal) for item in cleaned] == [
        ("Next line", 2_000, 3_000)
    ]


def test_clean_srt_does_not_discard_duplicates_across_a_large_gap(
    tmp_path: Path,
) -> None:
    # Given: A close duplicate pair followed by the same line after a larger gap.
    items = [
        _subtitle(1, (0, 1_000), "Again"),
        _subtitle(2, (1_100, 2_000), "Again"),
        _subtitle(3, (2_300, 3_000), "Again"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: The pair merges, while the separate occurrence remains.
    assert [item.text for item in cleaned] == ["Again", "Again"]
    assert [(item.start.ordinal, item.end.ordinal) for item in cleaned] == [
        (0, 2_000),
        (2_300, 3_000),
    ]


def test_clean_srt_discards_a_zero_duration_cue(tmp_path: Path) -> None:
    # Given: A zero-duration native cue that no subtitle renderer can display.
    items = [_subtitle(1, (1_000, 1_000), "Invisible text")]

    # When: The conservative cleaner processes it.
    cleaned = _clean(tmp_path, items)

    # Then: The unrenderable cue is discarded.
    assert cleaned == []


def test_clean_srt_rejects_a_negative_duration(tmp_path: Path) -> None:
    # Given: A negative-duration cue is surrounded by matching duplicates.
    items = [
        _subtitle(1, (0, 1_000), "Broken timeline"),
        _subtitle(2, (1_000, 900), "Broken timeline"),
        _subtitle(3, (900, 2_000), "Broken timeline"),
    ]

    # When / Then: The cleaner rejects it instead of deleting it silently.
    with pytest.raises(ValueError):
        _ = _clean(tmp_path, items)
