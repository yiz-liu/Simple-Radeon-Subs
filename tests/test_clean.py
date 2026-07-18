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


def test_clean_srt_preserves_meaningful_short_long_and_marked_up_cues(
    tmp_path: Path,
) -> None:
    # Given: Meaningful cues that the previous language and duration heuristics removed.
    items = [
        _subtitle(1, (0, 10_000), " I "),
        _subtitle(2, (10_100, 50_000), "<i>Stay</i> [aside] ♪"),
    ]

    # When: The conservative cleaner processes them.
    cleaned = _clean(tmp_path, items)

    # Then: Only surrounding whitespace changes.
    assert [(item.text, item.start.ordinal, item.end.ordinal) for item in cleaned] == [
        ("I", 0, 10_000),
        ("<i>Stay</i> [aside] ♪", 10_100, 50_000),
    ]


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


def test_clean_srt_rejects_a_non_positive_duration(tmp_path: Path) -> None:
    # Given: A structurally invalid zero-duration cue.
    items = [_subtitle(1, (1_000, 1_000), "Broken timeline")]

    # When / Then: The cleaner rejects it instead of deleting it silently.
    with pytest.raises(ValueError):
        _ = _clean(tmp_path, items)
