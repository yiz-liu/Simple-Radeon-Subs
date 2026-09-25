import pytest

from src.aligner import (
    AlignmentRecord,
    merge_alignments,
    needs_alignment,
    render_timed_record,
    sentence_groups,
)


def record(
    times: tuple[float, ...], text: str = "First;Middle;Last;"
) -> AlignmentRecord:
    units = tuple(text.rstrip(";").split(";"))
    return AlignmentRecord(
        segment_id="w1",
        start=160000,
        end=320000,
        text=text,
        units=units,
        fixed=times,
    )


def test_short_multisentence_window_needs_no_alignment() -> None:
    source = AlignmentRecord(
        segment_id="short", start=16000, end=64000, text="Yes;Next;"
    )
    assert not needs_alignment(source)
    cues = render_timed_record(source)
    assert len(cues) == 1
    assert (cues[0].start, cues[0].end, cues[0].text) == (1, 4, source.text)
    assert cues[0].published


def test_long_window_routes_to_aligner_but_empty_and_failed_do_not() -> None:
    source = record((1, 2, 3, 4, 5, 6))
    assert needs_alignment(source)
    assert not needs_alignment(source.model_copy(update={"text": ""}))
    failed = source.model_copy(update={"error": "asr_truncated"})
    assert not needs_alignment(failed)
    assert not render_timed_record(failed)[0].published


def test_valid_times_are_unchanged_and_small_overshoot_is_clipped() -> None:
    cues = render_timed_record(record((1, 2, 3, 4, 8, 10.028)))
    assert [(c.start, c.end) for c in cues] == [(11, 12), (13, 14), (18, 20)]
    assert [c.text for c in cues] == ["First;", "Middle;", "Last;"]
    assert all(c.published for c in cues)


def test_consecutive_invalid_sentences_share_one_bounded_cue() -> None:
    source = record((1, 2, 3, 3, 3, 3, 6, 7), "First;Middle;Next;Last;")
    cues = render_timed_record(source)
    assert [(c.start, c.end) for c in cues] == [(11, 12), (12, 16), (16, 17)]
    assert cues[1].text.replace("\n", "") == "Middle;Next;"
    assert all(c.published for c in cues)


@pytest.mark.parametrize("gap", [0, 0.16, 0.24])
def test_tiny_or_zero_gap_merges_text_without_overlap(gap: float) -> None:
    cues = render_timed_record(record((1, 2, 2, 2, 2 + gap, 4)))
    assert len(cues) == 2
    assert cues[0].text.replace("\n", "") == "First;Middle;"
    assert (cues[0].start, cues[0].end) == (11, 12 + gap)
    assert cues[0].end == cues[1].start
    assert all(c.published for c in cues)


def test_first_invalid_sentence_merges_forward_when_no_space() -> None:
    cues = render_timed_record(record((0, 0, 0.16, 2, 4, 5)))
    assert cues[0].text.replace("\n", "") == "First;Middle;"
    assert (cues[0].start, cues[0].end) == (10, 12)
    assert all(c.published for c in cues)


def test_entire_invalid_window_falls_back_without_losing_text() -> None:
    source = record((0, 0, 0, 0, 0, 0))
    cues = render_timed_record(source)
    assert len(cues) == 1
    assert (cues[0].start, cues[0].end) == (10, 20)
    assert cues[0].text.replace("\n", "") == source.text
    assert cues[0].published


def test_last_invalid_sentence_uses_window_edge() -> None:
    cues = render_timed_record(record((1, 2, 3, 4, 8, 8)))
    assert (cues[-1].start, cues[-1].end) == (14, 20)
    assert cues[-1].text == "Last;" and cues[-1].published


def test_merge_keeps_source_order_and_rejects_missing_worker_results() -> None:
    short = AlignmentRecord(segment_id="short", start=0, end=16000, text="Yes;")
    long = record((1, 2, 3, 4, 5, 6))
    merged = merge_alignments([short, long], [long])
    assert [r.segment_id for r in merged] == ["short", "w1"]
    assert merged == [short, long]
    with pytest.raises(ValueError, match="mismatch"):
        merge_alignments([short, long], [])
    with pytest.raises(ValueError, match="mismatch"):
        merge_alignments([short, long], [long, long])


def test_merge_rejects_changed_text_or_audio_bounds() -> None:
    source = record((1, 2, 3, 4, 5, 6))
    with pytest.raises(ValueError, match="changed"):
        merge_alignments([source], [source.model_copy(update={"text": "different"})])


def test_periods_split_sentences_but_decimal_points_do_not() -> None:
    groups = sentence_groups(
        "It costs 3.14 euros. Thank you; Goodbye!",
        ("It", "costs", "314", "euros", "Thank", "you", "Goodbye"),
    )
    assert [x[2] for x in groups] == [
        "It costs 3.14 euros. ",
        "Thank you; ",
        "Goodbye!",
    ]
