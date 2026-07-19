import json

import pytest

from src.translation_requests import (
    TranslationOutputError,
    build_translation_requests,
    parse_translation_output,
)


def test_requests_keep_core_cues_away_from_window_boundaries() -> None:
    # Given: More subtitle cues than fit in two core chunks.
    texts = tuple(f"cue {index}" for index in range(40))

    # When: Translation requests are assembled.
    requests = build_translation_requests(texts)

    # Then: Cores do not overlap, while their request windows do.
    assert [request.core_ids for request in requests] == [
        tuple(str(index) for index in range(1, 17)),
        tuple(str(index) for index in range(17, 33)),
        tuple(str(index) for index in range(33, 41)),
    ]
    assert [request.window_ids for request in requests] == [
        tuple(str(index) for index in range(1, 21)),
        tuple(str(index) for index in range(13, 37)),
        tuple(str(index) for index in range(29, 41)),
    ]


def test_request_size_depends_on_cues_instead_of_token_count() -> None:
    # Given: One unusually long subtitle cue.
    texts = ("long cue " * 2_000,)

    # When: Translation requests are assembled.
    requests = build_translation_requests(texts)

    # Then: The cue remains intact in one request.
    assert len(requests) == 1
    assert requests[0].texts == texts


def test_translation_output_keeps_only_the_request_core() -> None:
    # Given: A valid response containing translated overlap and core cues.
    request = build_translation_requests(tuple(f"cue {index}" for index in range(20)))[1]
    response = json.dumps(
        {cue_id: f"translation {cue_id}" for cue_id in request.window_ids}
    )

    # When: The structured response is parsed.
    result = parse_translation_output(response, request)

    # Then: Only the non-overlapping core is retained for SRT assembly.
    assert result.start_index == 16
    assert result.texts == tuple(
        f"translation {cue_id}" for cue_id in request.core_ids
    )


@pytest.mark.parametrize(
    "response",
    (
        "not json",
        '{"1": "translated"}',
        '{"1": "", "2": "translated"}',
    ),
)
def test_translation_output_rejects_incomplete_results(response: str) -> None:
    # Given: A two-cue request and a malformed or incomplete model response.
    request = build_translation_requests(("first", "second"))[0]

    # When/Then: Invalid output cannot become a partial subtitle file.
    with pytest.raises(TranslationOutputError):
        _ = parse_translation_output(response, request)
