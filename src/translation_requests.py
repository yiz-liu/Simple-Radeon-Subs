from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, override

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

CORE_CUE_COUNT: Final = 16
CONTEXT_CUE_COUNT: Final = 4


class _StringSchema(TypedDict):
    type: Literal["string"]
    minLength: int


class TranslationSchema(TypedDict):
    type: Literal["object"]
    properties: dict[str, _StringSchema]
    required: list[str]
    additionalProperties: bool


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """One core subtitle range plus translated context on either side."""

    core_start: int
    core_stop: int
    window_start: int
    texts: tuple[str, ...]

    @property
    def core_ids(self) -> tuple[str, ...]:
        return tuple(str(index + 1) for index in range(self.core_start, self.core_stop))

    @property
    def window_ids(self) -> tuple[str, ...]:
        window_stop = self.window_start + len(self.texts)
        return tuple(str(index + 1) for index in range(self.window_start, window_stop))

    @property
    def schema(self) -> TranslationSchema:
        ids = self.window_ids
        return {
            "type": "object",
            "properties": {
                cue_id: {"type": "string", "minLength": 1} for cue_id in ids
            },
            "required": list(ids),
            "additionalProperties": False,
        }

    def messages(self, target_lang: str) -> list[ChatCompletionMessageParam]:
        """Build the model conversation for this subtitle window."""
        targets = dict(zip(self.window_ids, self.texts, strict=True))
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional movie subtitle translator. Translate every "
                    f"value in targets into {target_lang}. Treat the targets as continuous "
                    "dialogue and preserve meaning, tone, proper names, and relationships. "
                    "Return every translation under the same ID. Do not merge, split, "
                    "reorder, omit, explain, or copy source text. Return only the required "
                    "structured result."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"targets": targets},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Validated translations owned by one non-overlapping core."""

    start_index: int
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranslationOutputError(RuntimeError):
    """Report malformed or incomplete structured model output."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def build_translation_requests(texts: Sequence[str]) -> list[TranslationRequest]:
    """Split subtitle cues into fixed cores with overlapping translated context."""
    requests: list[TranslationRequest] = []
    for core_start in range(0, len(texts), CORE_CUE_COUNT):
        core_stop = min(core_start + CORE_CUE_COUNT, len(texts))
        window_start = max(0, core_start - CONTEXT_CUE_COUNT)
        window_stop = min(len(texts), core_stop + CONTEXT_CUE_COUNT)
        requests.append(
            TranslationRequest(
                core_start=core_start,
                core_stop=core_stop,
                window_start=window_start,
                texts=tuple(texts[window_start:window_stop]),
            )
        )
    return requests


def parse_translation_output(
    raw_text: str,
    request: TranslationRequest,
) -> TranslationResult:
    """Parse an exact-ID response and retain only its non-overlapping core."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise TranslationOutputError("Translation output is not valid JSON") from error

    expected_ids = request.window_ids
    if not isinstance(payload, dict) or set(payload) != set(expected_ids):
        raise TranslationOutputError("Translation output has an unexpected set of IDs")

    translations: dict[str, str] = {}
    for cue_id in expected_ids:
        value = payload[cue_id]
        if not isinstance(value, str) or not value.strip():
            raise TranslationOutputError(
                f"Translation output for subtitle {cue_id} is empty or invalid"
            )
        translations[cue_id] = value.strip()

    return TranslationResult(
        start_index=request.core_start,
        texts=tuple(translations[cue_id] for cue_id in request.core_ids),
    )
