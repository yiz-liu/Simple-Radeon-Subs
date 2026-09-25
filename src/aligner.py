from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING
import unicodedata

from pydantic import BaseModel, ConfigDict, TypeAdapter

from src.config import (
    AUDIO_SAMPLE_RATE as SAMPLE_RATE,
    QWEN_ALIGNER_MODEL_PATH,
    QWEN_FALLBACK_MIN_SECONDS,
    QWEN_SHORT_WINDOW_SECONDS,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from vllm import LLM


class AlignmentError(ValueError):
    """Report inconsistent transcript units or invalid alignment results."""


class AlignmentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    start: int
    end: int
    text: str
    language: str | None = None
    units: tuple[str, ...] = ()
    fixed: tuple[float, ...] = ()
    error: str | None = None


class SubtitleCue(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    sentence: int
    start: float
    end: float
    text: str
    published: bool


def read_records(path: Path) -> list[AlignmentRecord]:
    return TypeAdapter(list[AlignmentRecord]).validate_json(path.read_bytes())


def write_records(path: Path, records: list[AlignmentRecord]) -> None:
    path.write_bytes(TypeAdapter(list[AlignmentRecord]).dump_json(records))


def restore_text(text: str, units: list[str]) -> list[str]:
    positions: list[int] = []
    normalized: list[str] = []
    for index, char in enumerate(text):
        if char.isalnum():
            normalized.append(char)
            positions.append(index)
    joined = "".join(normalized)
    cursor = 0
    boundaries = [0]
    for unit in units:
        clean = "".join(c for c in unit if c.isalnum())
        if not clean or not joined.startswith(clean, cursor):
            raise AlignmentError(
                "Alignment units do not match transcript; refusing to drop text"
            )
        cursor += len(clean)
        boundaries.append(positions[cursor] if cursor < len(positions) else len(text))
    if cursor != len(joined):
        raise AlignmentError("Alignment units do not cover the complete transcript")
    return [text[a:b] for a, b in zip(boundaries, boundaries[1:])]


def display_width(text: str) -> float:
    return sum(1 if unicodedata.east_asian_width(c) in "WF" else 0.5 for c in text)


def wrap_text(parts: list[str]) -> str:
    if (
        len(parts) > 1
        and all(len(part) == 1 for part in parts)
        and " " in "".join(parts)
    ):
        parts = re.findall(r"\S+\s*|^\s+", "".join(parts))
    if display_width("".join(parts)) <= 20 or len(parts) < 2:
        return "".join(parts)
    split = min(
        range(1, len(parts)),
        key=lambda i: abs(
            display_width("".join(parts[:i])) - display_width("".join(parts[i:]))
        ),
    )
    return "".join(parts[:split]) + "\n" + "".join(parts[split:])


def sentence_groups(
    text: str, units: tuple[str, ...]
) -> tuple[tuple[int, int, str], ...]:
    if not units:
        return ()
    parts = restore_text(text, list(units))
    groups: list[tuple[int, int, str]] = []
    first = 0
    for i, part in enumerate(parts):
        if (
            re.search(r"[。？！?!;；؛…]|(?<!\d)\.|\.(?!\d)", part)
            or i == len(parts) - 1
        ):
            groups.append((first, i + 1, "".join(parts[first : i + 1])))
            first = i + 1
    return tuple(groups)


def render_record(record: AlignmentRecord) -> list[SubtitleCue]:
    groups = sentence_groups(record.text, record.units)
    offset, duration = (
        record.start / SAMPLE_RATE,
        (record.end - record.start) / SAMPLE_RATE,
    )
    if record.error or not groups or len(record.fixed) != 2 * len(record.units):
        return (
            [
                SubtitleCue(
                    segment_id=record.segment_id,
                    sentence=0,
                    start=offset,
                    end=offset,
                    text=record.text,
                    published=False,
                )
            ]
            if record.text
            else []
        )
    cues: list[SubtitleCue] = []
    for number, (first, stop, text) in enumerate(groups):
        start = record.fixed[2 * first]
        end = record.fixed[2 * (stop - 1) + 1]
        valid = (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start < end <= duration + 0.001
            and round((offset + end) * 1000) > round((offset + start) * 1000)
        )
        cues.append(
            SubtitleCue(
                segment_id=record.segment_id,
                sentence=number,
                start=offset + start,
                end=offset + end,
                text=wrap_text(restore_text(text, list(record.units[first:stop]))),
                published=valid,
            )
        )
    return cues


def needs_alignment(record: AlignmentRecord) -> bool:
    return (
        bool(record.text)
        and not record.error
        and (record.end - record.start > QWEN_SHORT_WINDOW_SECONDS * SAMPLE_RATE)
    )


def merge_alignments(
    records: list[AlignmentRecord], aligned: list[AlignmentRecord]
) -> list[AlignmentRecord]:
    expected = {r.segment_id: r for r in records if needs_alignment(r)}
    actual = {r.segment_id: r for r in aligned}
    if expected.keys() != actual.keys() or len(actual) != len(aligned):
        raise AlignmentError("Alignment result IDs mismatch")
    for key, result in actual.items():
        source = expected[key]
        if (source.start, source.end, source.text) != (
            result.start,
            result.end,
            result.text,
        ):
            raise AlignmentError(f"Alignment input changed: {key}")
    return [actual.get(r.segment_id, r) for r in records]


def _combine(cues: list[SubtitleCue], span: tuple[float, float]) -> SubtitleCue:
    text = "".join(c.text.replace("\n", "") for c in cues)
    return cues[0].model_copy(
        update={
            "start": span[0],
            "end": span[1],
            "text": wrap_text(list(text)),
            "published": True,
        }
    )


def render_timed_record(record: AlignmentRecord) -> list[SubtitleCue]:
    if not record.text:
        return []
    if record.error:
        return render_record(record)
    lower, upper = record.start / SAMPLE_RATE, record.end / SAMPLE_RATE
    if not needs_alignment(record):
        return [
            SubtitleCue(
                segment_id=record.segment_id,
                sentence=0,
                start=lower,
                end=upper,
                text=wrap_text(list(record.text)),
                published=True,
            )
        ]

    candidates: list[SubtitleCue] = []
    previous_end = lower
    for cue in render_record(record):
        start, end = max(lower, cue.start), min(upper, cue.end)
        valid = (
            math.isfinite(cue.start)
            and math.isfinite(cue.end)
            and cue.start < cue.end
            and start >= previous_end
            and round(end * 1000) > round(start * 1000)
        )
        candidates.append(
            cue.model_copy(
                update={
                    "start": start if valid else cue.start,
                    "end": end if valid else cue.end,
                    "published": valid,
                }
            )
        )
        if valid:
            previous_end = end

    output: list[SubtitleCue] = []
    position = 0
    while position < len(candidates):
        if candidates[position].published:
            output.append(candidates[position])
            position += 1
            continue
        stop = position + 1
        while stop < len(candidates) and not candidates[stop].published:
            stop += 1
        group = candidates[position:stop]
        left = output[-1].end if output else lower
        right = candidates[stop].start if stop < len(candidates) else upper
        if right - left >= QWEN_FALLBACK_MIN_SECONDS:
            output.append(_combine(group, (left, right)))
        elif output:
            previous = output.pop()
            output.append(_combine([previous, *group], (previous.start, right)))
        elif stop < len(candidates):
            following = candidates[stop]
            output.append(_combine([*group, following], (lower, following.end)))
            stop += 1
        position = stop
    return output


class ForcedAligner:
    def __init__(self, llm: LLM) -> None:
        from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor

        self.processor = Qwen3ASRProcessor.from_pretrained(
            str(QWEN_ALIGNER_MODEL_PATH), local_files_only=True
        )
        config = json.loads((QWEN_ALIGNER_MODEL_PATH / "config.json").read_bytes())
        self.timestamp_token_id = int(config["timestamp_token_id"])
        self.timestamp_segment_time = float(config["timestamp_segment_time"])
        self.llm = llm

    def align(
        self,
        records: list[AlignmentRecord],
        audio: NDArray[np.float32],
    ) -> list[AlignmentRecord]:
        import torch
        from vllm import PoolingParams
        from vllm.inputs import TextPrompt

        prompts: list[TextPrompt] = []
        prepared: list[AlignmentRecord] = []
        for record in records:
            units = tuple(
                self.processor.split_words_for_alignment(
                    record.text,
                    language=record.language,
                )
            )
            if not units:
                raise AlignmentError(f"No alignable units: {record.segment_id}")
            sentence_groups(record.text, units)
            prepared.append(record.model_copy(update={"units": units}))
            prompts.append(
                {
                    "prompt": "<|audio_start|><|audio_pad|><|audio_end|>"
                    + "".join(unit + "<timestamp><timestamp>" for unit in units),
                    "multi_modal_data": {
                        "audio": (audio[record.start : record.end], SAMPLE_RATE)
                    },
                }
            )
        outputs = self.llm.encode(
            prompts,
            pooling_task="token_classify",
            pooling_params=PoolingParams(use_activation=False),
            use_tqdm=False,
        )
        result: list[AlignmentRecord] = []
        for record, output in zip(prepared, outputs, strict=True):
            ids = torch.tensor(output.prompt_token_ids)
            scores = output.outputs.data.cpu().float()
            mask = ids == self.timestamp_token_id
            if scores.shape[0] != len(ids) or int(mask.sum()) != 2 * len(record.units):
                raise AlignmentError(
                    f"Timestamp slot count mismatch: {record.segment_id}"
                )
            decoded = self.processor.decode_forced_alignment(
                scores.unsqueeze(0),
                ids.unsqueeze(0),
                [list(record.units)],
                timestamp_token_id=self.timestamp_token_id,
                timestamp_segment_time=self.timestamp_segment_time,
            )[0]
            fixed = tuple(
                float(value)
                for unit in decoded
                for value in (unit["start_time"], unit["end_time"])
            )
            result.append(record.model_copy(update={"fixed": fixed}))
        return result
