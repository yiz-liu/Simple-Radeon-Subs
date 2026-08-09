import argparse
import re
from itertools import pairwise
from pathlib import Path
from typing import Final, final

import pysrt

from src.logger import logger

DUPLICATE_MERGE_GAP_MS: Final = 250
EXCESSIVE_REPETITION_LENGTH: Final = 4
PHRASE_LOOP_TEXT_LENGTH: Final = 32
PRESERVED_REPETITION_COUNT: Final = 3
MAX_SUBTITLE_DURATION_MS: Final = 8_000
SHORTENED_SUBTITLE_DURATION_MS: Final = 8_000
NON_WORD_PATTERN: Final = re.compile(r"[^\w]", flags=re.UNICODE)
PHRASE_LOOP_PATTERN: Final = re.compile(r"(.{2,8})\1{3,}")
PATHOLOGICAL_REPETITION_PATTERN: Final = re.compile(r"(.{1,8}?)\1{3,}")


@final
class InvalidSubtitleTimelineError(ValueError):
    """Raised when an SRT cue has a non-positive duration."""

    def __init__(self, subtitle_index: int, start_ms: int, end_ms: int) -> None:
        self.subtitle_index = subtitle_index
        self.start_ms = start_ms
        self.end_ms = end_ms
        message = (
            f"Subtitle {subtitle_index} has a non-positive duration: "
            + f"{start_ms}ms -> {end_ms}ms"
        )
        super().__init__(message)


def clean_text(text: str) -> str:
    """Remove surrounding whitespace without changing subtitle content."""
    return text.strip()


def _effective_text(text: str) -> str:
    return NON_WORD_PATTERN.sub("", text).replace("_", "").casefold()


def is_garbage(text: str) -> bool:
    """Return whether a cue is empty or contains punctuation only."""
    return not _effective_text(text)


def is_excessive_repetition(text: str) -> bool:
    """Return whether effective text is an excessive character or phrase loop."""
    effective_text = _effective_text(text)
    return (
        len(effective_text) >= EXCESSIVE_REPETITION_LENGTH
        and len(set(effective_text)) == 1
    ) or (
        len(effective_text) >= PHRASE_LOOP_TEXT_LENGTH
        and PHRASE_LOOP_PATTERN.search(effective_text) is not None
    )


def filter_consecutive_duplicates(
    subtitles: list[pysrt.SubRipItem],
) -> list[pysrt.SubRipItem]:
    """Merge close duplicate pairs and discard close runs of three or more."""
    filtered: list[pysrt.SubRipItem] = []
    run_start = 0

    while run_start < len(subtitles):
        run_end = run_start + 1
        while run_end < len(subtitles):
            previous = subtitles[run_end - 1]
            current = subtitles[run_end]
            gap_ms = current.start.ordinal - previous.end.ordinal
            if (
                current.text != previous.text
                or not 0 <= gap_ms <= DUPLICATE_MERGE_GAP_MS
            ):
                break
            run_end += 1

        run_length = run_end - run_start
        if run_length == 1:
            filtered.append(subtitles[run_start])
        elif run_length == 2:
            first = subtitles[run_start]
            last = subtitles[run_end - 1]
            filtered.append(
                pysrt.SubRipItem(
                    index=first.index,
                    start=first.start,
                    end=last.end,
                    text=first.text,
                    position=first.position,
                )
            )

        run_start = run_end

    return filtered


def clean_srt(
    file_path: Path,
    output_path: Path | None = None,
    enable_vad: bool = True,
) -> None:
    """Conservatively normalize an SRT while preserving uncertain content."""
    input_file = file_path.resolve()
    output_file = output_path.resolve() if output_path is not None else input_file
    raw_content = input_file.read_bytes()
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as error:
        logger.warning(
            "Recovered invalid UTF-8 in subtitle input %s at byte %d.",
            input_file,
            error.start,
        )
        content = raw_content.decode("utf-8", errors="replace")
    subtitles: pysrt.SubRipFile = pysrt.from_string(content)
    original_count = len(subtitles)
    normalized: list[pysrt.SubRipItem] = []

    subtitle: pysrt.SubRipItem
    for subtitle in subtitles:
        if subtitle.end.ordinal < subtitle.start.ordinal:
            raise InvalidSubtitleTimelineError(
                subtitle.index, subtitle.start.ordinal, subtitle.end.ordinal
            )
        text = clean_text(subtitle.text)
        if is_garbage(text) or is_excessive_repetition(text):
            continue
        text = PATHOLOGICAL_REPETITION_PATTERN.sub(
            lambda match: match.group(1) * PRESERVED_REPETITION_COUNT,
            text,
        )
        normalized.append(
            pysrt.SubRipItem(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                text=text,
                position=subtitle.position,
            )
        )

    deduplicated = filter_consecutive_duplicates(normalized)
    final_subtitles: list[pysrt.SubRipItem] = []
    zero_duration_count = 0
    for subtitle in deduplicated:
        if subtitle.end.ordinal == subtitle.start.ordinal:
            zero_duration_count += 1
            continue
        if (
            enable_vad
            and subtitle.end.ordinal - subtitle.start.ordinal
            > MAX_SUBTITLE_DURATION_MS
        ):
            subtitle.start = pysrt.SubRipTime(
                milliseconds=subtitle.end.ordinal - SHORTENED_SUBTITLE_DURATION_MS
            )
        subtitle.index = len(final_subtitles) + 1
        final_subtitles.append(subtitle)

    overlap_count = sum(
        current.start.ordinal < previous.end.ordinal
        for previous, current in pairwise(final_subtitles)
    )

    pysrt.SubRipFile(items=final_subtitles).save(
        str(output_file),
        encoding="utf-8",
    )

    if overlap_count:
        logger.warning(
            "Preserved %d overlapping subtitle(s).",
            overlap_count,
        )
    logger.info(
        "Cleaned subtitles: %d input, %d safely filtered, "
        + "%d merged or duplicate-filtered, %d output.",
        original_count,
        original_count - len(normalized) + zero_duration_count,
        len(normalized) - len(deduplicated),
        len(final_subtitles),
    )


class _Arguments(argparse.Namespace):
    input: str = ""
    output: str | None = None
    enable_vad: bool = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively normalize SRT subtitles."
    )
    _ = parser.add_argument("input", help="Path to the input SRT file.")
    _ = parser.add_argument(
        "-o",
        "--output",
        help="Path to the output SRT file (defaults to in-place).",
    )
    _ = parser.add_argument(
        "--disable-vad",
        action="store_false",
        dest="enable_vad",
        help="Preserve long cue timestamps for subtitles produced without VAD.",
    )
    arguments = parser.parse_args(namespace=_Arguments())
    input_path = Path(arguments.input).resolve()
    output_path = Path(arguments.output).resolve() if arguments.output else None

    try:
        clean_srt(input_path, output_path, enable_vad=arguments.enable_vad)
    except (OSError, UnicodeError, pysrt.Error, InvalidSubtitleTimelineError) as error:
        logger.error("Subtitle cleaning failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
