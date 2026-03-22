import argparse
import re
from pathlib import Path
from typing import List, Optional

import pysrt
from tqdm import tqdm

from src.logger import logger


def clean_text(text: str) -> str:
    """
    Cleans the subtitle text by removing SDH tags, HTML tags, and common hallucinations.
    """
    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 2. Remove SDH tags (e.g., [Music], (Applause), *Cheering*)
    # Text inside square brackets
    text = re.sub(r"\[.*?\]", "", text)
    # Text inside parentheses (careful not to remove legitimate dialog)
    # Heuristic: If it's all uppercase or clearly sound description
    text = re.sub(r"\([A-Z\s]+\)", "", text)
    # Text inside asterisks
    text = re.sub(r"\*.*?\*", "", text)
    # Music notes
    text = re.sub(r"[♪♫♬]", "", text)

    # 3. Remove common Whisper hallucinations / Metadata
    hallucinations = [
        "Subtitle by",
        "Translated by",
        "Amara.org",
        "Captioning by",
        "www.",
        ".com",
        "Sync and corrections by",
    ]
    for h in hallucinations:
        if h.lower() in text.lower():
            return ""  # Treat as garbage line

    # 4. Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_garbage(text: str) -> bool:
    """Returns True if the line should be removed completely."""
    if not text:
        return True
    # If it's just punctuation
    if re.fullmatch(r"[^\w\s]+", text):
        return True
    # If it's too short (e.g. single letter like "a" or "I" might be valid, but "t" isn't)
    # Heuristic: Keep it simple for now.
    return False


def is_filler(text: str) -> bool:
    """Detects low-information filler text using structure signals only. Language-agnostic."""
    # Strip non-word characters (punctuation, symbols, spaces), keep unicode word chars
    stripped = re.sub(r'[^\w]', '', text, flags=re.UNICODE).replace('_', '')
    if not stripped:
        return True
    # Rule A: effective char count <= 1
    if len(stripped) <= 1:
        return True
    # Rule B: all characters identical (e.g. "ああ", "ooo", "!!!")
    if len(set(stripped)) == 1:
        return True
    # Rule C: multi-word but all words identical (e.g. "no no no", "you you you")
    words = text.strip().split()
    if len(words) >= 2 and len(set(w.lower() for w in words)) == 1:
        return True
    return False


def is_duration_anomaly(sub: pysrt.SubRipItem) -> bool:
    """Detects segments with implausibly low speech density (long duration, tiny text)."""
    duration = (sub.end.ordinal - sub.start.ordinal) / 1000.0
    if duration <= 0:
        return True
    chars = len(sub.text.strip())
    if duration > 5.0 and chars / duration < 0.5:
        return True
    return False


def filter_consecutive_duplicates(
    subs: List[pysrt.SubRipItem],
    discard_threshold: int = 3,
) -> List[pysrt.SubRipItem]:
    """
    Handles consecutive duplicate subtitles:
    - run >= discard_threshold: discard entirely (non-linguistic filler)
    - run == 2: merge into one extended entry
    - run == 1: keep as-is
    """
    if not subs:
        return []
    result = []
    i = 0
    while i < len(subs):
        j = i + 1
        while j < len(subs) and subs[j].text.strip() == subs[i].text.strip():
            j += 1
        run_length = j - i
        if run_length >= discard_threshold:
            pass  # discard entire run
        elif run_length == 2:
            merged = subs[i]
            merged.end = subs[j - 1].end
            result.append(merged)
        else:
            result.append(subs[i])
        i = j
    return result


def clean_srt(file_path: Path, output_path: Optional[Path] = None):
    try:
        subs = pysrt.open(str(file_path))
    except Exception as e:
        logger.error("Error loading %s: %s", file_path, e)
        return

    original_count = len(subs)

    # Step 1: Clean text, filter garbage and filler
    cleaned_subs = []
    for sub in tqdm(subs, desc="Cleaning", unit="line"):
        cleaned_text = clean_text(sub.text)
        if is_garbage(cleaned_text) or is_filler(cleaned_text):
            continue
        sub.text = cleaned_text
        cleaned_subs.append(sub)

    count_after_text = len(cleaned_subs)

    # Step 2: Duration anomaly filter
    duration_filtered = [
        sub for sub in cleaned_subs if not is_duration_anomaly(sub)
    ]
    count_after_duration = len(duration_filtered)

    # Step 3: Filter consecutive duplicates
    final_subs = filter_consecutive_duplicates(duration_filtered)

    # Step 4: Re-index
    for i, sub in enumerate(final_subs):
        sub.index = i + 1

    # Step 5: Save
    out_file = output_path or file_path
    pysrt.SubRipFile(items=final_subs).save(str(out_file), encoding="utf-8")

    logger.info(
        "Result: %d -> %d (text filter) -> %d (duration filter) -> %d (dedup) lines.",
        original_count,
        count_after_text,
        count_after_duration,
        len(final_subs),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean SRT subtitles (Remove SDH, HTML, Hallucinations)."
    )
    parser.add_argument("input", help="Path to the input SRT file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output SRT file (optional, defaults to inplace).",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    if not input_path.exists():
        logger.error("Error: %s not found.", input_path)
        exit(1)

    clean_srt(input_path, output_path)


if __name__ == "__main__":
    main()
