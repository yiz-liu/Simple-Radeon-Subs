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


def merge_consecutive_duplicates(
    subs: List[pysrt.SubRipItem],
) -> List[pysrt.SubRipItem]:
    """Merges consecutive subtitles with identical text."""
    if not subs:
        return []

    merged = [subs[0]]

    for current in subs[1:]:
        last = merged[-1]

        # Normalize for comparison
        if current.text.strip() == last.text.strip():
            # Extend the duration of the last subtitle
            last.end = current.end
        else:
            merged.append(current)

    return merged


def clean_srt(file_path: Path, output_path: Optional[Path] = None):
    try:
        subs = pysrt.open(str(file_path))
    except Exception as e:
        logger.error("Error loading %s: %s", file_path, e)
        return

    original_count = len(subs)

    # 1. Clean Text & Filter Garbage with progress bar
    cleaned_subs = []
    for sub in tqdm(subs, desc="Cleaning", unit="line"):
        cleaned_text = clean_text(sub.text)
        if not is_garbage(cleaned_text):
            sub.text = cleaned_text
            cleaned_subs.append(sub)

    count_after_filter = len(cleaned_subs)

    # 2. Merge Duplicates
    final_subs = merge_consecutive_duplicates(cleaned_subs)

    # Re-index
    for i, sub in enumerate(final_subs):
        sub.index = i + 1

    # Save
    out_file = output_path or file_path
    pysrt.SubRipFile(items=final_subs).save(str(out_file), encoding="utf-8")

    logger.info(
        "Result: %d -> %d (filtered) -> %d (merged) lines.",
        original_count,
        count_after_filter,
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
