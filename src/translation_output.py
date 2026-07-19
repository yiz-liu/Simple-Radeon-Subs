import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pysrt


def _build_final_subtitles(
    subtitles: pysrt.SubRipFile,
    translations: Sequence[str],
    translated_only: bool,
) -> pysrt.SubRipFile:
    items: list[pysrt.SubRipItem] = []
    for subtitle, translated in zip(subtitles, translations, strict=True):
        source = subtitle.text.strip()
        text = (
            translated if translated_only or not source else f"{source}\n{translated}"
        )
        items.append(
            pysrt.SubRipItem(
                index=len(items) + 1,
                start=subtitle.start,
                end=subtitle.end,
                text=text,
                position=subtitle.position,
            )
        )
    return pysrt.SubRipFile(items=items)


def _save_subtitles(path: Path, subtitles: pysrt.SubRipFile) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        subtitles.save(str(temporary_path), encoding="utf-8")
        _ = temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
