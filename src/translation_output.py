from collections.abc import Sequence

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
