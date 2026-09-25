from __future__ import annotations

from collections.abc import Callable, Iterable
import os
from pathlib import Path
import tempfile
from typing import TextIO

import pysrt
from tqdm.auto import tqdm


def inference_progress(
    stream: TextIO | None, label: str
) -> bool | Callable[..., tqdm[object]]:
    """Render vLLM's request progress separately from its native output."""
    if stream is None:
        return False

    def create_bar(
        iterable: Iterable[object] | None = None,
        *,
        total: int | None = None,
        desc: str = "",
        dynamic_ncols: bool = False,
        postfix: str | None = None,
    ) -> tqdm[object]:
        size = (
            os.get_terminal_size(stream.fileno())
            if stream.isatty()
            else os.terminal_size((80, 24))
        )
        columns = size.columns or 80
        return tqdm(
            iterable,
            total=total,
            desc=label + (" inputs" if desc.startswith("Rendering") else ""),
            unit="window",
            file=stream,
            position=1,
            leave=False,
            ncols=columns,
            nrows=size.lines or 24,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt}"
                if columns < 80
                else "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    return create_bar


def save_subtitles_atomic(path: Path, subtitles: pysrt.SubRipFile) -> None:
    """Replace an SRT only after a complete UTF-8 write in the same directory."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subtitles.save(str(temporary), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
