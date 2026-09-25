import os
from pathlib import Path
import tempfile

import pysrt


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
