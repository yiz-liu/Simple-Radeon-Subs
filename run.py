import argparse
import sys
from pathlib import Path
from typing import Final

from src.logger import logger
from src.config import ASRBackend
from src.pipeline import (
    WORK_DIR_SUFFIX,
    PipelineOptions,
    PipelinePlan,
    build_jobs,
    run_pipeline,
)

MEDIA_EXTENSIONS: Final = {
    ".3gp",
    ".aac",
    ".avi",
    ".flac",
    ".flv",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".wav",
    ".webm",
    ".wmv",
}


def scan_directory(directory: Path) -> list[Path]:
    """Find supported media files recursively."""
    if directory.name.endswith(WORK_DIR_SUFFIX):
        return []
    media_files: list[Path] = []
    for root, directories, filenames in directory.walk():
        directories[:] = [
            name for name in directories if not name.endswith(WORK_DIR_SUFFIX)
        ]
        media_files.extend(
            root / filename
            for filename in filenames
            if Path(filename).suffix.lower() in MEDIA_EXTENSIONS
        )
    return sorted(media_files)


class _Arguments(argparse.Namespace):
    input: str = ""
    output_dir: str | None = None
    lang: str = "Chinese"
    src_lang: str | None = None
    keep_temp: bool = False
    force: bool = False
    translated_only: bool = False
    enable_vad: bool = True
    asr_backend: ASRBackend = "whisper"


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        description="End-to-End Movie Subtitle Translator Pipeline."
    )
    _ = parser.add_argument(
        "input", help="Path to a media file or a directory scanned recursively."
    )
    _ = parser.add_argument(
        "--asr-backend",
        choices=("whisper", "qwen"),
        default="whisper",
        help="Transcription backend (default: whisper).",
    )
    _ = parser.add_argument("-o", "--output-dir", help="Final subtitle directory.")
    _ = parser.add_argument("--lang", default="Chinese", help="Target language.")
    _ = parser.add_argument(
        "--src-lang",
        default=None,
        help="Source language code. Auto-detects if omitted.",
    )
    _ = parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep successful intermediate sidecars.",
    )
    _ = parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Rebuild selected jobs from clean sidecars.",
    )
    _ = parser.add_argument(
        "--translated-only",
        action="store_true",
        help="Generate translated subtitles without source text.",
    )
    _ = parser.add_argument(
        "--disable-vad",
        action="store_false",
        dest="enable_vad",
        help="Disable integrated VAD and transcribe the complete audio stream.",
    )
    return parser.parse_args(namespace=_Arguments())


def main() -> int:
    arguments = _parse_arguments()
    input_path = Path(arguments.input).resolve()
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return 1

    if input_path.is_dir():
        input_paths = tuple(scan_directory(input_path))
        input_root = input_path
        if not input_paths:
            logger.warning("No media files found in directory.")
            return 0
    else:
        input_paths = (input_path,)
        input_root = input_path.parent

    options = PipelineOptions(
        target_lang=arguments.lang,
        src_lang=arguments.src_lang,
        keep_temp=arguments.keep_temp,
        force=arguments.force,
        translated_only=arguments.translated_only,
        enable_vad=arguments.enable_vad,
        asr_backend=arguments.asr_backend,
    )
    try:
        jobs = build_jobs(
            PipelinePlan(
                input_paths=input_paths,
                input_root=input_root,
                output_dir=(
                    Path(arguments.output_dir).resolve()
                    if arguments.output_dir is not None
                    else None
                ),
                options=options,
            )
        )
        result = run_pipeline(jobs, options)
    except (OSError, ValueError) as error:
        logger.error("Pipeline setup failed: %s", error)
        return 1

    logger.info(
        "Batch complete: %d succeeded, %d skipped, %d failed.",
        len(result.succeeded),
        len(result.skipped),
        len(result.failures),
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
