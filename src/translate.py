import argparse
from pathlib import Path

from src.logger import logger
from src.translation import (
    DEFAULT_BATCH_SIZE,
    TranslationOptions,
    VLLMTranslator,
)


class _Arguments(argparse.Namespace):
    input: str = ""
    output: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    lang: str = "Chinese"
    translated_only: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles with the managed local vLLM model."
    )
    _ = parser.add_argument("input", help="Path to the input SRT file.")
    _ = parser.add_argument("-o", "--output", help="Path to the output SRT file.")
    _ = parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Subtitle lines per model prompt. Default: {DEFAULT_BATCH_SIZE}.",
    )
    _ = parser.add_argument("--lang", default="Chinese", help="Target language.")
    _ = parser.add_argument(
        "--translated-only",
        action="store_true",
        help="Generate translated subtitles without the source text.",
    )
    args = parser.parse_args(namespace=_Arguments())
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"Input SRT not found: {input_path}")

    output_path = Path(
        args.output
        or input_path.parent
        / f"{input_path.stem}.{args.lang.lower().replace(' ', '_')}.srt"
    ).resolve()
    VLLMTranslator().translate_srt(
        input_path,
        output_path,
        TranslationOptions(
            target_lang=args.lang,
            batch_size=args.batch_size,
            translated_only=args.translated_only,
        ),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
