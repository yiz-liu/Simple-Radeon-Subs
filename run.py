import argparse
import gc
import shutil
import sys
import time
import torch
from pathlib import Path
from typing import List, Optional

from src.audio import AudioExtractor
from src.clean import clean_srt
from src.config import TRANSLATION_PROVIDER
from src.logger import logger
from src.transcribe import Transcriber
from src.translate import create_translator

PROJECT_ROOT = Path(__file__).parent

# Supported video extensions for directory scanning
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
}


def cleanup_intermediate_files(paths: List[Path]) -> None:
    """Deletes intermediate files or directories if they exist."""
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def process_video(
    input_path: Path,
    output_dir: Optional[Path],
    target_lang: str,
    src_lang: Optional[str],
    model_name: str,
    keep_temp: bool,
    force: bool,
    translated_only: bool,
    provider: Optional[str] = None,
):
    """
    Processes a single video file: Extract -> Transcribe -> Clean -> Translate.
    """
    # Determine final output directory
    final_output_dir = (output_dir or input_path.parent).resolve()
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Define filenames
    base_name = input_path.stem
    final_srt_name = f"{base_name}.{target_lang}.srt"
    final_srt_path = final_output_dir / final_srt_name
    temp_wav_path = final_output_dir / f"{base_name}.wav"
    raw_srt_path = final_output_dir / f"{base_name}.srt"
    cleaned_srt_path = final_output_dir / f"{base_name}.cleaned.srt"
    intermediate_paths = [temp_wav_path, raw_srt_path, cleaned_srt_path]

    # Check if output already exists
    if final_srt_path.exists() and not force:
        logger.info("Subtitle already exists: %s (Skipping)", final_srt_path)
        return

    if force:
        cleanup_intermediate_files(intermediate_paths)

    logger.info("Processing: %s", input_path.name)
    start_time = time.time()

    logger.info("Intermediate files will be stored in: %s", final_output_dir)

    try:
        # =================================================================
        # Step 1: Extract Audio
        # =================================================================
        logger.info("[%s] Step 1: Extract Audio", base_name)
        audio_extractor = AudioExtractor()
        audio_path = audio_extractor.extract(
            input_path, output_path=temp_wav_path, force=force
        )

        # =================================================================
        # Step 2: Transcribe
        # =================================================================
        logger.info("[%s] Step 2: Transcribe Audio", base_name)
        transcriber = Transcriber(model_name=model_name)
        if raw_srt_path.exists() and not force:
            logger.info("Reusing transcription: %s", raw_srt_path)
        else:
            raw_srt_path = transcriber.transcribe(
                audio_path,
                output_dir=final_output_dir,
                language=src_lang,
                verbose=False,
            )

        # Free GPU memory from Whisper before potential vLLM translation
        del transcriber
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU memory released after transcription.")

        # =================================================================
        # Step 3: Clean & Merge
        # =================================================================
        logger.info("[%s] Step 3: Clean Subtitles", base_name)
        if cleaned_srt_path.exists() and not force:
            logger.info("Reusing cleaned subtitles: %s", cleaned_srt_path)
        else:
            clean_srt(raw_srt_path, output_path=cleaned_srt_path)

        # =================================================================
        # Step 4: Translate
        # =================================================================
        logger.info("[%s] Step 4: Translate to %s", base_name, target_lang)

        translator = create_translator(provider)
        translator.translate_srt(
            input_path=cleaned_srt_path,
            output_path=final_srt_path,
            target_lang=target_lang,
            translated_only=translated_only,
        )

        elapsed = time.time() - start_time
        logger.info("✅ Done: %s (Time: %.2fs)", input_path.name, elapsed)
        logger.info("💾 Output: %s", final_srt_path)

        if keep_temp:
            logger.info("Keeping intermediate files in: %s", final_output_dir)
        else:
            cleanup_intermediate_files(intermediate_paths)
            logger.info("Cleaned intermediate files for %s.", input_path.name)

    except Exception as e:
        logger.error("Failed to process %s: %s", input_path.name, e, exc_info=True)


def scan_directory(directory: Path) -> List[Path]:
    """Recursively scans a directory and all subdirectories for supported video files."""
    video_files = []
    for file in directory.rglob("*"):
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(file)
    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Movie Subtitle Translator Pipeline."
    )
    parser.add_argument(
        "input", help="Path to the input video/audio file or directory."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save the final translated subtitle. Defaults to input file's directory.",
    )
    parser.add_argument(
        "--lang",
        default="Chinese",
        help="Target language for translation (default: Chinese).",
    )
    parser.add_argument(
        "--src-lang",
        default=None,
        help="Source language of the audio (e.g., 'en', 'zh'). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files after successful completion.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite final subtitles and restart from clean intermediate files.",
    )
    parser.add_argument(
        "--translated-only",
        action="store_true",
        help="Generate translated subtitles only instead of bilingual subtitles.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "vllm"],
        default=None,
        help="Translation provider (overrides TRANSLATION_PROVIDER env var).",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    # Identify files to process
    files_to_process = []
    if input_path.is_dir():
        logger.info("Scanning directory: %s", input_path)
        files_to_process = scan_directory(input_path)
        if not files_to_process:
            logger.warning("No video files found in directory.")
            sys.exit(0)
        logger.info("Found %d video files.", len(files_to_process))
    else:
        files_to_process = [input_path]

    # Process files
    total_files = len(files_to_process)
    for i, file_path in enumerate(files_to_process, 1):
        logger.info("------------------------------------------------------------")
        logger.info("Processing File %d/%d: %s", i, total_files, file_path.name)
        logger.info("------------------------------------------------------------")

        process_video(
            input_path=file_path,
            output_dir=output_dir,
            target_lang=args.lang,
            src_lang=args.src_lang,
            model_name=args.model,
            keep_temp=args.keep_temp,
            force=args.force,
            translated_only=args.translated_only,
            provider=args.provider,
        )

    logger.info("All tasks completed.")


if __name__ == "__main__":
    main()
