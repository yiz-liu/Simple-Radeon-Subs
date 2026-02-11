import argparse
import sys
import tempfile
import time
from pathlib import Path

from src.audio import AudioExtractor
from src.clean import clean_srt
from src.config import GEMINI_API_KEY, GEMINI_API_URL
from src.logger import logger
from src.transcribe import Transcriber
from src.translate import GeminiTranslator

PROJECT_ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Movie Subtitle Translator Pipeline."
    )
    parser.add_argument("input", help="Path to the input video/audio file.")
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
        "--keep-temp", action="store_true", help="Keep temporary files for debugging."
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Determine final output directory
    final_output_dir = Path(args.output_dir or input_path.parent).resolve()
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Define filenames
    base_name = input_path.stem
    final_srt_name = f"{base_name}.{args.lang}.srt"
    final_srt_path = final_output_dir / final_srt_name

    start_time = time.time()

    # Create a temporary directory for intermediate files
    if args.keep_temp:
        temp_dir_obj = None
        temp_dir = PROJECT_ROOT / "temp_workspace"
        temp_dir.mkdir(exist_ok=True)
        logger.warning("Temp files will be kept in: %s", temp_dir)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="movie_translator_")
        temp_dir = Path(temp_dir_obj.name)
        logger.info("Using temporary directory: %s", temp_dir)

    try:
        # =================================================================
        # Step 1: Extract Audio
        # =================================================================
        logger.info("Step 1: Extract Audio")
        audio_extractor = AudioExtractor()
        temp_wav_path = temp_dir / f"{base_name}.wav"

        audio_path = audio_extractor.extract(
            input_path, output_path=temp_wav_path, force=True
        )

        # =================================================================
        # Step 2: Transcribe
        # =================================================================
        logger.info("Step 2: Transcribe Audio")
        transcriber = Transcriber(model_name=args.model)
        # Transcriber saves .srt to output_dir
        raw_srt_path = transcriber.transcribe(
            audio_path, output_dir=temp_dir, language=args.src_lang, verbose=True
        )

        # =================================================================
        # Step 3: Clean & Merge
        # =================================================================
        logger.info("Step 3: Clean Subtitles")
        cleaned_srt_path = temp_dir / f"{base_name}.cleaned.srt"

        # clean_srt reads input, writes to output (if provided)
        clean_srt(raw_srt_path, output_path=cleaned_srt_path)

        # =================================================================
        # Step 4: Translate
        # =================================================================
        logger.info("Step 4: Translate to %s", args.lang)

        translator = GeminiTranslator(GEMINI_API_KEY, GEMINI_API_URL)
        translator.translate_srt(
            input_path=cleaned_srt_path,
            output_path=final_srt_path,  # Write directly to final destination
            target_lang=args.lang,
            batch_size=30,
            workers=10,
        )

        elapsed = time.time() - start_time
        logger.info("✅ All Done! Total time: %.2fs", elapsed)
        logger.info("💾 Output: %s", final_srt_path)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if temp_dir_obj:
            logger.info("Cleaning up temporary files...")
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
