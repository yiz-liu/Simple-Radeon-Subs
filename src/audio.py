import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import AUDIO_CHANNELS, AUDIO_CODEC, AUDIO_SAMPLE_RATE
from src.logger import logger


class AudioExtractionError(RuntimeError):
    """Report an FFmpeg failure while extracting audio."""


def parse_out_time_seconds(line: str) -> float | None:
    key, separator, raw_value = line.partition("=")
    if separator != "=" or key != "out_time_us":
        return None

    try:
        microseconds = int(raw_value)
    except ValueError:
        return None

    if microseconds < 0:
        return None
    return microseconds / 1_000_000


class AudioExtractor:
    """
    Extracts audio from video files and optimizes it for Whisper transcription.
    Target format: 16kHz, Mono, 16-bit PCM WAV.
    """

    def __init__(self):
        """Initialize the audio extractor with system FFmpeg tools."""
        self.ffmpeg_path = self._resolve_binary("ffmpeg")
        self.ffprobe_path = self._resolve_binary("ffprobe")

    @staticmethod
    def _resolve_binary(name: str) -> str:
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(
                f"Required system executable '{name}' was not found on PATH. "
                "Install FFmpeg with: sudo apt install ffmpeg"
            )
        return executable

    def get_duration(self, input_path: Path) -> float:
        """Gets the duration of the input file in seconds using ffprobe."""
        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        try:
            output = subprocess.check_output(cmd).decode().strip()
            return float(output)
        except (OSError, subprocess.CalledProcessError, ValueError):
            logger.warning(
                "Unable to determine media duration: %s",
                input_path,
                exc_info=True,
            )
            return 0.0

    def extract(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        force: bool = False,
    ) -> Path:
        """
        Extracts audio from the input video/audio file with a progress bar.
        """
        input_file = Path(input_path).resolve()

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_path:
            output_file = Path(output_path).resolve()
        else:
            output_file = input_file.with_suffix(".wav")

        # Skip if already exists and not forcing overwrite
        if output_file.exists() and not force:
            logger.info(
                "Audio file already exists: %s (Skipping extraction)", output_file
            )
            return output_file

        duration = self.get_duration(input_file)
        logger.info("Extracting optimized audio: %s", input_file.name)

        # Construct FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(input_file),
            "-vn",
            "-acodec",
            AUDIO_CODEC,
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-ac",
            str(AUDIO_CHANNELS),
            str(output_file),
        ]

        progress_total = duration if duration > 0 else None
        with subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            with tqdm(
                total=progress_total,
                desc="Processing audio",
                unit="s",
            ) as pbar:
                last_time = 0.0
                if process.stdout is not None:
                    for line in process.stdout:
                        current_time = parse_out_time_seconds(line.strip())
                        if current_time is None:
                            continue
                        displayed_time = (
                            min(current_time, duration)
                            if duration > 0
                            else current_time
                        )
                        if displayed_time > last_time:
                            pbar.update(displayed_time - last_time)
                            last_time = displayed_time

                return_code = process.wait()
                if return_code == 0 and duration > 0 and last_time < duration:
                    pbar.update(duration - last_time)

        if return_code != 0:
            raise AudioExtractionError(
                f"FFmpeg failed to extract audio (code {return_code})."
            )

        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract Whisper-optimized audio from video files."
    )
    parser.add_argument("input", help="Path to the input video/audio file.")
    parser.add_argument(
        "-o", "--output", help="Path to the output .wav file (optional)."
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite if output exists."
    )
    args = parser.parse_args()

    try:
        extractor = AudioExtractor()
        extractor.extract(args.input, output_path=args.output, force=args.force)
    except (AudioExtractionError, FileNotFoundError, OSError, ValueError) as error:
        logger.error("Error: %s", error, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
