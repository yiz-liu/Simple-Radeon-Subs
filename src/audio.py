import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import AUDIO_CHANNELS, AUDIO_CODEC, AUDIO_SAMPLE_RATE, FFMPEG_LOCAL_PATH
from src.logger import logger


class AudioExtractor:
    """
    Extracts audio from video files and optimizes it for Whisper transcription.
    Target format: 16kHz, Mono, 16-bit PCM WAV.
    """

    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize the AudioExtractor.

        Args:
            ffmpeg_path (str, optional): Path to the ffmpeg binary.
                                         Defaults to searching in PATH or project-local tools.
        """
        self.ffmpeg_path = self._resolve_ffmpeg(ffmpeg_path)
        # ffprobe is usually in the same directory as ffmpeg
        self.ffprobe_path = str(Path(self.ffmpeg_path).parent / "ffprobe")
        if not Path(self.ffprobe_path).exists():
            # Fallback to system path if not found in the same folder
            self.ffprobe_path = shutil.which("ffprobe") or "ffprobe"

    def _resolve_ffmpeg(self, custom_path: Optional[str]) -> str:
        """Finds the ffmpeg executable."""
        if custom_path:
            path = Path(custom_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Custom ffmpeg path not found: {custom_path}")
            return str(path)

        # Priority 1: Check system PATH
        system_tool = shutil.which("ffmpeg")
        if system_tool:
            return system_tool

        # Priority 2: Check project-local tools
        project_tool = FFMPEG_LOCAL_PATH
        if project_tool.exists():
            return str(project_tool)

        raise FileNotFoundError(
            "ffmpeg binary not found. Please install it or place it in tools/ffmpeg/ffmpeg."
        )

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
        except Exception:
            return 0.0

    def _parse_time(self, line: str) -> Optional[float]:
        """Parses FFmpeg output line for time=HH:MM:SS.mm"""
        # Example: frame=  517 fps=0.0 q=-0.0 size=    1506kB time=00:00:20.64 bitrate= 597.1kbits/s
        match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
        if match:
            h, m, s = map(float, match.groups())
            return h * 3600 + m * 60 + s
        return None

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
            "-i",
            str(input_file),
            "-vn",
            "-acodec",
            AUDIO_CODEC,
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-ac",
            str(AUDIO_CHANNELS),
            "-stats",  # Force stats output
            str(output_file),
        ]

        # Use Popen to read stderr line by line
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier reading
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        with tqdm(total=int(duration), desc="Processing audio", unit="s") as pbar:
            last_time = 0.0
            if process.stdout:
                for line in process.stdout:
                    current_time = self._parse_time(line)
                    if current_time is not None:
                        diff = current_time - last_time
                        if diff > 0:
                            pbar.update(int(diff))
                            last_time = current_time

            process.wait()
            # Ensure bar reaches 100%
            if last_time < duration:
                pbar.update(int(duration - last_time))

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed to extract audio (code {process.returncode})."
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
    parser.add_argument("--ffmpeg-path", help="Custom path to the ffmpeg binary.")

    args = parser.parse_args()

    try:
        extractor = AudioExtractor(ffmpeg_path=args.ffmpeg_path)
        extractor.extract(args.input, output_path=args.output, force=args.force)
    except Exception as e:
        logger.error("Error: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
