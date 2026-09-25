from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import wave
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from src.config import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_SAMPLE_RATE,
    AUDIO_TIMELINE_FILTER,
    QWEN_WINDOW_MERGE_GAP_SECONDS,
    QWEN_WINDOW_TARGET_SECONDS,
    QWEN_VAD_MODEL_PATH,
    QWEN_VAD_THRESHOLD,
    QWEN_VAD_MIN_SPEECH_MS,
    QWEN_VAD_MIN_SILENCE_MS,
    QWEN_VAD_PAD_MS,
    QWEN_VAD_MAX_SECONDS,
)
from src.logger import logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


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
        progress_position: int | None = None,
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
        if progress_position is None:
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
            "-af",
            AUDIO_TIMELINE_FILTER,
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
                desc=(
                    "Processing audio"
                    if progress_position is None
                    else f"Current file: {input_file.name}"
                ),
                unit="s",
                position=progress_position or 0,
                leave=progress_position is None,
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


class InvalidAudio(ValueError):
    """Reject PCM inputs or speech windows outside the audio contract."""


def read_audio(path: Path) -> NDArray[np.float32]:
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            AUDIO_SAMPLE_RATE,
            1,
            2,
        ):
            raise InvalidAudio(f"Expected 16 kHz mono PCM16 WAV: {path}")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if not len(audio):
        raise InvalidAudio(f"Empty audio: {path}")
    return audio.astype(np.float32) / 32768.0


@dataclass(frozen=True, slots=True)
class SampleSpan:
    start: int
    end: int


def processing_windows(
    audio: NDArray[np.float32],
    spans: list[SampleSpan],
    target_seconds: float = QWEN_WINDOW_TARGET_SECONDS,
) -> list[SampleSpan]:
    if not 5 <= target_seconds <= 120:
        raise InvalidAudio("target_seconds must be between 5 and 120")
    groups: list[SampleSpan] = []
    previous_end = 0
    for span in spans:
        if not previous_end <= span.start < span.end <= len(audio):
            raise InvalidAudio(f"invalid or unordered audio span: {span}")
        previous_end = span.end
        if (
            groups
            and span.start - groups[-1].end
            <= QWEN_WINDOW_MERGE_GAP_SECONDS * AUDIO_SAMPLE_RATE
        ):
            groups[-1] = SampleSpan(groups[-1].start, span.end)
        else:
            groups.append(span)
    return [
        SampleSpan(start, end)
        for group in groups
        for start, end in partition_audio(audio, group.start, group.end, target_seconds)
    ]


def partition_audio(
    audio: NDArray[np.float32], start: int, end: int, target_seconds: float
) -> list[tuple[int, int]]:
    """Partition without gaps, choosing low RMS boundaries near balanced targets."""
    import numpy as np

    pieces = math.ceil((end - start) / (target_seconds * AUDIO_SAMPLE_RATE))
    if pieces <= 1:
        return [(start, end)]
    points = [start]
    for part in range(1, pieces):
        target = start + round((end - start) * part / pieces)
        low = max(points[-1] + AUDIO_SAMPLE_RATE, target - 2 * AUDIO_SAMPLE_RATE)
        high = min(
            end - AUDIO_SAMPLE_RATE * (pieces - part), target + 2 * AUDIO_SAMPLE_RATE
        )
        candidates = list(range(low, high + 1, 256))
        point = min(
            candidates,
            key=lambda p: float(np.mean(np.square(audio[p - 256 : p + 256]))),
        )
        points.append(point)
    points.append(end)
    return list(zip(points, points[1:]))


class WindowPreparer:
    def __init__(self, enable_vad: bool) -> None:
        from silero_vad.utils_vad import OnnxWrapper

        self.model = (
            OnnxWrapper(str(QWEN_VAD_MODEL_PATH), force_onnx_cpu=True)
            if enable_vad
            else None
        )

    def prepare(self, audio_path: Path) -> list[SampleSpan]:
        import torch
        from silero_vad import get_speech_timestamps

        audio = read_audio(audio_path)
        spans = [SampleSpan(0, len(audio))]
        if self.model is not None:
            regions = get_speech_timestamps(
                torch.from_numpy(audio),
                self.model,
                sampling_rate=AUDIO_SAMPLE_RATE,
                return_seconds=False,
                threshold=QWEN_VAD_THRESHOLD,
                min_speech_duration_ms=QWEN_VAD_MIN_SPEECH_MS,
                min_silence_duration_ms=QWEN_VAD_MIN_SILENCE_MS,
                speech_pad_ms=QWEN_VAD_PAD_MS,
                max_speech_duration_s=QWEN_VAD_MAX_SECONDS,
            )
            spans = [SampleSpan(int(s["start"]), int(s["end"])) for s in regions]
        return processing_windows(audio, spans)


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
