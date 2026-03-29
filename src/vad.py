import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import whisper

from src.config import (
    AUDIO_SAMPLE_RATE,
    VAD_MAX_SPEECH_DURATION_S,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
)
from src.logger import logger

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except ImportError:
    get_speech_timestamps = None
    load_silero_vad = None


@dataclass(slots=True)
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def validate_clip_timestamps(clip_timestamps: list[float]) -> list[float]:
    normalized = [float(timestamp) for timestamp in clip_timestamps]

    if len(normalized) % 2 != 0:
        raise ValueError("clip_timestamps must contain an even number of values.")

    previous_end = -1.0
    for index in range(0, len(normalized), 2):
        start = normalized[index]
        end = normalized[index + 1]

        if start < 0 or end < 0:
            raise ValueError("clip_timestamps cannot contain negative values.")
        if start >= end:
            raise ValueError(
                "clip_timestamps must contain strictly increasing start/end pairs."
            )
        if start < previous_end:
            raise ValueError(
                "clip_timestamps pairs must be monotonic and non-overlapping."
            )

        previous_end = end

    return normalized


def get_vad_settings(detector: Optional["VoiceActivityDetector"] = None) -> dict[str, float | int]:
    if detector is None:
        return {
            "threshold": VAD_THRESHOLD,
            "min_speech_duration_ms": VAD_MIN_SPEECH_DURATION_MS,
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
            "max_speech_duration_s": VAD_MAX_SPEECH_DURATION_S,
        }

    return {
        "threshold": detector.threshold,
        "min_speech_duration_ms": detector.min_speech_duration_ms,
        "min_silence_duration_ms": detector.min_silence_duration_ms,
        "speech_pad_ms": detector.speech_pad_ms,
        "max_speech_duration_s": detector.max_speech_duration_s,
    }


def load_vad_report(report_path: str | Path) -> dict[str, Any]:
    file_path = Path(report_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"VAD report not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_clip_timestamps(report: dict[str, Any]) -> list[float]:
    raw_timestamps = report.get("clip_timestamps")
    if not isinstance(raw_timestamps, list):
        raise ValueError("VAD report does not contain a valid 'clip_timestamps' list.")

    return validate_clip_timestamps(raw_timestamps)


class VoiceActivityDetector:

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_speech_duration_ms: int = VAD_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms: int = VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        max_speech_duration_s: float = VAD_MAX_SPEECH_DURATION_S,
        sampling_rate: int = AUDIO_SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.sampling_rate = sampling_rate
        self.model = None

    def _load_model(self) -> None:
        if load_silero_vad is None or get_speech_timestamps is None:
            raise ImportError(
                "silero-vad is not installed. Install project requirements before using VAD."
            )

        if self.model is None:
            logger.info("Loading Silero VAD model...")
            self.model = load_silero_vad()
            logger.info("Silero VAD model loaded successfully.")

    def detect_speech(self, audio_path: str | Path) -> tuple[list[SpeechSegment], float]:
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        self._load_model()
        logger.info("Detecting speech regions in %s...", audio_file.name)

        waveform = torch.from_numpy(whisper.load_audio(str(audio_file)))
        audio_duration = waveform.shape[0] / self.sampling_rate
        raw_segments = get_speech_timestamps(
            waveform,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        segments = [
            SpeechSegment(start=float(item["start"]), end=float(item["end"]))
            for item in raw_segments
            if float(item["end"]) > float(item["start"])
        ]

        logger.info(
            "Detected %d speech segments covering %.2fs of %.2fs audio.",
            len(segments),
            sum(segment.duration for segment in segments),
            audio_duration,
        )
        return segments, audio_duration

    def to_clip_timestamps(self, segments: list[SpeechSegment]) -> list[float]:
        clip_timestamps: list[float] = []
        for segment in segments:
            clip_timestamps.extend([round(segment.start, 3), round(segment.end, 3)])
        return clip_timestamps

    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        audio_file = Path(audio_path).resolve()
        segments, audio_duration = self.detect_speech(audio_file)
        total_speech_duration = sum(segment.duration for segment in segments)
        clip_timestamps = self.to_clip_timestamps(segments)

        return {
            "audio_path": str(audio_file),
            "sampling_rate": self.sampling_rate,
            "settings": get_vad_settings(self),
            "audio_duration": round(audio_duration, 3),
            "speech_duration": round(total_speech_duration, 3),
            "speech_ratio": round(
                total_speech_duration / audio_duration if audio_duration > 0 else 0.0,
                4,
            ),
            "segment_count": len(segments),
            "segments": [
                {
                    **asdict(segment),
                    "duration": round(segment.duration, 3),
                }
                for segment in segments
            ],
            "clip_timestamps": clip_timestamps,
        }

    def save_report(self, report: dict[str, Any], output_path: str | Path) -> Path:
        file_path = Path(output_path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        logger.info("Saved VAD report: %s", file_path)
        return file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect speech regions with Silero VAD and export Whisper clip timestamps."
    )
    parser.add_argument("input", help="Path to the input audio file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the VAD JSON report. Defaults to <input>.vad.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VAD_THRESHOLD,
        help=f"Speech probability threshold (default: {VAD_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=VAD_MIN_SPEECH_DURATION_MS,
        help=(
            "Minimum speech duration in milliseconds "
            f"(default: {VAD_MIN_SPEECH_DURATION_MS})."
        ),
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=VAD_MIN_SILENCE_DURATION_MS,
        help=(
            "Minimum silence duration in milliseconds before splitting segments "
            f"(default: {VAD_MIN_SILENCE_DURATION_MS})."
        ),
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=VAD_SPEECH_PAD_MS,
        help=f"Padding to add around each speech segment (default: {VAD_SPEECH_PAD_MS}).",
    )
    parser.add_argument(
        "--max-speech-duration-s",
        type=float,
        default=VAD_MAX_SPEECH_DURATION_S,
        help=(
            "Maximum speech segment length before Silero forces a split "
            f"(default: {VAD_MAX_SPEECH_DURATION_S})."
        ),
    )
    parser.add_argument(
        "--print-clip-timestamps",
        action="store_true",
        help="Print the flattened Whisper clip_timestamps list after analysis.",
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        output_path = (
            Path(args.output).resolve()
            if args.output
            else input_path.parent / f"{input_path.stem}.vad.json"
        )

        detector = VoiceActivityDetector(
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            min_silence_duration_ms=args.min_silence_duration_ms,
            speech_pad_ms=args.speech_pad_ms,
            max_speech_duration_s=args.max_speech_duration_s,
        )
        report = detector.analyze(input_path)
        detector.save_report(report, output_path)

        if args.print_clip_timestamps:
            print(report["clip_timestamps"])
    except Exception as e:
        logger.error("Error: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
