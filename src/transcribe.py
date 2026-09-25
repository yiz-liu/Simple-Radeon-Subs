"""Transcriber selects Whisper's native batch or Qwen's staged batch.

QwenBatchRunner stays in the parent; QwenWorker owns models in child processes.
QwenASR decodes text, while ForcedAligner supplies timestamps in src.aligner.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never

from src.audio import WindowPreparer, read_audio
from src.config import (
    ASRBackend,
    AUDIO_SAMPLE_RATE,
    PROJECT_ROOT,
    QWEN_ASR_MODEL_PATH,
    QWEN_ALIGNER_MODEL_PATH,
    QWEN_VAD_MODEL_PATH,
    QWEN_GPU_MEMORY_UTILIZATION,
    QWEN_MAX_MODEL_LEN,
    QWEN_MAX_NUM_SEQS,
    QWEN_MAX_BATCHED_TOKENS,
    QWEN_MAX_OUTPUT_TOKENS,
    QWEN_ERROR_TAIL_BYTES,
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_VAD_MODEL_PATH,
)
from src.logger import logger
from src.utils import save_subtitles_atomic
from src.whisper_batch import (
    TranscriptionFailure,
    TranscriptionTask,
    WhisperBatchRunner,
    WhisperRuntime,
    parse_progress,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from vllm import LLM
    from src.aligner import AlignmentRecord, ForcedAligner

type QwenStage = Literal["prepare", "asr", "align"]


__all__ = ["Transcriber", "TranscriptionFailure", "TranscriptionTask", "parse_progress"]


class TranscriptionError(RuntimeError):
    """Report failed Qwen transcription or incomplete subtitle publication."""


class Transcriber:
    """Transcribe audio through a managed local backend."""

    def __init__(self, backend: ASRBackend = "whisper") -> None:
        self.backend: ASRBackend = backend

    def transcribe_many(
        self,
        tasks: tuple[TranscriptionTask, ...],
        language: str | None = None,
        quiet: bool = False,
        enable_vad: bool = True,
    ) -> list[TranscriptionFailure]:
        """Create multiple SRT files while reusing each backend model across the batch."""
        if tasks:
            logger.info("Transcribing %d audio file(s)...", len(tasks))
        match self.backend:
            case "whisper":
                runner = WhisperBatchRunner(
                    WhisperRuntime(
                        cli_path=WHISPER_CLI_PATH,
                        model_path=WHISPER_MODEL_PATH,
                        vad_model_path=WHISPER_VAD_MODEL_PATH,
                    )
                )

            case "qwen":
                runner = QwenBatchRunner()
            case unreachable:
                assert_never(unreachable)
        failures = runner.run(tasks, language, quiet, enable_vad)
        logger.info(
            "Transcription batch complete: %d succeeded, %d failed.",
            len(tasks) - len(failures),
            len(failures),
        )
        return failures

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None,
        language: str | None = None,
        quiet: bool = False,
        enable_vad: bool = True,
    ) -> Path:
        """Create an SRT beside the audio or in the requested output directory."""
        audio_file = Path(audio_path).resolve()
        output_directory = Path(output_dir or audio_file.parent).resolve()
        destination = output_directory / f"{audio_file.stem}.srt"
        task = TranscriptionTask(audio_file, destination)
        failures = self.transcribe_many(
            (task,),
            language=language,
            quiet=quiet,
            enable_vad=enable_vad,
        )
        if failures:
            raise RuntimeError(failures[0].reason)
        return destination


class QwenBatchRunner:
    """Coordinate prepare → ASR → alignment processes, then publish each SRT."""

    def run(
        self,
        tasks: tuple[TranscriptionTask, ...],
        language: str | None,
        quiet: bool,
        enable_vad: bool,
    ) -> list[TranscriptionFailure]:
        if not tasks:
            return []
        language = QwenASR.resolve_language(language)
        self._validate_models(enable_vad)
        failures: list[TranscriptionFailure] = []
        with tempfile.TemporaryDirectory(prefix="qwen-batch-") as scratch:
            pending: dict[TranscriptionTask, Path] = {}
            for index, task in enumerate(tasks):
                try:
                    task.audio_path.resolve(strict=True)
                    directory = Path(scratch) / str(index)
                    directory.mkdir()
                    pending[task] = directory
                except OSError as error:
                    failures.append(TranscriptionFailure(task, str(error)))
            for stage in ("prepare", "asr", "align"):
                if not pending:
                    break
                if not quiet:
                    logger.info("Qwen %s: %d file(s)", stage, len(pending))
                succeeded, failed = self._run_stage_process(
                    pending, stage, language, enable_vad
                )
                pending = dict(succeeded)
                failures.extend(failed)
            for task, directory in pending.items():
                try:
                    self._publish(task, directory)
                except (OSError, ValueError, RuntimeError) as error:
                    logger.error(
                        "Qwen subtitle publication failed for %s: %s",
                        task.audio_path,
                        error,
                    )
                    failures.append(TranscriptionFailure(task, str(error)))
        return failures

    @staticmethod
    def _validate_models(enable_vad: bool) -> None:
        for directory in (QWEN_ASR_MODEL_PATH, QWEN_ALIGNER_MODEL_PATH):
            required = (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            )
            if any(not (directory / name).is_file() for name in required) or not list(
                directory.glob("*.safetensors")
            ):
                raise FileNotFoundError(
                    f"Incomplete Qwen model: {directory}; run scripts/download_weights.sh --asr-backend qwen"
                )
        if enable_vad and not QWEN_VAD_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Silero ONNX model missing: {QWEN_VAD_MODEL_PATH}")

    @staticmethod
    def _run_stage_process(
        pending: Mapping[TranscriptionTask, Path],
        stage: QwenStage,
        language: str | None,
        enable_vad: bool,
    ) -> tuple[Mapping[TranscriptionTask, Path], list[TranscriptionFailure]]:
        from pydantic import TypeAdapter

        inputs = TypeAdapter(list[tuple[Path, Path]]).dump_json(
            [
                (task.audio_path.resolve(), directory)
                for task, directory in pending.items()
            ]
        )
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        with tempfile.TemporaryFile() as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from src.transcribe import QwenWorker; QwenWorker.main()",
                    stage,
                    language or "",
                    "1" if enable_vad else "0",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                input=inputs,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.seek(max(0, handle.tell() - QWEN_ERROR_TAIL_BYTES))
            error_tail = handle.read().decode("utf-8", errors="replace").strip()
        succeeded: dict[TranscriptionTask, Path] = {}
        failures: list[TranscriptionFailure] = []
        for task, directory in pending.items():
            error_path = directory / f"{stage}.error"
            if not error_path.is_file() and (directory / f"{stage}.json").is_file():
                succeeded[task] = directory
                continue
            try:
                reason = error_path.read_text(encoding="utf-8")
            except OSError:
                reason = "Worker did not produce records"
            detail = f"Qwen {stage} failed (exit {completed.returncode}): {reason}"
            if error_tail:
                detail += f"\nWorker output tail:\n{error_tail}"
            logger.error("%s: %s", task.audio_path, detail)
            failures.append(TranscriptionFailure(task, detail))
        return succeeded, failures

    @staticmethod
    def _publish(
        task: TranscriptionTask,
        directory: Path,
    ) -> None:
        import pysrt
        from src.aligner import read_records, render_timed_record

        records = read_records(directory / "align.json")
        errors = [f"{r.segment_id}: {r.error}" for r in records if r.error]
        if errors:
            raise TranscriptionError("; ".join(errors))
        cues = [cue for record in records for cue in render_timed_record(record)]
        subtitles = pysrt.SubRipFile()
        previous_end = 0
        for index, cue in enumerate(cues, 1):
            start, end = round(cue.start * 1000), round(cue.end * 1000)
            if not cue.published or not previous_end <= start < end:
                raise TranscriptionError(
                    f"Invalid subtitle span: {cue.segment_id}/{cue.sentence}"
                )
            previous_end = end
            subtitles.append(
                pysrt.SubRipItem(index, start=start, end=end, text=cue.text.strip())
            )
        if records and not cues:
            raise TranscriptionError("All speech windows returned empty text")
        empty_windows = sum(not record.text for record in records)
        if empty_windows:
            logger.warning(
                "%s: %d speech windows returned empty text",
                task.audio_path,
                empty_windows,
            )
        save_subtitles_atomic(task.output_path, subtitles)


class QwenWorker:
    """Own one stage's model and isolate each input failure inside the child."""

    def __init__(
        self, stage: QwenStage, language: str | None, enable_vad: bool
    ) -> None:
        self.stage: QwenStage = stage
        self.language = language
        self.enable_vad = enable_vad
        self._asr: QwenASR | None = None
        self._aligner: ForcedAligner | None = None
        self._preparer: WindowPreparer | None = None

    def run(self, inputs: list[tuple[Path, Path]]) -> None:
        from src.aligner import (
            read_records,
            write_records,
            needs_alignment,
            merge_alignments,
        )

        for audio_path, directory in inputs:
            try:
                match self.stage:
                    case "prepare":
                        result = self._prepare(audio_path)
                    case "asr":
                        records = read_records(directory / "prepare.json")
                        result = self._transcribe(audio_path, records)
                    case "align":
                        records = read_records(directory / "asr.json")
                        active = [
                            record for record in records if needs_alignment(record)
                        ]
                        result = merge_alignments(
                            records, self._align(audio_path, active)
                        )
                    case unreachable:
                        assert_never(unreachable)
                write_records(directory / f"{self.stage}.json", result)
            except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
                logger.exception("Qwen %s failed for %s", self.stage, audio_path)
                (directory / f"{self.stage}.error").write_text(
                    str(error) or type(error).__name__, encoding="utf-8"
                )

    def _prepare(self, audio_path: Path) -> list[AlignmentRecord]:
        from src.aligner import AlignmentRecord

        if self._preparer is None:
            self._preparer = WindowPreparer(self.enable_vad)
        return [
            AlignmentRecord(
                segment_id=f"w{i:05d}",
                start=span.start,
                end=span.end,
                text="",
                language=self.language,
            )
            for i, span in enumerate(self._preparer.prepare(audio_path))
        ]

    def _transcribe(
        self, audio_path: Path, records: list[AlignmentRecord]
    ) -> list[AlignmentRecord]:
        if not records:
            return []
        audio = self._read_audio(audio_path, records)
        if self._asr is None:
            self._asr = QwenASR(self._load_model("asr"))
        return self._asr.transcribe(records, audio)

    def _align(
        self, audio_path: Path, records: list[AlignmentRecord]
    ) -> list[AlignmentRecord]:
        from src.aligner import ForcedAligner

        if not records:
            return []
        audio = self._read_audio(audio_path, records)
        if self._aligner is None:
            self._aligner = ForcedAligner(self._load_model("align"))
        return self._aligner.align(records, audio)

    @staticmethod
    def _read_audio(
        audio_path: Path, records: list[AlignmentRecord]
    ) -> NDArray[np.float32]:
        audio = read_audio(audio_path)
        if any(not 0 <= r.start < r.end <= len(audio) for r in records):
            raise TranscriptionError(f"Audio bounds mismatch: {audio_path}")
        return audio

    @staticmethod
    def main() -> None:
        from pydantic import TypeAdapter

        inputs = TypeAdapter(list[tuple[Path, Path]]).validate_json(
            sys.stdin.buffer.read()
        )
        stage = TypeAdapter(QwenStage).validate_python(sys.argv[1])
        QwenWorker(stage, sys.argv[2] or None, sys.argv[3] == "1").run(inputs)

    @staticmethod
    def _load_model(stage: Literal["asr", "align"]) -> LLM:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        from vllm import LLM

        is_asr = stage == "asr"
        model = QWEN_ASR_MODEL_PATH if is_asr else QWEN_ALIGNER_MODEL_PATH
        return LLM(
            model=str(model),
            runner="generate" if is_asr else "pooling",
            hf_overrides={}
            if is_asr
            else {"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]},
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=QWEN_GPU_MEMORY_UTILIZATION,
            max_model_len=QWEN_MAX_MODEL_LEN,
            max_num_seqs=QWEN_MAX_NUM_SEQS,
            max_num_batched_tokens=QWEN_MAX_BATCHED_TOKENS,
            limit_mm_per_prompt={"audio": 1},
        )


class QwenASR:
    """Decode window text using a model owned by the current QwenWorker."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def transcribe(
        self,
        records: list[AlignmentRecord],
        audio: NDArray[np.float32],
    ) -> list[AlignmentRecord]:
        from vllm import SamplingParams
        from vllm.inputs import TextPrompt

        prompts: list[TextPrompt] = []
        for record in records:
            language = (
                f"language {record.language}<asr_text>" if record.language else ""
            )
            prompts.append(
                {
                    "prompt": "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                    + "<|im_end|>\n<|im_start|>assistant\n"
                    + language,
                    "multi_modal_data": {
                        "audio": (audio[record.start : record.end], AUDIO_SAMPLE_RATE)
                    },
                }
            )
        outputs = self.llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=QWEN_MAX_OUTPUT_TOKENS),
            use_tqdm=False,
        )
        result: list[AlignmentRecord] = []
        for record, output in zip(records, outputs, strict=True):
            completion = output.outputs[0]
            language, text = self.parse_output(completion.text, record.language)
            if text and language is None:
                raise TranscriptionError(
                    "ASR returned text without a detected language"
                )
            result.append(
                record.model_copy(
                    update={
                        "text": text,
                        "language": language,
                        "error": "asr_truncated"
                        if completion.finish_reason == "length"
                        else None,
                    }
                )
            )
        return result

    _LANGUAGES = {
        "yue": "Cantonese",
        "zh": "Chinese",
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
    }

    @staticmethod
    def resolve_language(language: str | None) -> str | None:
        if language is None:
            return None
        for code, name in QwenASR._LANGUAGES.items():
            if language.lower() in (code, name.lower()):
                return name
        raise TranscriptionError(
            f"Qwen subtitle backend does not support {language!r}; use one of {', '.join(QwenASR._LANGUAGES)}"
        )

    @staticmethod
    def parse_output(
        output: str, forced_language: str | None
    ) -> tuple[str | None, str]:
        prefix, marker, text = output.partition("<asr_text>")
        if not marker:
            return forced_language, output.strip()
        text = text.strip()
        language = prefix.removeprefix("language ").strip()
        return QwenASR.resolve_language(language) if text else forced_language, text


class _Arguments(argparse.Namespace):
    input: str = ""
    output_dir: str | None = None
    language: str | None = None
    quiet: bool = False
    enable_vad: bool = True
    asr_backend: ASRBackend = "whisper"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using a managed local backend."
    )
    _ = parser.add_argument(
        "--asr-backend",
        choices=("whisper", "qwen"),
        default="whisper",
        help="Transcription backend (default: whisper).",
    )
    _ = parser.add_argument("input", help="Path to the input audio file.")
    _ = parser.add_argument(
        "-o", "--output-dir", help="Directory to save the SRT file."
    )
    _ = parser.add_argument(
        "-l",
        "--language",
        help="Source language code. Auto-detects if omitted.",
    )
    _ = parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable transcription progress output.",
    )
    _ = parser.add_argument(
        "--disable-vad",
        action="store_false",
        dest="enable_vad",
        help="Disable integrated VAD and transcribe the complete audio stream.",
    )
    arguments = parser.parse_args(namespace=_Arguments())
    try:
        _ = Transcriber(arguments.asr_backend).transcribe(
            arguments.input,
            output_dir=arguments.output_dir,
            language=arguments.language,
            quiet=arguments.quiet,
            enable_vad=arguments.enable_vad,
        )
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Transcription failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
