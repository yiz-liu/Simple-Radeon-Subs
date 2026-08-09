import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tqdm import tqdm

from src.audio import AudioExtractor
from src.clean import clean_srt
from src.logger import logger
from src.transcribe import Transcriber, TranscriptionTask
from src.translation import (
    TranslationOptions,
    TranslationTask,
    VLLMTranslator,
)

SAFE_LABEL_PATTERN: Final = re.compile(r"[^\w.-]+", flags=re.UNICODE)
WORK_DIR_SUFFIX: Final = ".simple-radeon-subs"


class InvalidLanguageLabelError(ValueError):
    """Raised when a language label cannot form a safe output filename."""


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Options shared by every job in one pipeline run."""

    target_lang: str = "Chinese"
    src_lang: str | None = None
    keep_temp: bool = False
    force: bool = False
    translated_only: bool = False
    enable_vad: bool = True


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Resolved input collection and output layout."""

    input_paths: tuple[Path, ...]
    input_root: Path
    output_dir: Path | None
    options: PipelineOptions


@dataclass(frozen=True, slots=True)
class PipelineJob:
    """All source, checkpoint, and final paths for one media file."""

    input_path: Path
    work_dir: Path
    audio_path: Path
    raw_srt_path: Path
    cleaned_srt_path: Path
    final_srt_path: Path


@dataclass(frozen=True, slots=True)
class JobFailure:
    """A failed pipeline stage scoped to one media job."""

    job: PipelineJob
    stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Terminal outcomes for one batch invocation."""

    succeeded: tuple[PipelineJob, ...]
    skipped: tuple[PipelineJob, ...]
    failures: tuple[JobFailure, ...]


def _safe_label(value: str) -> str:
    label = SAFE_LABEL_PATTERN.sub("_", value).strip("._")
    if not label:
        raise InvalidLanguageLabelError(
            "Language must contain a filename-safe character"
        )
    return label


def build_jobs(plan: PipelinePlan) -> tuple[PipelineJob, ...]:
    """Derive collision-safe sidecars and relative final output paths."""
    language_label = _safe_label(plan.options.target_lang)
    source_label = _safe_label(plan.options.src_lang or "auto")
    vad_label = "vad" if plan.options.enable_vad else "no-vad"
    jobs: list[PipelineJob] = []
    for input_path in plan.input_paths:
        source = input_path.resolve()
        relative_parent = source.relative_to(plan.input_root.resolve()).parent
        output_parent = (
            source.parent
            if plan.output_dir is None
            else plan.output_dir.resolve() / relative_parent
        )
        work_dir = source.parent / f".{source.name}{WORK_DIR_SUFFIX}"
        asr_dir = work_dir / f"asr-{source_label}-{vad_label}"
        jobs.append(
            PipelineJob(
                input_path=source,
                work_dir=work_dir,
                audio_path=work_dir / "audio.wav",
                raw_srt_path=asr_dir / "raw.srt",
                cleaned_srt_path=asr_dir / "cleaned.srt",
                final_srt_path=output_parent / f"{source.stem}.{language_label}.srt",
            )
        )
    return tuple(jobs)


def _reusable(output_path: Path, input_path: Path) -> bool:
    return (
        output_path.is_file()
        and output_path.stat().st_mtime_ns >= input_path.stat().st_mtime_ns
    )


def run_pipeline(
    jobs: tuple[PipelineJob, ...],
    options: PipelineOptions,
) -> PipelineResult:
    """Run extraction, ASR, cleanup, and translation as collection-wide stages."""
    skipped = tuple(
        job for job in jobs if job.final_srt_path.is_file() and not options.force
    )
    active = [job for job in jobs if job not in skipped]
    failures: dict[Path, JobFailure] = {}

    if options.force:
        for job in active:
            if job.work_dir.is_dir():
                shutil.rmtree(job.work_dir)

    extraction_jobs = [
        job
        for job in active
        if not _reusable(job.audio_path, job.input_path)
        and not _reusable(job.raw_srt_path, job.input_path)
        and not _reusable(job.cleaned_srt_path, job.input_path)
    ]
    if extraction_jobs:
        extractor = AudioExtractor()
        with tqdm(
            total=len(extraction_jobs),
            desc="Extracting audio",
            unit="file",
            position=0,
        ) as extraction_progress:
            for job in extraction_jobs:
                try:
                    job.audio_path.parent.mkdir(parents=True, exist_ok=True)
                    _ = extractor.extract(
                        job.input_path,
                        job.audio_path,
                        force=True,
                        progress_position=1,
                    )
                except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
                    failures[job.input_path] = JobFailure(job, "extract", str(error))
                finally:
                    extraction_progress.update()

    transcription_jobs = [
        job
        for job in active
        if job.input_path not in failures
        and not _reusable(job.raw_srt_path, job.input_path)
        and not _reusable(job.cleaned_srt_path, job.input_path)
    ]
    if transcription_jobs:
        try:
            transcription_failures = Transcriber().transcribe_many(
                tuple(
                    TranscriptionTask(job.audio_path, job.raw_srt_path)
                    for job in transcription_jobs
                ),
                language=options.src_lang,
                enable_vad=options.enable_vad,
            )
            jobs_by_output = {job.raw_srt_path: job for job in transcription_jobs}
            for failure in transcription_failures:
                job = jobs_by_output[failure.task.output_path]
                failures[job.input_path] = JobFailure(
                    job,
                    "transcribe",
                    failure.reason,
                )
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            for job in transcription_jobs:
                failures[job.input_path] = JobFailure(job, "transcribe", str(error))

    cleaning_jobs = [
        job
        for job in active
        if job.input_path not in failures
        and not _reusable(job.cleaned_srt_path, job.raw_srt_path)
    ]
    for job in cleaning_jobs:
        try:
            clean_srt(
                job.raw_srt_path,
                job.cleaned_srt_path,
                enable_vad=options.enable_vad,
            )
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            failures[job.input_path] = JobFailure(job, "clean", str(error))

    translation_jobs = [job for job in active if job.input_path not in failures]
    if translation_jobs:
        try:
            translator = VLLMTranslator()
            translation_failures = translator.translate_many(
                tuple(
                    TranslationTask(job.cleaned_srt_path, job.final_srt_path)
                    for job in translation_jobs
                ),
                TranslationOptions(
                    target_lang=options.target_lang,
                    translated_only=options.translated_only,
                ),
            )
            jobs_by_input = {job.cleaned_srt_path: job for job in translation_jobs}
            for failure in translation_failures:
                job = jobs_by_input[failure.task.input_path]
                failures[job.input_path] = JobFailure(
                    job,
                    "translate",
                    failure.reason,
                )
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            for job in translation_jobs:
                failures[job.input_path] = JobFailure(job, "translate", str(error))

    succeeded = tuple(job for job in active if job.input_path not in failures)
    if not options.keep_temp:
        for job in succeeded:
            if job.work_dir.is_dir():
                shutil.rmtree(job.work_dir)

    for failure in failures.values():
        logger.error(
            "Pipeline stage %s failed for %s: %s",
            failure.stage,
            failure.job.input_path,
            failure.reason,
        )
    return PipelineResult(succeeded, skipped, tuple(failures.values()))
